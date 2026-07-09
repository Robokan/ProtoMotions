# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the battle task's pure components: hit-energy FSM,
observation kernel, and reward kernels."""

import math

import pytest
import torch

from protomotions.envs.battle.hit_state import (
    BattleHitState,
    HitStateConfig,
    resolve_body_ids,
)
from protomotions.envs.battle.obs import compute_battle_task_obs
from protomotions.envs.battle.rewards import (
    compute_arena_boundary_penalty,
    compute_facing_reward,
    compute_idle_penalty,
    compute_range_reward,
    compute_win_reward,
)


# ---------- hit-state FSM ------------------------------------------------


def _make_hit_state(num_envs=2, num_bodies=4, proximity_radius=0.5):
    damage_ids = torch.tensor([0, 1])
    strike_ids = torch.tensor([2, 3])
    return BattleHitState(
        num_envs=num_envs,
        damage_body_ids=damage_ids,
        strike_body_ids=strike_ids,
        damage_multipliers=torch.tensor([2.0, 1.0]),
        config=HitStateConfig(proximity_radius=proximity_radius, warmup_steps=0),
        dt=0.02,
        device=torch.device("cpu"),
    )


def _hit_inputs(num_envs=2, num_bodies=4, force=100.0, close=True, closing_speed=2.0):
    contact_forces = torch.zeros(num_envs, num_bodies, 3)
    contact_forces[:, 0, 0] = force  # force on damage body 0

    body_pos = torch.zeros(num_envs, num_bodies, 3)
    body_pos[:, 1, 2] = 0.5
    # Opponent strike body 2 near/far from damage body 0
    opp_body_pos = torch.zeros(num_envs, num_bodies, 3)
    opp_body_pos[:, 2, 0] = 0.1 if close else 5.0
    opp_body_pos[:, 3, 1] = 3.0

    body_vel = torch.zeros(num_envs, num_bodies, 3)
    opp_body_vel = torch.zeros(num_envs, num_bodies, 3)
    # Striker closing along +x (the contact normal direction)
    opp_body_vel[:, 2, 0] = closing_speed

    progress = torch.full((num_envs,), 100, dtype=torch.long)
    return contact_forces, body_pos, body_vel, opp_body_pos, opp_body_vel, progress


def test_hit_state_scores_proximal_gated_contact():
    hs = _make_hit_state()
    taken = hs.step(*_hit_inputs(close=True))
    assert taken.shape == (2,)
    assert (taken > 0).all(), "forceful, proximal, closing contact must score"


def test_hit_state_ignores_contact_without_nearby_striker():
    """Ground contact attribution: force with no opponent nearby scores zero."""
    hs = _make_hit_state()
    taken = hs.step(*_hit_inputs(close=False))
    assert (taken == 0).all()


def test_hit_state_requires_closing_velocity():
    """Pushing (no closing speed) must not accumulate hit energy."""
    hs = _make_hit_state()
    taken = hs.step(*_hit_inputs(closing_speed=0.0))
    assert (taken == 0).all()


def test_hit_state_warmup_gates_early_steps():
    hs = _make_hit_state()
    hs.config = HitStateConfig(proximity_radius=0.5, warmup_steps=10)
    inputs = list(_hit_inputs(close=True))
    inputs[5] = torch.full((2,), 3, dtype=torch.long)  # progress < warmup
    taken = hs.step(*inputs)
    assert (taken == 0).all()


def test_hit_state_reset_clears_accumulators():
    hs = _make_hit_state()
    hs.step(*_hit_inputs())
    hs.reset(torch.tensor([0, 1]))
    assert (hs._e_accum == 0).all()
    assert not hs._active.any()


def test_resolve_body_ids_rejects_unknown_names():
    with pytest.raises(ValueError, match="not found"):
        resolve_body_ids(["Nope"], ["Head", "Chest"])
    ids = resolve_body_ids(["Chest", "Head"], ["Head", "Chest"])
    assert ids.tolist() == [1, 0]


# ---------- observation kernel -------------------------------------------


def _identity_quat(n):
    q = torch.zeros(n, 4)
    q[:, 3] = 1.0  # w-last
    return q


def test_facing_reward_is_gaze_based():
    """Facing must follow the HEAD, not the root: a fighter whose head is
    turned away scores low even if the body would face the opponent."""
    n = 1
    head_pos = torch.zeros(n, 3)
    opp_head = torch.zeros(n, 3)
    opp_head[:, 0] = 2.0  # opponent head along +x

    looking_at = compute_facing_reward(head_pos, _identity_quat(n), opp_head)
    assert looking_at[0] == pytest.approx(1.0, abs=1e-5)

    # Head yawed 180 degrees: looking directly away
    turned = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    looking_away = compute_facing_reward(head_pos, turned, opp_head)
    assert looking_away[0] == pytest.approx(0.0, abs=1e-5)

    # 90 degrees off: neutral 0.5
    half = torch.tensor([[0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]])
    sideways = compute_facing_reward(head_pos, half, opp_head)
    assert sideways[0] == pytest.approx(0.5, abs=1e-5)


def test_battle_task_obs_shape_and_clamps():
    n, k = 4, 5
    obs = compute_battle_task_obs(
        root_pos=torch.zeros(n, 3),
        root_rot=_identity_quat(n),
        opp_root_pos=torch.full((n, 3), 100.0),  # far away -> clamped
        opp_root_rot=_identity_quat(n),
        opp_root_vel=torch.full((n, 3), 500.0),  # spiking -> clamped
        opp_root_ang_vel=torch.zeros(n, 3),
        opp_key_body_pos=torch.zeros(n, k, 3),
        opp_key_body_vel=torch.zeros(n, k, 3),
        health=torch.ones(n),
        opp_health=torch.ones(n),
        downed=torch.zeros(n),
        opp_downed=torch.zeros(n),
        round_time_left=torch.ones(n),
        arena_center=torch.zeros(n, 2),
        arena_half_size=3.5,
    )
    expected_dim = 3 + 6 + 3 + 3 + 3 * k + 3 * k + 2 + 5
    assert obs.shape == (n, expected_dim)
    assert obs.abs().max() <= 50.0, "all channels must respect their clamps"
    # Opponent-relative position channels clamp at pos_clamp (14.0)
    assert obs[:, :3].abs().max() <= 14.0


def test_battle_task_obs_heading_frame_invariance():
    """An opponent 2m ahead reads identically for any ego yaw."""
    n, k = 1, 1

    def build(yaw):
        axis_angle_half = yaw / 2.0
        root_rot = torch.tensor(
            [[0.0, 0.0, math.sin(axis_angle_half), math.cos(axis_angle_half)]]
        )
        forward = torch.tensor([[math.cos(yaw), math.sin(yaw), 0.0]])
        return compute_battle_task_obs(
            root_pos=torch.zeros(n, 3),
            root_rot=root_rot,
            opp_root_pos=forward * 2.0,
            opp_root_rot=root_rot.clone(),
            opp_root_vel=torch.zeros(n, 3),
            opp_root_ang_vel=torch.zeros(n, 3),
            opp_key_body_pos=(forward * 2.0).unsqueeze(1),
            opp_key_body_vel=torch.zeros(n, k, 3),
            health=torch.ones(n),
            opp_health=torch.ones(n),
            downed=torch.zeros(n),
            opp_downed=torch.zeros(n),
            round_time_left=torch.ones(n),
            arena_center=torch.zeros(n, 2),
            arena_half_size=3.5,
        )

    obs_0 = build(0.0)
    obs_90 = build(math.pi / 2)
    assert torch.allclose(obs_0, obs_90, atol=1e-5)


# ---------- reward kernels -------------------------------------------------


def test_win_reward_passthrough():
    signal = torch.tensor([1.0, -1.0, 0.0])
    assert torch.equal(compute_win_reward(signal), signal)


def test_facing_reward_bounds():
    n = 2
    head_pos = torch.zeros(n, 3)
    opp_head = torch.zeros(n, 3)
    opp_head[:, 0] = 2.0  # opponent head along +x
    facing = compute_facing_reward(head_pos, _identity_quat(n), opp_head)
    assert torch.allclose(facing, torch.ones(n), atol=1e-5)

    # Looking away: yaw pi
    head_rot = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(n, 1)
    facing_away = compute_facing_reward(head_pos, head_rot, opp_head)
    assert torch.allclose(facing_away, torch.zeros(n), atol=1e-5)


def test_range_reward_backs_away_from_downed_opponent():
    n = 1
    root_pos = torch.zeros(n, 3)
    root_vel = torch.zeros(n, 3)
    root_vel[:, 0] = 1.0
    opp = torch.zeros(n, 3)
    opp[:, 0] = 3.0

    active = compute_range_reward(
        root_pos, root_vel, opp, opp_downed=torch.zeros(n)
    )
    downed = compute_range_reward(
        root_pos, root_vel, opp, opp_downed=torch.ones(n)
    )
    assert active[0] > 0  # closing on a standing opponent is rewarded
    assert downed[0] == pytest.approx(1.0)  # at back_away_distance exactly


def test_idle_penalty_engages_past_half():
    idle = torch.tensor([0.0, 0.4, 0.75, 1.0])
    penalty = compute_idle_penalty(idle)
    assert penalty[0] == 0 and penalty[1] == 0
    assert penalty[2] == pytest.approx(-0.5)
    assert penalty[3] == pytest.approx(-1.0)


def test_arena_boundary_penalty_ramps():
    n = 3
    pos = torch.zeros(n, 3)
    pos[0, 0] = 0.0  # center
    pos[1, 0] = 3.0  # 0.5m from the wall of a 7m arena
    pos[2, 0] = 3.5  # at the wall
    penalty = compute_arena_boundary_penalty(
        pos, torch.zeros(n, 2), arena_half_size=3.5, margin=1.0
    )
    assert penalty[0] == 0.0
    assert penalty[1] == pytest.approx(-0.5)
    assert penalty[2] == pytest.approx(-1.0)
