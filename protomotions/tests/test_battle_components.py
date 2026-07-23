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
    taken, _, _, _ = hs.step(*_hit_inputs(close=True))
    assert taken.shape == (2,)
    assert (taken > 0).all(), "forceful, proximal, closing contact must score"


def test_hit_state_ignores_contact_without_nearby_striker():
    """Ground contact attribution: force with no opponent nearby scores zero."""
    hs = _make_hit_state()
    taken, _, _, _ = hs.step(*_hit_inputs(close=False))
    assert (taken == 0).all()


def test_hit_state_requires_closing_velocity():
    """Pushing (no closing speed) must not accumulate hit energy."""
    hs = _make_hit_state()
    taken, _, _, _ = hs.step(*_hit_inputs(closing_speed=0.0))
    assert (taken == 0).all()


def test_hit_state_warmup_gates_early_steps():
    hs = _make_hit_state()
    hs.config = HitStateConfig(proximity_radius=0.5, warmup_steps=10)
    inputs = list(_hit_inputs(close=True))
    inputs[5] = torch.full((2,), 3, dtype=torch.long)  # progress < warmup
    taken, _, _, _ = hs.step(*inputs)
    assert (taken == 0).all()


def test_hit_state_reset_clears_accumulators():
    hs = _make_hit_state()
    hs.step(*_hit_inputs())
    hs.reset(torch.tensor([0, 1]))
    assert (hs._e_accum == 0).all()
    assert not hs._active.any()


def test_ke_damage_gates_on_impact_speed():
    """HEALTH damage (4th return) is per-hit kinetic energy: a fast strike
    scores 0.5*m*v^2 at contact onset; a slow push scores exactly zero."""
    cfg = HitStateConfig(proximity_radius=0.5, warmup_steps=0, strike_min_speed=2.0)

    hs = _make_hit_state()
    hs.config = cfg
    _, _, _, ke_strike = hs.step(*_hit_inputs(force=100.0, closing_speed=4.0))
    # unit mass -> KE = 0.5 * 1.0 * 4^2 = 8 J on the contacted body
    assert torch.allclose(
        ke_strike.sum(dim=-1), torch.full((2,), 8.0)
    ), "a qualifying strike must deposit 0.5*m*v^2"

    hs = _make_hit_state()
    hs.config = cfg
    _, _, _, ke_push = hs.step(*_hit_inputs(force=500.0, closing_speed=0.5))
    assert (ke_push == 0).all(), "a slow push must deal ZERO HP however forceful"


def test_ke_reward_is_continuous_and_ungated():
    """In KE-reward mode the dense reward (1st return) pays continuously —
    a sub-gate tap earns a small positive guide, a faster hit earns more —
    while HEALTH (4th return) stays speed-gated to zero for the tap."""
    cfg = HitStateConfig(proximity_radius=0.5, warmup_steps=0, strike_min_speed=2.0)

    hs = _make_hit_state()
    hs.config = cfg
    hs.reward_from_event_ke = True
    r_tap, _, _, ke_tap = hs.step(*_hit_inputs(force=100.0, closing_speed=1.0))
    assert (r_tap > 0).all(), "a tap must still earn a small positive reward"
    assert (ke_tap == 0).all(), "but a tap deals zero HEALTH damage"

    hs = _make_hit_state()
    hs.config = cfg
    hs.reward_from_event_ke = True
    r_hit, _, _, ke_hit = hs.step(*_hit_inputs(force=100.0, closing_speed=4.0))
    assert (r_hit > r_tap).all(), "reward must grow with impact speed"
    assert (ke_hit.sum(dim=-1) > 0).all(), "a qualifying strike damages health"


def test_ke_reward_hit_flat_adds_once_per_onset():
    """hit_flat is added once per env on contact onset, on top of log1p(KE)."""
    cfg = HitStateConfig(
        proximity_radius=0.5,
        warmup_steps=0,
        strike_min_speed=2.0,
        ke_reward_ref=5.0,
        hit_flat=0.05,
    )
    hs = _make_hit_state()
    hs.config = cfg
    hs.reward_from_event_ke = True
    r, _, _, _ = hs.step(*_hit_inputs(force=100.0, closing_speed=1.0))
    # tap KE = 0.5*1*1^2 = 0.5 J on damage body 0 (mult 2.0), then + flat.
    expected = 2.0 * torch.log1p(torch.tensor(0.5 / 5.0)) + 0.05
    assert torch.allclose(r, expected.expand_as(r), atol=1e-4)


def test_ke_damage_deposits_once_per_contact_event():
    """Lingering contact must not re-score: KE deposits only on the FSM
    rising edge, so step 2 of the same contact adds nothing."""
    cfg = HitStateConfig(proximity_radius=0.5, warmup_steps=0, strike_min_speed=2.0)
    hs = _make_hit_state()
    hs.config = cfg
    inputs = _hit_inputs(force=100.0, closing_speed=4.0)
    _, _, _, first = hs.step(*inputs)
    assert (first.sum(dim=-1) > 0).all()
    _, _, _, second = hs.step(*inputs)  # same sustained contact
    assert (second == 0).all(), "sustained contact must not deposit again"


def test_ke_damage_scales_with_speed_squared_and_mass():
    """KE = 0.5*m*v^2: doubling impact speed quadruples damage; heavier
    striking limbs (legs) hit harder at the same speed."""
    cfg = HitStateConfig(proximity_radius=0.5, warmup_steps=0, strike_min_speed=2.0)

    hs = _make_hit_state()
    hs.config = cfg
    _, _, _, slow = hs.step(*_hit_inputs(force=100.0, closing_speed=3.0))

    hs = _make_hit_state()
    hs.config = cfg
    _, _, _, fast = hs.step(*_hit_inputs(force=100.0, closing_speed=6.0))
    ratio = (fast[0].sum() / slow[0].sum()).item()
    assert abs(ratio - 4.0) < 1e-3, f"expected 4x from 2x speed, got {ratio:.3f}x"

    hs = _make_hit_state()
    hs.config = cfg
    hs.set_strike_body_masses(torch.tensor([3.0, 3.0]))  # heavier limbs
    _, _, _, heavy = hs.step(*_hit_inputs(force=100.0, closing_speed=3.0))
    ratio_m = (heavy[0].sum() / slow[0].sum()).item()
    assert abs(ratio_m - 3.0) < 1e-3, f"expected 3x from 3x mass, got {ratio_m:.3f}x"


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

    looking_at = compute_facing_reward(head_pos, _identity_quat(n), opp_head, forward_axis=(1.0, 0.0, 0.0))
    assert looking_at[0] == pytest.approx(1.0, abs=1e-5)

    # Head yawed 180 degrees: looking directly away
    turned = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    looking_away = compute_facing_reward(head_pos, turned, opp_head, forward_axis=(1.0, 0.0, 0.0))
    assert looking_away[0] == pytest.approx(0.0, abs=1e-5)

    # 90 degrees off: neutral 0.5
    half = torch.tensor([[0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]])
    sideways = compute_facing_reward(head_pos, half, opp_head, forward_axis=(1.0, 0.0, 0.0))
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
    facing = compute_facing_reward(head_pos, _identity_quat(n), opp_head, forward_axis=(1.0, 0.0, 0.0))
    assert torch.allclose(facing, torch.ones(n), atol=1e-5)

    # Looking away: yaw pi
    head_rot = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(n, 1)
    facing_away = compute_facing_reward(head_pos, head_rot, opp_head, forward_axis=(1.0, 0.0, 0.0))
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


# ---------- strike groups & kickboxing diversity ---------------------------


def _make_grouped_hit_state(num_envs=2):
    """Strike body 2 = 'hands' (group 0), strike body 3 = 'legs' (group 1)."""
    return BattleHitState(
        num_envs=num_envs,
        damage_body_ids=torch.tensor([0, 1]),
        strike_body_ids=torch.tensor([2, 3]),
        damage_multipliers=torch.tensor([2.0, 1.0]),
        config=HitStateConfig(proximity_radius=0.5, warmup_steps=0),
        dt=0.02,
        device=torch.device("cpu"),
        strike_body_groups=torch.tensor([0, 1]),
        num_strike_groups=2,
    )


def test_hit_energy_attributed_to_striker_group():
    hs = _make_grouped_hit_state()
    # Hand striker (body 2) close to damage body 0; leg striker far away
    inputs = _hit_inputs(close=True)
    taken, by_group, _, _ = hs.step(*inputs)
    assert by_group.shape == (2, 2)
    assert (by_group[:, 0] > 0).all(), "hand-group strike must be attributed"
    assert (by_group[:, 1] == 0).all(), "leg group dealt nothing"
    assert torch.allclose(by_group.sum(dim=-1), taken, atol=1e-6)


def test_hit_energy_attributed_to_leg_group():
    hs = _make_grouped_hit_state()
    (cf, bp, bv, opp_pos, opp_vel, prog) = _hit_inputs(close=False)
    # Move the LEG striker (body 3) close and closing; hand striker far
    opp_pos[:, 3, :] = 0.0
    opp_pos[:, 3, 0] = 0.1
    opp_vel[:, 3, 0] = 2.0
    taken, by_group, _, _ = hs.step(cf, bp, bv, opp_pos, opp_vel, prog)
    assert (by_group[:, 1] > 0).all()
    assert (by_group[:, 0] == 0).all()


def test_diversity_bonus_pays_only_for_lesser_group_growth():
    """Replicates BattleControl's min-growth accounting: hand-only damage
    stops earning once hands lead; leg damage then pays."""
    cum = torch.zeros(1, 2)

    def step(dealt):
        nonlocal cum
        prev_min = cum.min(dim=-1).values
        cum = cum + torch.tensor([dealt])
        return (cum.min(dim=-1).values - prev_min).clamp_min(0.0)

    assert step([1.0, 0.0])[0] == 0.0  # first punch: hands lead, min unchanged
    assert step([1.0, 0.0])[0] == 0.0  # more punching earns no diversity
    assert step([0.0, 1.5])[0] == pytest.approx(1.5)  # kicks catch up: paid
    assert step([0.0, 1.0])[0] == pytest.approx(0.5)  # paid until legs pass hands


def test_default_reward_set_is_simple_kickboxing():
    from protomotions.envs.battle.factories import default_battle_reward_components

    components = default_battle_reward_components()
    assert "battle_win" in components
    assert "battle_facing" in components
    assert "battle_hit" in components
    assert "battle_strike_diversity" in components
    assert "battle_range" not in components, "approach shaping was dropped"
    # Annealing zeroes every dense term but never the win signal
    annealed = default_battle_reward_components(dense_scale=0.0)
    assert annealed["battle_win"].static_params["weight"] > 0
    assert annealed["battle_hit"].static_params["weight"] == 0


def test_facing_default_axis_is_soma_minus_y():
    """SOMA faces body-frame -y: identity head rotation with the opponent
    along -y must read as looking straight at them under the default axis."""
    n = 1
    head_pos = torch.zeros(n, 3)
    opp_head = torch.zeros(n, 3)
    opp_head[:, 1] = -2.0  # opponent along -y = in front of a SOMA T-pose
    facing = compute_facing_reward(head_pos, _identity_quat(n), opp_head)
    assert facing[0] == pytest.approx(1.0, abs=1e-5)

    # +x (the calc_heading convention) is exactly sideways: neutral 0.5
    opp_side = torch.zeros(n, 3)
    opp_side[:, 0] = 2.0
    sideways = compute_facing_reward(head_pos, _identity_quat(n), opp_side)
    assert sideways[0] == pytest.approx(0.5, abs=1e-5)
