# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Impulse damage model: the structural guarantees, not the numerics.

The four properties that make the model safe to train on:

1. one contact onset -> exactly ONE impulse deposit (window-bounded);
2. a sustained push NEVER re-scores while held — no per-step accrual, which
   is the exploit that killed the old F*v energy model (guard-grinding
   drained 100% health in 3 s);
3. the reward equals log1p(J / impulse_reward_ref) for that single deposit;
4. after force release AND cooldown, a new onset scores again.

Constructed on CPU with hand-built tensors: 2 envs, 2 bodies (damage body 0,
strike body 1), the striker hovering 10 cm from the damage body so the
proximity gate passes, closing at 1 m/s so v_gate passes.
"""

import math

import pytest
import torch

from protomotions.envs.battle.hit_state import BattleHitState, HitStateConfig

DT = 1.0 / 15.0  # battle control step (physics 60 Hz, decimation 4)
N = 2  # paired envs


def _make(force_on_body0: float = 0.0, **cfg_kw):
    cfg = HitStateConfig(warmup_steps=0, **cfg_kw)
    hs = BattleHitState(
        num_envs=N,
        damage_body_ids=torch.tensor([0]),
        strike_body_ids=torch.tensor([1]),
        damage_multipliers=torch.tensor([1.0]),
        config=cfg,
        dt=DT,
        device=torch.device("cpu"),
        reward_from_event_impulse=True,
    )
    return hs, cfg


def _step(hs, force_n: float):
    """One FSM step with `force_n` newtons on the damage body of env 0."""
    forces = torch.zeros(N, 2, 3)
    forces[0, 0, 1] = force_n  # +y force on env0's damage body
    body_pos = torch.zeros(N, 2, 3)
    body_pos[:, 1, 1] = 5.0  # own strike body: far away, irrelevant
    opp_pos = torch.zeros(N, 2, 3)
    opp_pos[:, 1, 1] = 0.10  # opponent strike body 10 cm away (< 0.35 gate)
    opp_vel = torch.zeros(N, 2, 3)
    opp_vel[:, 1, 1] = 1.0  # closing along the contact normal (+y)
    return hs.step(
        contact_forces=forces,
        body_pos=body_pos,
        body_vel=torch.zeros(N, 2, 3),
        opp_body_pos=opp_pos,
        opp_body_vel=opp_vel,
        progress=torch.full((N,), 100, dtype=torch.long),
    )


def test_one_onset_one_deposit_then_push_scores_nothing():
    hs, cfg = _make()
    window_steps = hs._steps_impulse
    assert window_steps == max(1, round(cfg.impulse_window / DT)) == 1

    # onset: 120 N -> window opens and (window = 1 step) closes same step
    r, _, _, dmg = _step(hs, 120.0)
    j_expected = 120.0 * DT  # 8 N.s
    assert dmg[0, 0].item() == pytest.approx(j_expected, rel=1e-5)
    assert r[0].item() == pytest.approx(
        math.log1p(j_expected / cfg.impulse_reward_ref), rel=1e-5
    )
    assert dmg[1, 0].item() == 0.0  # untouched env stays clean

    # the push: same 120 N held for 3 seconds -> not one more joule-second
    for _ in range(int(3.0 / DT)):
        r, _, _, dmg = _step(hs, 120.0)
        assert dmg[0, 0].item() == 0.0
        assert r[0].item() == 0.0


def test_rearm_requires_release_and_cooldown():
    hs, cfg = _make()
    _step(hs, 120.0)  # first hit
    _step(hs, 120.0)  # still held: nothing (verified above)

    # release below force_off, wait out the cooldown
    cooldown_steps = hs._steps_cool
    for _ in range(cooldown_steps + 1):
        r, _, _, dmg = _step(hs, 0.0)
        assert dmg[0, 0].item() == 0.0

    # second strike scores a fresh deposit
    r, _, _, dmg = _step(hs, 60.0)
    assert dmg[0, 0].item() == pytest.approx(60.0 * DT, rel=1e-5)
    assert r[0].item() > 0.0


def test_release_without_cooldown_does_not_rearm():
    hs, _ = _make()
    _step(hs, 120.0)
    # drop and immediately re-apply: cooldown (0.15 s = 2-3 steps) blocks it
    _step(hs, 0.0)
    r, _, _, dmg = _step(hs, 120.0)
    assert dmg[0, 0].item() == 0.0
    assert r[0].item() == 0.0


def test_modes_are_mutually_exclusive():
    with pytest.raises(ValueError):
        BattleHitState(
            num_envs=N,
            damage_body_ids=torch.tensor([0]),
            strike_body_ids=torch.tensor([1]),
            damage_multipliers=torch.tensor([1.0]),
            config=HitStateConfig(),
            dt=DT,
            device=torch.device("cpu"),
            reward_from_event_ke=True,
            reward_from_event_impulse=True,
        )
