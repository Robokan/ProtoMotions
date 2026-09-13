# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Shape of the backward-locomotion task reward.

The first implementation used a Gaussian centred on -target_speed. Wide enough
to accept the corpus spread (-0.13..-0.43 m/s), it also paid 0.78 out of 1.0
for standing perfectly still -- so freezing was nearly as good as walking. The
saturating ramp below fixes that, and these tests pin the property.
"""

import torch
from scipy.spatial.transform import Rotation as R

from protomotions.envs.rewards.task import compute_backward_velocity_rew

TARGET = 0.25
FACING_X = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # identity quat, robot faces +x


def reward(vx: float, vy: float = 0.0, quat: torch.Tensor = FACING_X) -> float:
    vel = torch.zeros(1, 17, 3)
    vel[0, 0, 0] = vx
    vel[0, 0, 1] = vy
    return float(
        compute_backward_velocity_rew(quat, vel, target_speed=TARGET)[0]
    )


def test_standing_still_earns_nothing():
    """The bug that motivated the rewrite: a Gaussian paid 0.78 here."""
    assert reward(0.0) == 0.0


def test_walking_forward_earns_nothing():
    assert reward(+0.10) == 0.0
    assert reward(+0.50) == 0.0


def test_backward_at_target_is_full_credit():
    assert reward(-TARGET) == 1.0


def test_reward_increases_with_backward_speed():
    slow, med, fast = reward(-0.06), reward(-0.13), reward(-0.21)
    assert 0.0 < slow < med < fast < 1.0, (slow, med, fast)


def test_exceeding_target_is_not_penalised():
    """Style keeps the gait plausible; this term must not fight faster backing."""
    assert reward(-0.43) == 1.0
    assert reward(-0.80) == 1.0


def test_turning_around_and_walking_forward_earns_nothing():
    """Heading-relative: the reward must not be satisfiable by facing about.

    World velocity is -x, which for a robot yawed 180 deg is FORWARD motion.
    """
    yawed = torch.tensor(
        R.from_euler("z", 180, degrees=True).as_quat(), dtype=torch.float32
    ).unsqueeze(0)
    assert reward(-TARGET, quat=yawed) == 0.0


def test_crabbing_sideways_earns_nothing():
    assert reward(0.0, vy=0.5) == 0.0


def test_lateral_drift_is_penalised():
    straight = reward(-TARGET)
    drifting = reward(-TARGET, vy=0.5)
    assert drifting < straight, (drifting, straight)


def test_reward_stays_in_unit_range():
    for vx in (-2.0, -0.25, 0.0, 0.25, 2.0):
        for vy in (-1.0, 0.0, 1.0):
            r = reward(vx, vy)
            assert 0.0 <= r <= 1.0, (vx, vy, r)
