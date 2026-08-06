# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Utahraptor: the raptor scaled to 200 kg, as a match for the tiger.

Generated from raptor.xml by data/scripts/scale_robot_mjcf.py with every
length multiplied by

    s = (200 / 40.05)^(1/3) = 1.709260

so mass follows as s^3 at unchanged density. Hips sit at 0.87 m instead
of 0.51 m; the whole animal is 1.71x longer and 5x heavier.

GAINS. Torque needed to hold a pose against gravity goes as m*g*L, i.e.
s^4 = 8.54x, so every effort here is the raptor's x8.54. Damping needs
one more factor: joint angular velocity in a dynamically similar motion
scales as 1/sqrt(s), so to produce s^4 times the damping torque from
1/sqrt(s) times the velocity, kd must scale as s^4.5 = 11.2x. dog_v2's
_pd() hardcodes kd = kp/10 and cannot express that, hence _pd_big below.
Getting this wrong is not subtle -- kd short by sqrt(s) is a 200 kg
animal with a Velociraptor's damping, which oscillates.

MOTIONS. The corpus is NOT the raptor's. A bigger body cannot perform the
same motion at the same speed because gravity does not scale; see
data/scripts/scale_motions.py, which applies Froude similarity (length s,
time sqrt(s)) to produce data/motions/utahraptor.
"""

from dataclasses import dataclass, field, replace as _replace
from typing import Dict, List

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
)
from protomotions.robot_configs.dog_v2 import VELOCITY_LIMIT

# length scale vs the 40 kg raptor
SCALE = 1.709260
TORQUE_SCALE = SCALE ** 4          # 8.536  -- m*g*L
DAMPING_SCALE = SCALE ** 4.5       # 11.161 -- torque / (angular velocity)


def _pd_big(effort: float) -> ControlInfo:
    """dog_v2's _pd, but with the two scale exponents applied separately.

    kp scales with torque; kd scales with torque DIVIDED by the scaled
    angular velocity, which is 1/sqrt(s) of the original.
    """
    kp = 2.0 * effort * TORQUE_SCALE
    return ControlInfo(
        stiffness=kp,
        damping=(2.0 * effort / 10.0) * DAMPING_SCALE,
        effort_limit=effort * TORQUE_SCALE,
        # joint speeds are LOWER on a bigger animal (1/sqrt(s)), so the
        # ceiling comes down rather than up
        velocity_limit=VELOCITY_LIMIT / (SCALE ** 0.5),
    )


CONTROL_OVERRIDES = {
    # Catch-all FIRST (later matches override).
    r".*_[xyz]": _pd_big(6.0),
    r"Spine1?_[xyz]": _pd_big(90.0),
    r"Neck[0-9]?_[xyz]": _pd_big(35.0),
    r"Head_[xyz]": _pd_big(25.0),
    r"Jaw_[xyz]": _pd_big(20.0),
    r"Tail[0-9]_[xyz]": _pd_big(15.0),
    r"Tail_[xyz]": _pd_big(100.0),
    r"Tail1_[xyz]": _pd_big(80.0),
    r"Tail2_[xyz]": _pd_big(55.0),
    r"Tail3_[xyz]": _pd_big(30.0),
    r"(Left|Right)UpLeg_[xyz]": _pd_big(120.0),
    r"(Left|Right)Leg_[xyz]": _pd_big(100.0),
    r"(Left|Right)Foot_[xyz]": _pd_big(60.0),
    r"(Left|Right)ToeBase_[xyz]": _pd_big(25.0),
    r"(Left|Right)Shoulder_[xyz]": _pd_big(30.0),
    r"(Left|Right)Arm_[xyz]": _pd_big(25.0),
    r"(Left|Right)ForeArm_[xyz]": _pd_big(20.0),
    r"(Left|Right)Hand_[xyz]": _pd_big(10.0),
}

# No armature, as on the raptor: BUILT_IN_PD is IsaacLab's implicit
# actuator, so the explicit-integration stability argument never applied,
# and a nonzero value here is a flywheel on every limb.
CONTROL_OVERRIDES = {
    _k: _replace(_v, armature=None) for _k, _v in CONTROL_OVERRIDES.items()
}


@dataclass
class UtahraptorRobotConfig(RobotConfig):
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["LeftToeBase"],
            "all_right_foot_bodies": ["RightToeBase"],
            "all_left_hand_bodies": ["LeftHand"],
            "all_right_hand_bodies": ["RightHand"],
            "head_body_name": ["Head"],
            "torso_body_name": ["Spine1"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "Hips", "Spine1", "Head", "Jaw", "Tail3", "Tail5",
            "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
            "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
            "LeftArm", "LeftHand", "RightArm", "RightHand",
        ]
    )

    # Same skeleton as the raptor, so the same discriminator argument
    # applies: the digit segments are unreproducible and would let the
    # discriminator win on jitter instead of gait. Tips are named so they
    # stay visible and cannot be planted for free support.
    disc_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "Hips", "Spine1", "Head", "Jaw", "Tail3", "Tail5",
            "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
            "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
            "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
            "RightShoulder", "RightArm", "RightForeArm", "RightHand",
            "LeftHandIndex3", "LeftHandMiddle3", "LeftHandRing3",
            "RightHandIndex3", "RightHandMiddle3", "RightHandRing3",
            "LeftFootIndex3", "LeftFootMiddle3", "LeftFootRing3",
            "RightFootIndex3", "RightFootMiddle3", "RightFootRing3",
        ]
    )

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/utahraptor.xml",
            usd_asset_file_name="usd/utahraptor/utahraptor_flat.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/Hips/",
            self_collisions=False,
        )
    )

    default_root_height: float = 0.51 * SCALE      # 0.8717 m
    contact_bodies: List[str] = None

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info=dict(CONTROL_OVERRIDES),
        )
    )
