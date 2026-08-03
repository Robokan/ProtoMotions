# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Raptor robot (fbx2robot import of the UE RaptorDinosaur pack).

27 bodies / 78 hinges (<Body>_x/_y/_z, world-aligned bind frames),
40 kg, hips at 0.51 m. Bipedal theropod: legs carry locomotion, jaw and
feet are the strike surfaces, tail balances. PD gains follow the dog_v2
translation (kp = 2*effort, kd = kp/10), efforts scaled for 40 kg.
See RAPTOR_TIGER_PLAN.md.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
)
from protomotions.robot_configs.dog_v2 import _pd


CONTROL_OVERRIDES = {
    # Catch-all FIRST (later matches override): digits, tongue, eyes,
    # sockets and every other small bone of the full skeleton.
    r".*_[xyz]": _pd(6.0),
    r"Spine1?_[xyz]": _pd(90.0),
    r"Neck[0-9]?_[xyz]": _pd(35.0),
    r"Head_[xyz]": _pd(25.0),
    r"Jaw_[xyz]": _pd(20.0),
    r"Tail[0-9]_[xyz]": _pd(15.0),
    # Tail gains sized from measured gravity-hold torque x2 (the tail is a
    # 10 kg cantilever off the hips; the base needed 50 Nm against a 6 Nm
    # limit). Gradient from base to tip.
    r"Tail_[xyz]": _pd(100.0),
    r"Tail1_[xyz]": _pd(80.0),
    r"Tail2_[xyz]": _pd(55.0),
    r"Tail3_[xyz]": _pd(30.0),
    r"(Left|Right)UpLeg_[xyz]": _pd(120.0),
    r"(Left|Right)Leg_[xyz]": _pd(100.0),
    r"(Left|Right)Foot_[xyz]": _pd(60.0),
    r"(Left|Right)ToeBase_[xyz]": _pd(25.0),
    r"(Left|Right)Shoulder_[xyz]": _pd(30.0),
    r"(Left|Right)Arm_[xyz]": _pd(25.0),
    r"(Left|Right)ForeArm_[xyz]": _pd(20.0),
    r"(Left|Right)Hand_[xyz]": _pd(10.0),
}


# Rotor/gear inertia. fbx2robot writes armature="0.02" into the MJCF joint
# default, but _pd() leaves ControlInfo.armature None and IsaacLab then
# applies its own 0.0 — so the thin distal links (digits, jaw, tail tip)
# would be driven with only their own inertia. A 1.4 g finger phalanx has
# I ~ 1.5e-8 kg m2 against kp 12: omega ~ 28,000 rad/s, omega*dt ~ 143 at
# 200 Hz, i.e. guaranteed divergence -> NaN observations -> dead policy.
# Mirror the MJCF value so every joint is integrable.
from dataclasses import replace as _replace
CONTROL_OVERRIDES = {
    _k: _replace(_v, armature=0.02) for _k, _v in CONTROL_OVERRIDES.items()
}


@dataclass
class RaptorRobotConfig(RobotConfig):
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["LeftToeBase"],
            "all_right_foot_bodies": ["RightToeBase"],
            # forelimb claws play the "hands" role for battle tables later
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

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/raptor.xml",
            usd_asset_file_name="usd/raptor/raptor_flat.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/Hips/",
            self_collisions=False,
        )
    )

    default_root_height: float = 0.51
    contact_bodies: List[str] = None

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info=dict(CONTROL_OVERRIDES),
        )
    )
