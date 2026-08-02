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
    r"Spine1?_[xyz]": _pd(90.0),
    r"Neck[13]?_[xyz]": _pd(35.0),
    r"Head_[xyz]": _pd(25.0),
    r"Jaw_[xyz]": _pd(20.0),
    r"Tail[135]_[xyz]": _pd(15.0),
    r"(Left|Right)UpLeg_[xyz]": _pd(120.0),
    r"(Left|Right)Leg_[xyz]": _pd(100.0),
    r"(Left|Right)Foot_[xyz]": _pd(60.0),
    r"(Left|Right)ToeBase_[xyz]": _pd(25.0),
    r"(Left|Right)Shoulder_[xyz]": _pd(30.0),
    r"(Left|Right)Arm_[xyz]": _pd(25.0),
    r"(Left|Right)ForeArm_[xyz]": _pd(20.0),
    r"(Left|Right)Hand_[xyz]": _pd(10.0),
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
