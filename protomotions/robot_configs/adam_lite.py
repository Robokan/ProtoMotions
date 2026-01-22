# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# PNDbotics Adam Lite robot configuration for ProtoMotions
# Adam Lite is the body-only version (no dexterous hands)

from protomotions.robot_configs.base import (
    RobotConfig,
    RobotAssetConfig,
    ControlConfig,
    ControlType,
    SimulatorParams,
)
from protomotions.simulator.isaacgym.config import (
    IsaacGymSimParams,
    IsaacGymPhysXParams,
)
from protomotions.simulator.isaaclab.config import (
    IsaacLabSimParams,
    IsaacLabPhysXParams,
)
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.components.pose_lib import ControlInfo
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class AdamLiteRobotConfig(RobotConfig):
    """
    PNDbotics Adam Lite humanoid robot configuration.
    
    Adam Lite is the body-only version with:
    - 25 actuated DOF (legs, waist, arms, wrists)
    - No finger joints
    - No head/neck joints
    
    This is suitable for locomotion and whole-body control without hand manipulation.
    """
    
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["toeLeft"],
            "all_right_foot_bodies": ["toeRight"],
            "all_left_hand_bodies": ["wristYawLeft"],
            "all_right_hand_bodies": ["wristYawRight"],
            "head_body_name": ["waistYaw_link"],  # No head in Adam Lite, use torso
            "torso_body_name": ["torso"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "torso",
            "toeRight",
            "toeLeft",
            "wristYawLeft",
            "wristYawRight",
        ]
    )

    default_root_height: float = 0.95

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/adam_lite.xml",
            usd_asset_file_name="usd/adam_lite/adam_lite.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/pelvis/",
            replace_cylinder_with_capsule=True,
            thickness=0.01,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            density=0.001,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                # Hip joints - high torque
                "hipPitch_.*": ControlInfo(
                    stiffness=100.0,
                    damping=10.0,
                    effort_limit=230,
                    velocity_limit=32,
                ),
                "hipRoll_.*": ControlInfo(
                    stiffness=100.0,
                    damping=10.0,
                    effort_limit=160,
                    velocity_limit=32,
                ),
                "hipYaw_.*": ControlInfo(
                    stiffness=80.0,
                    damping=8.0,
                    effort_limit=105,
                    velocity_limit=32,
                ),
                # Knee joints
                "kneePitch_.*": ControlInfo(
                    stiffness=100.0,
                    damping=10.0,
                    effort_limit=230,
                    velocity_limit=32,
                ),
                # Ankle joints
                "anklePitch_.*": ControlInfo(
                    stiffness=40.0,
                    damping=4.0,
                    effort_limit=40,
                    velocity_limit=37,
                ),
                "ankleRoll_.*": ControlInfo(
                    stiffness=20.0,
                    damping=2.0,
                    effort_limit=12,
                    velocity_limit=37,
                ),
                # Waist joints
                "waist.*": ControlInfo(
                    stiffness=80.0,
                    damping=8.0,
                    effort_limit=110,
                    velocity_limit=32,
                ),
                # Shoulder joints
                "shoulderPitch_.*": ControlInfo(
                    stiffness=50.0,
                    damping=5.0,
                    effort_limit=65,
                    velocity_limit=37,
                ),
                "shoulderRoll_.*": ControlInfo(
                    stiffness=50.0,
                    damping=5.0,
                    effort_limit=65,
                    velocity_limit=37,
                ),
                "shoulderYaw_.*": ControlInfo(
                    stiffness=50.0,
                    damping=5.0,
                    effort_limit=65,
                    velocity_limit=37,
                ),
                # Elbow
                "elbow_.*": ControlInfo(
                    stiffness=30.0,
                    damping=3.0,
                    effort_limit=30,
                    velocity_limit=37,
                ),
                # Wrist joints
                "wristYaw_.*": ControlInfo(
                    stiffness=10.0,
                    damping=1.0,
                    effort_limit=6.4,
                    velocity_limit=37,
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=100,
                decimation=2,
                substeps=2,
                physx=IsaacGymPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            isaaclab=IsaacLabSimParams(
                fps=200,
                decimation=4,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            genesis=GenesisSimParams(
                fps=100,
                decimation=2,
                substeps=2,
            ),
            newton=NewtonSimParams(
                fps=200,
                decimation=4,
            ),
        )
    )
