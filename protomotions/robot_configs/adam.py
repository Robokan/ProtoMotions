# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# PNDbotics Adam robot configuration for ProtoMotions
# Based on adam_sp (full humanoid with dexterous hands)

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
class AdamRobotConfig(RobotConfig):
    """
    PNDbotics Adam humanoid robot configuration.
    
    Adam is a full humanoid with:
    - 31 main body DOF (legs, waist, arms, wrists, neck)
    - 24 finger DOF per hand (48 total)
    - Stereo cameras in the head
    
    This config focuses on the main body joints for locomotion and manipulation.
    Finger joints are controlled via position actuators (not included in RL action space).
    """
    
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["toeLeft"],
            "all_right_foot_bodies": ["toeRight"],
            "all_left_hand_bodies": ["wristRollLeft"],
            "all_right_hand_bodies": ["wristRollRight"],
            "head_body_name": ["neckPitch_link"],
            "torso_body_name": ["torso"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "torso",
            "neckPitch_link",
            "toeRight",
            "toeLeft",
            "wristRollLeft",
            "wristRollRight",
        ]
    )

    default_root_height: float = 0.95

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/adam.xml",
            usd_asset_file_name="usd/adam/adam.usda",
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
                "wristPitch_.*": ControlInfo(
                    stiffness=10.0,
                    damping=1.0,
                    effort_limit=6.4,
                    velocity_limit=37,
                ),
                "wristRoll_.*": ControlInfo(
                    stiffness=10.0,
                    damping=1.0,
                    effort_limit=6.4,
                    velocity_limit=37,
                ),
                # Neck joints
                "neck.*": ControlInfo(
                    stiffness=10.0,
                    damping=1.0,
                    effort_limit=6.4,
                    velocity_limit=20,
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
