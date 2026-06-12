# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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

# Go2 PD gains used in RL (kp=20, kd=0.5 is standard for position-controlled Go2)
# Effort limits from MJCF: abduction class = 23.7 Nm, hip/knee class = 45.43 Nm
KP_ABDUCTION = 20.0
KD_ABDUCTION = 0.5
EFFORT_ABDUCTION = 23.7

KP_HIP_KNEE = 20.0
KD_HIP_KNEE = 0.5
EFFORT_HIP_KNEE = 45.43

VELOCITY_LIMIT = 30.0

# Default standing pose (radians) matching MJCF keyframe: thigh=0.9, calf=-1.8
DEFAULT_JOINT_POS = {
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 0.9,
    ".*_calf_joint": -1.8,
}


@dataclass
class Go2RobotConfig(RobotConfig):
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["FL_foot", "RL_foot"],
            "all_right_foot_bodies": ["FR_foot", "RR_foot"],
            "all_left_hand_bodies": [],
            "all_right_hand_bodies": [],
            "head_body_name": ["base_link"],
            "torso_body_name": ["base_link"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "base_link",
            "FL_thigh",
            "FL_calf",
            "FL_foot",
            "FR_thigh",
            "FR_calf",
            "FR_foot",
            "RL_thigh",
            "RL_calf",
            "RL_foot",
            "RR_thigh",
            "RR_calf",
            "RR_foot",
        ]
    )

    default_root_height: float = 0.34
    default_dof_pos: Dict[str, float] = field(default_factory=lambda: DEFAULT_JOINT_POS)
    anchor_body_name: str = "base_link"

    # Only feet need contact sensing for termination/reward
    contact_bodies: List[str] = field(
        default_factory=lambda: ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    )

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/go2_nomesh.xml",
            usd_asset_file_name="usd/go2/go2.usd",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/go2_description/",
            replace_cylinder_with_capsule=True,
            thickness=0.01,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                ".*_hip_joint": ControlInfo(
                    stiffness=KP_ABDUCTION,
                    damping=KD_ABDUCTION,
                    effort_limit=EFFORT_ABDUCTION,
                    velocity_limit=VELOCITY_LIMIT,
                ),
                ".*_thigh_joint": ControlInfo(
                    stiffness=KP_HIP_KNEE,
                    damping=KD_HIP_KNEE,
                    effort_limit=EFFORT_HIP_KNEE,
                    velocity_limit=VELOCITY_LIMIT,
                ),
                ".*_calf_joint": ControlInfo(
                    stiffness=KP_HIP_KNEE,
                    damping=KD_HIP_KNEE,
                    effort_limit=EFFORT_HIP_KNEE,
                    velocity_limit=VELOCITY_LIMIT,
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=200,
                decimation=4,
                substeps=2,
                physx=IsaacGymPhysXParams(
                    num_position_iterations=4,
                    num_velocity_iterations=0,
                    max_depenetration_velocity=1,
                ),
            ),
            isaaclab=IsaacLabSimParams(
                fps=200,
                decimation=4,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=4,
                    num_velocity_iterations=0,
                    max_depenetration_velocity=1,
                ),
            ),
            genesis=GenesisSimParams(
                fps=200,
                decimation=4,
                substeps=2,
            ),
            newton=NewtonSimParams(
                fps=200,
                decimation=4,
                use_cuda_graph=True,
                nconmax=40,
                njmax=350,
                ccd_iterations=16,
            ),
        )
    )
