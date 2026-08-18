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

# ANYdrive 3.0 specs (from IsaacLab ANYDRIVE_3_SIMPLE_ACTUATOR_CFG)
KP = 40.0
KD = 5.0
EFFORT = 200.0
VELOCITY_LIMIT = 40.0

# Default X-stance matching IsaacLab ANYMAL_D_CFG init_state
DEFAULT_JOINT_POS = {
    ".*HAA": 0.0,
    ".*F_HFE": 0.4,
    ".*H_HFE": -0.4,
    ".*F_KFE": -0.8,
    ".*H_KFE": 0.8,
}


@dataclass
class AnymalDRobotConfig(RobotConfig):
    # ASE/AMP discriminator body subset. IsaacLabASE's working ANYmal ASE
    # shows the discriminator FOUR key bodies (the shanks) rather than all
    # 17; the un-subsetted version has the raptor's failure mode -- hips and
    # thighs move in ways the policy cannot reproduce, so the discriminator
    # separates agent from reference on jitter instead of gait. Feet are
    # omitted here for the same reason they are named on the raptor: the
    # shank already carries the swing, and the foot adds a fast segment the
    # policy would be judged on but cannot track.
    disc_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK",
        ]
    )

    # Action-scaling ranges, measured from anymal_d_flat.pt (164k frames,
    # 372 clips) and widened by 20 deg. The URDF declares every HFE/KFE as
    # +-9.42 rad (+-540 deg), so scaling actions to the asset limits put the
    # entire usable range inside ~2% of the action space and the policy could
    # only flail -- 12k epochs of it. IsaacLabASE dodges this by disabling
    # action scaling for ANYmal entirely (its comment cites the same +-540).
    # Simulator joint limits are unchanged; this only rescales actions.
    action_scaling_limits: Dict[str, tuple] = field(
        default_factory=lambda: {
            r".*HAA": (-1.13, 1.13),      # mocap +-0.79, symmetric
            r".*F_HFE": (-1.30, 2.51),    # front hips swing forward
            r".*H_HFE": (-3.01, 0.68),    # hind hips swing back
            r".*F_KFE": (-2.93, 0.55),
            r".*H_KFE": (-0.55, 3.19),
        }
    )

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["LF_FOOT", "LH_FOOT"],
            "all_right_foot_bodies": ["RF_FOOT", "RH_FOOT"],
            "all_left_hand_bodies": [],
            "all_right_hand_bodies": [],
            "head_body_name": ["base"],
            "torso_body_name": ["base"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "base",
            "LF_THIGH",
            "LF_SHANK",
            "LF_FOOT",
            "LH_THIGH",
            "LH_SHANK",
            "LH_FOOT",
            "RF_THIGH",
            "RF_SHANK",
            "RF_FOOT",
            "RH_THIGH",
            "RH_SHANK",
            "RH_FOOT",
        ]
    )

    default_root_height: float = 0.6
    default_dof_pos: Dict[str, float] = field(default_factory=lambda: DEFAULT_JOINT_POS)
    anchor_body_name: str = "base"

    # Only feet need contact sensing for termination/reward
    contact_bodies: List[str] = field(
        default_factory=lambda: ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"]
    )

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/anymal_d_nomesh.xml",
            urdf_asset_file_name="urdf/anymal_d/anymal.urdf",
            usd_asset_file_name="usd/anymal_d/anymal_d.usd",
            # Byte-identical to Isaac Lab 6.0 Nucleus
            # Isaac/IsaacLab/Robots/ANYbotics/ANYmal-D/anymal_d.usd
            lab3_usd_asset_file_name="usd/anymal_d/anymal_d.usd",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/",
            apply_default_visual_material=False,  # keep the USD's authored textures
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
                ".*HAA": ControlInfo(
                    stiffness=KP,
                    damping=KD,
                    effort_limit=EFFORT,
                    velocity_limit=VELOCITY_LIMIT,
                ),
                ".*HFE": ControlInfo(
                    stiffness=KP,
                    damping=KD,
                    effort_limit=EFFORT,
                    velocity_limit=VELOCITY_LIMIT,
                ),
                ".*KFE": ControlInfo(
                    stiffness=KP,
                    damping=KD,
                    effort_limit=EFFORT,
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
