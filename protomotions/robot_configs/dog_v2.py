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
"""Robot config for the dm_control dog_v2 (mesh-free MJCF variant).

Control gains: the original dog_v2 uses first-order-filtered torque actuators
(general actuators with ctrlrange [-1, 1] and per-group gainprm G), so G is
the peak torque [Nm] of each actuator group. ProtoMotions drives joints with
PD control instead, so we translate per group:
    effort_limit = G            (peak torque preserved)
    stiffness kp = 2 * G        (kp * ~0.5 rad tracking error saturates torque)
    damping  kd = kp / 10       (moderately damped; standard quadruped ratio)
The 8 tendon-coupled spine/neck/tail actuators in the original drive the
vertebral joints as groups; here every vertebral joint gets its own PD gain
derived from the same group gainprm. These are initial values intended to be
tuned during training.
"""

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

VELOCITY_LIMIT = 30.0


def _pd(effort: float) -> ControlInfo:
    """Translate an actuator group's gainprm (peak torque) to PD ControlInfo."""
    kp = 2.0 * effort
    return ControlInfo(
        stiffness=kp,
        damping=kp / 10.0,
        effort_limit=effort,
        velocity_limit=VELOCITY_LIMIT,
    )


# joint-name regex -> PD gains. BVH-matched skeleton: every non-root body has
# three orthogonal hinges <Body>_x/_y/_z. Cover all 20 bodies (60 hinges) so
# every DOF has a stiffness (make_pd_action_config requires a real value).
CONTROL_OVERRIDES = {
    r"Spine1?_[xyz]": _pd(50.0),  # spine
    r"Neck_[xyz]": _pd(20.0),
    r"Head_[xyz]": _pd(15.0),
    r"(Left|Right)UpLeg_[xyz]": _pd(40.0),  # hind hip
    r"(Left|Right)Leg_[xyz]": _pd(30.0),  # hind knee
    r"(Left|Right)Foot_[xyz]": _pd(20.0),  # hind ankle/foot
    r"(Left|Right)Shoulder_[xyz]": _pd(30.0),  # front shoulder/scapula
    r"(Left|Right)Arm_[xyz]": _pd(30.0),  # front upper arm
    r"(Left|Right)ForeArm_[xyz]": _pd(20.0),  # front elbow
    r"(Left|Right)Hand_[xyz]": _pd(10.0),  # front wrist/paw
    r"Tail1?_[xyz]": _pd(2.0),  # tail
}


@dataclass
class DogV2RobotConfig(RobotConfig):
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            # hind feet
            "all_left_foot_bodies": ["LeftFoot"],
            "all_right_foot_bodies": ["RightFoot"],
            # front feet (the BVH "hand" joints are the front legs of the dog)
            "all_left_hand_bodies": ["LeftHand"],
            "all_right_hand_bodies": ["RightHand"],
            "head_body_name": ["Head"],
            "torso_body_name": ["trunk"],
        }
    )

    # trunk + ends of each leg/arm chain
    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "trunk",
            "Spine1",
            "Head",
            "LeftUpLeg",
            "LeftLeg",
            "LeftFoot",
            "RightUpLeg",
            "RightLeg",
            "RightFoot",
            "LeftShoulder",
            "LeftArm",
            "LeftForeArm",
            "LeftHand",
            "RightShoulder",
            "RightArm",
            "RightForeArm",
            "RightHand",
        ]
    )

    # BVH-matched rest-pose root (trunk = Hips) height
    default_root_height: float = 0.47
    # None -> zeros, which is exactly the MJCF rest pose (a standing dog)
    default_dof_pos: Dict[str, float] = None
    anchor_body_name: str = "trunk"

    # ground-contact extremities only
    contact_bodies: List[str] = field(
        default_factory=lambda: ["LeftFoot", "RightFoot", "LeftHand", "RightHand"]
    )

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/dog_v2_nomesh.xml",
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
            override_control_info=dict(CONTROL_OVERRIDES),
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
                nconmax=60,
                njmax=500,
                ccd_iterations=16,
            ),
        )
    )
