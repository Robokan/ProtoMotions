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
    # Commandable PD-target range per dof, fitted as the minimal circular
    # arc covering the full corpus (+-0.1 rad margin). The skeleton's hinges
    # are UNLIMITED in the MJCF (physical limits are impossible here: the
    # importer collapses each 3-hinge triplet into a PhysX D6 joint and
    # mangles per-axis limits to [0,0]), so ROM is enforced on the ACTION
    # side instead -- the policy cannot command targets outside the mocap
    # envelope. 18 euler-degenerate dofs (gimbal-smeared ForeArm/Arm/Tail
    # axes whose per-hinge angles wind) are unlisted and keep the default
    # +-pi scaling. Values are RADIANS in each dof's own corpus branch
    # (some arcs are far from zero, e.g. Spine_z ~ -4.7..-1.6).
    # Commandable PD-target range per dof, centered on a LITERAL stance
    # frame -- quietest frame (min mean joint speed) of the CURRENT training
    # corpus, data/motions/dog_v2/dog_full_flat.pt (202 clips, climb clips
    # excluded). Half-range covers this corpus's full excursion each side
    # +15%, capped +-pi. REGENERATE whenever the training corpus changes
    # (was clip-1-only; broader corpus needs wider per-dof arcs or actions
    # clip on everything but clip 1).
    action_scaling_limits: Dict[str, tuple] = field(
        default_factory=lambda: {
            r"Spine_x": (-3.135, 3.148),
            r"Spine_y": (-0.797, 0.772),
            r"Spine_z": (-6.101, 0.183),
            r"Spine1_x": (-3.147, 3.137),
            r"Spine1_y": (-1.474, 1.426),
            r"Spine1_z": (-1.553, 1.054),
            r"Neck_x": (-1.908, 1.968),
            r"Neck_y": (-1.582, 1.521),
            r"Neck_z": (-2.860, 2.139),
            r"Head_x": (-1.555, 4.728),
            r"Head_y": (-1.467, 2.073),
            r"Head_z": (-3.131, 3.152),
            r"LeftShoulder_x": (-3.099, -1.479),
            r"LeftShoulder_y": (-0.692, 0.298),
            r"LeftShoulder_z": (-2.793, 0.911),
            r"LeftArm_x": (-0.565, 5.719),
            r"LeftArm_y": (-1.889, 3.960),
            r"LeftArm_z": (-1.670, 4.613),
            r"LeftForeArm_x": (-3.044, 3.240),
            r"LeftForeArm_y": (-2.250, 1.825),
            r"LeftForeArm_z": (-4.015, 2.268),
            r"LeftHand_x": (-3.157, 3.126),
            r"LeftHand_y": (-1.701, 1.080),
            r"LeftHand_z": (-2.055, 1.572),
            r"RightShoulder_x": (-1.691, -0.094),
            r"RightShoulder_y": (-0.697, 0.365),
            r"RightShoulder_z": (-0.730, 2.780),
            r"RightArm_x": (-2.829, 3.454),
            r"RightArm_y": (-4.182, 1.957),
            r"RightArm_z": (-1.862, 4.421),
            r"RightForeArm_x": (-3.231, 3.052),
            r"RightForeArm_y": (-1.802, 2.519),
            r"RightForeArm_z": (-4.151, 2.132),
            r"RightHand_x": (-3.130, 3.153),
            r"RightHand_y": (-1.328, 1.510),
            r"RightHand_z": (-2.068, 1.766),
            r"LeftUpLeg_x": (-1.980, 1.747),
            r"LeftUpLeg_y": (-1.474, 1.849),
            r"LeftUpLeg_z": (-4.500, 0.164),
            r"LeftLeg_x": (-2.251, 2.135),
            r"LeftLeg_y": (-1.200, 1.521),
            r"LeftLeg_z": (-1.806, 4.477),
            r"LeftFoot_x": (-0.544, 0.621),
            r"LeftFoot_y": (-1.345, 1.143),
            r"LeftFoot_z": (-1.721, 3.437),
            r"RightUpLeg_x": (-2.029, 1.999),
            r"RightUpLeg_y": (-1.545, 1.565),
            r"RightUpLeg_z": (-4.498, 0.140),
            r"RightLeg_x": (-2.163, 2.253),
            r"RightLeg_y": (-1.144, 1.174),
            r"RightLeg_z": (-1.800, 4.483),
            r"RightFoot_x": (-0.632, 0.699),
            r"RightFoot_y": (-1.124, 1.063),
            r"RightFoot_z": (-1.728, 3.533),
            r"Tail_x": (-2.513, 3.770),
            r"Tail_y": (-3.568, 1.823),
            r"Tail_z": (-3.867, 2.416),
            r"Tail1_x": (-4.052, 2.232),
            r"Tail1_y": (-1.562, 1.607),
            r"Tail1_z": (-3.132, 2.991),
        }
    )

    # Median walking pose of clip 33 (unwrapped intrinsic-XYZ dps, the same
    # convention the corpus plays back through FK). Zeros is NOT a standing
    # dog here: the all-zero drop test measured foot origins at 0.42 m and
    # the sim repack grounded clips to it, floating the corpus 0.38 m up.
    default_dof_pos: Dict[str, float] = field(
        default_factory=lambda: {
            "Spine_x": -0.0594,
            "Spine_y": 0.1204,
            "Spine_z": -2.6936,
            "Spine1_x": 0.0323,
            "Spine1_y": 0.1077,
            "Spine1_z": -0.3129,
            "Neck_x": -0.1065,
            "Neck_y": 0.0669,
            "Neck_z": -0.2408,
            "Head_x": 1.6334,
            "Head_y": 0.2979,
            "Head_z": -0.1947,
            "LeftShoulder_x": -2.2458,
            "LeftShoulder_y": -0.1871,
            "LeftShoulder_z": -0.9060,
            "LeftArm_x": 2.4492,
            "LeftArm_y": 0.8608,
            "LeftArm_z": 1.8344,
            "LeftForeArm_x": 0.0866,
            "LeftForeArm_y": -0.0287,
            "LeftForeArm_z": -1.2477,
            "LeftHand_x": 0.0252,
            "LeftHand_y": -0.3170,
            "LeftHand_z": 0.0712,
            "RightShoulder_x": -0.9339,
            "RightShoulder_y": -0.1519,
            "RightShoulder_z": 1.0328,
            "RightArm_x": 0.3225,
            "RightArm_y": -1.0281,
            "RightArm_z": 1.5259,
            "RightForeArm_x": -0.1612,
            "RightForeArm_y": 0.1816,
            "RightForeArm_z": -1.3332,
            "RightHand_x": 0.0200,
            "RightHand_y": -0.0290,
            "RightHand_z": 0.0362,
            "LeftUpLeg_x": -0.3444,
            "LeftUpLeg_y": 0.1557,
            "LeftUpLeg_z": -2.2900,
            "LeftLeg_x": -0.1330,
            "LeftLeg_y": 0.1878,
            "LeftLeg_z": 1.8171,
            "LeftFoot_x": 0.0294,
            "LeftFoot_y": -0.0758,
            "LeftFoot_z": 0.4871,
            "RightUpLeg_x": 0.1358,
            "RightUpLeg_y": 0.0544,
            "RightUpLeg_z": -2.2302,
            "RightLeg_x": 0.0957,
            "RightLeg_y": -0.1091,
            "RightLeg_z": 1.8411,
            "RightFoot_x": -0.0006,
            "RightFoot_y": -0.0493,
            "RightFoot_z": 0.4694,
            "Tail_x": 0.3413,
            "Tail_y": 0.1230,
            "Tail_z": 0.2044,
            "Tail1_x": -0.7764,
            "Tail1_y": 0.0199,
            "Tail1_z": 0.1810,
        }
    )
    anchor_body_name: str = "trunk"

    # ground-contact extremities only
    contact_bodies: List[str] = field(
        default_factory=lambda: ["LeftFoot", "RightFoot", "LeftHand", "RightHand"]
    )

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/dog_v2_bones.xml",
            self_collisions=False,
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
            # Lab3-PhysX collapses each 3-hinge body into a rotvec-
            # parametrized D6; convert euler<->rotvec at the wire.
            hinge_triplet_rotvec_adapter=True,
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
