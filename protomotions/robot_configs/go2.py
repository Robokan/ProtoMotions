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
KP_ABDUCTION = 20.0
KD_ABDUCTION = 0.5
KP_HIP_KNEE = 20.0
KD_HIP_KNEE = 0.5

# Effort/velocity limits taken from the official Unitree URDF
# (urdf/go2/go2.urdf, verified limit-for-limit against upstream
# unitreerobotics/unitree_ros). All twelve joints use the SAME motor -- the
# knee just carries an extra 1.917:1 reduction, which is why it trades speed
# for torque: 23.7 x 30.1 = 713.4 W and 45.43 x 15.7 = 713.3 W are the same
# motor at the same mechanical power.
#
# Corrected 2026-09-06. The thigh had been given the knee's 45.43 Nm (1.92x the
# torque the real motor can deliver) and the calf the hip's 30.0 rad/s (1.91x
# its real speed). Both came from mjcf/go2_nomesh.xml: flattening the
# <default> class tree in mjcf/go2.xml applied the "knee" class's motor
# override to the "hip" (thigh) class as well. Unitree's own go2.xml overrides
# the base 23.7 Nm only inside class "knee". The MJCF was fixed in the same
# change.
EFFORT_ABDUCTION = 23.7
VELOCITY_ABDUCTION = 30.1

EFFORT_THIGH = 23.7
VELOCITY_THIGH = 30.1

EFFORT_CALF = 45.43
VELOCITY_CALF = 15.7

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

    # Measured, not guessed: default pose (thigh 0.9, calf -1.8) puts the foot
    # origins 0.2648 m below the root, and the foot collision sphere has radius
    # 0.022 -- so standing root height is 0.2868. The old 0.34 was 5.3 cm high,
    # and repack_motion_lib_via_sim anchors its foot-grounding reference to
    # this value, so every repacked corpus inherited that float (Eric spotted
    # the robot hovering in go2_flat, 2026-08-25).
    default_root_height: float = 0.2868
    default_dof_pos: Dict[str, float] = field(default_factory=lambda: DEFAULT_JOINT_POS)

    # NOTE: deliberately NO action_scaling_limits. Adding a standing-centered
    # table (2026-08-25) to tidy the zero-action pose also switched
    # make_pd_action_config to action_scale=0.5 -- exactly the spec instead of
    # the default 2x joint span -- which halved the policy's control authority
    # and broke a Go2 that had been walking fine. The sprawled zero-action
    # pose is cosmetic: the policy learns around it (Atlas does the same).
    anchor_body_name: str = "base_link"

    # Only feet need contact sensing for termination/reward
    contact_bodies: List[str] = field(
        default_factory=lambda: ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    )

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/go2_nomesh.xml",
            urdf_asset_file_name="urdf/go2/go2.urdf",
            usd_asset_file_name="usd/go2/go2.usd",
            # Local Go2 USD matches the MJCF skeleton (base_link, 17 bodies).
            # Isaac Lab 6.0 Nucleus Go2 (Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd)
            # uses `base` plus Head_upper/Head_lower rigid bodies, so it cannot
            # play ProtoMotions clips without a remapping. Textures live in
            # Props/instanceable_meshes.usd next to this file.
            lab3_usd_asset_file_name="usd/go2/go2.usd",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/go2_description/",
            apply_default_visual_material=False,
            replace_cylinder_with_capsule=True,
            thickness=0.01,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
            # Corpus retarget artifact, not a live-training tuning choice:
            # opposite-side hips/calves come within 0.2-1.0cm of each other
            # in go2_full (rest-pose separation is 9.3cm/28.4cm), so
            # self-collision fires a large separation impulse on the first
            # physics step of any reset landing near one of those frames --
            # violent and reset-universal, independent of training progress
            # (Eric, 2026-08-24). Same fix already proven safe for dog_v2.
            self_collisions=False,
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
                    velocity_limit=VELOCITY_ABDUCTION,
                ),
                ".*_thigh_joint": ControlInfo(
                    stiffness=KP_HIP_KNEE,
                    damping=KD_HIP_KNEE,
                    effort_limit=EFFORT_THIGH,
                    velocity_limit=VELOCITY_THIGH,
                ),
                ".*_calf_joint": ControlInfo(
                    stiffness=KP_HIP_KNEE,
                    damping=KD_HIP_KNEE,
                    effort_limit=EFFORT_CALF,
                    velocity_limit=VELOCITY_CALF,
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


def go2_front_camera(
    width: int = 224,
    height: int = 224,
    fov_deg: float = 120.0,
    update_period: float = 0.0,
):
    """The go2's forward camera, where the real one is.

    The MJCF puts the trunk's front face at x = 0.188 and the head assembly
    ahead of it (a cylinder at x = 0.285, a nose sphere at x = 0.293), so the
    lens sits at x = 0.33 -- just proud of the nose, on the centreline, level
    with the trunk origin and therefore about 0.34 m off the ground when the
    dog is standing.

    Level by default. A ball on the floor is only 5 deg below the axis at 4 m
    and 20 deg at arm's length, both well inside a 120 deg frame, so tilting
    down buys nothing and costs the horizon. The local go2 USD has no separate
    head body, so this rides base_link -- which is what the real camera does
    too, rigidly.
    """
    from protomotions.simulator.base_simulator.config import OnboardCameraConfig

    return OnboardCameraConfig(
        body_name="base_link",
        pos=(0.33, 0.0, 0.0),
        pitch_deg=0.0,
        width=width,
        height=height,
        horizontal_fov_deg=fov_deg,
        update_period=update_period,
    )
