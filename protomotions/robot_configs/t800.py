# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from protomotions.robot_configs.base import (
    RobotConfig,
    RobotAssetConfig,
    ControlConfig,
    ControlType,
    SimulatorParams,
)
from protomotions.simulator.isaacgym.config import IsaacGymSimParams
from protomotions.simulator.isaaclab.config import (
    IsaacLabPhysXParams,
    IsaacLabSimParams,
)
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.components.pose_lib import ControlInfo
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class T800RobotConfig(RobotConfig):
    """EngineAI T800 humanoid (25 DOF) for ProtoMotions / GPC.

    Source: GMR ``assets/t800/mujoco/t800_full_gmr.xml`` — already physics-ready
    with URDF-aligned masses (~85 kg), chained single-joint ankles, and
    actuatorfrcrange from the training URDF. Ported into
    ``protomotions/data/assets/mjcf/t800.xml`` (viewer floor/skybox stripped,
    mesh/texture paths rewritten).
    """

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["LINK_ANKLE_ROLL_L", "LINK_FOOT_L"],
            "all_right_foot_bodies": ["LINK_ANKLE_ROLL_R", "LINK_FOOT_R"],
            "all_left_hand_bodies": ["LINK_WRIST_END_L"],
            "all_right_hand_bodies": ["LINK_WRIST_END_R"],
            "head_body_name": ["LINK_HEAD_YAW"],
            "torso_body_name": ["LINK_TORSO_YAW"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "LINK_BASE",
            "LINK_HEAD_YAW",
            "LINK_ANKLE_ROLL_L",
            "LINK_ANKLE_ROLL_R",
            "LINK_WRIST_END_L",
            "LINK_WRIST_END_R",
        ]
    )

    default_root_height: float = 1.02

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/t800.xml",
            override_visual_material=False,
            usd_asset_file_name="usd/t800/t800_flat.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/LINK_BASE/",
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )

    # PD gains scaled to MJCF actuatorfrcrange (effort limits live in the asset).
    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                # hips pitch + knees (415 Nm)
                r"J0[0369]_(HIP_PITCH|KNEE_PITCH)_[LR]": ControlInfo(
                    stiffness=500, damping=18, effort_limit=415, velocity_limit=26,
                ),
                # hip roll (370 Nm)
                r"J0[17]_HIP_ROLL_[LR]": ControlInfo(
                    stiffness=450, damping=16, effort_limit=370, velocity_limit=26,
                ),
                # hip yaw + torso yaw (222 Nm)
                r"J(02|08)_HIP_YAW_[LR]|J12_TORSO_YAW": ControlInfo(
                    stiffness=300, damping=12, effort_limit=222, velocity_limit=26,
                ),
                # ankles (160 Nm)
                r"J(04|05|10|11)_ANKLE_(PITCH|ROLL)_[LR]": ControlInfo(
                    stiffness=200, damping=8, effort_limit=160, velocity_limit=35,
                ),
                # shoulders + elbow pitch (160 Nm)
                r"J(13|14|15|16|20|21|22|23)_(SHOULDER|ELBOW)_[A-Z]+_[LR]": ControlInfo(
                    stiffness=200, damping=8, effort_limit=160, velocity_limit=35,
                ),
                # elbow yaw + head (52 Nm)
                r"J(17|24)_ELBOW_YAW_[LR]|J2[78]_HEAD_(PITCH|YAW)": ControlInfo(
                    stiffness=60, damping=3, effort_limit=52, velocity_limit=35,
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=60,
                decimation=2,
                substeps=2,
            ),
            isaaclab=IsaacLabSimParams(
                fps=120,
                decimation=4,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=4,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            genesis=GenesisSimParams(
                fps=60,
                decimation=2,
                substeps=2,
            ),
            newton=NewtonSimParams(
                fps=120,
                decimation=4,
            ),
        )
    )
