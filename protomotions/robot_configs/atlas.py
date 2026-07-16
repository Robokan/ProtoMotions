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
class AtlasRobotConfig(RobotConfig):
    """Boston Dynamics Atlas (2025 rig) tuned as a 150 lb research humanoid.

    Source model: the GMR retargeting rig (GMR/assets/atlas_mujoco/atlas.xml)
    made physics-ready by data/scripts/retune_atlas_mjcf.py — 68.04 kg total,
    EngineAI-class strength-to-weight (280 Nm hips/knees), passive ankle ball
    joints replaced by actuated pitch+roll hinges. 32 bodies, 30 actuated DOFs.

    Effort limits live in the MJCF (actuatorfrcrange per joint); the control
    override here supplies PD gains scaled per joint group.
    """

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["Foot_L"],
            "all_right_foot_bodies": ["Foot_R"],
            "all_left_hand_bodies": ["Hand1_L"],
            "all_right_hand_bodies": ["Hand1_R"],
            "head_body_name": ["Head"],
            "torso_body_name": ["Chest"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "Hip",
            "Head",
            "Foot_L",
            "Foot_R",
            "Hand1_L",
            "Hand1_R",
        ]
    )

    default_root_height: float = 1.05

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/atlas.xml",
            usd_asset_file_name="usd/atlas/atlas_flat.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/Hip/",
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )

    # PD gains per motor class (efforts come from the MJCF actuatorfrcrange).
    # Stiffness ~ effort-scaled, following the h1_2/g1 big-humanoid pattern.
    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                # hips + knees (280 Nm class)
                r"Leg_[1358]_[LR]_Joint": ControlInfo(
                    stiffness=400, damping=15, effort_limit=280,
                    velocity_limit=26,
                ),
                # ankles (90 Nm pitch, 60 Nm roll/swivel)
                r"Foot_[LR]_Pitch": ControlInfo(
                    stiffness=120, damping=6, effort_limit=90,
                    velocity_limit=35,
                ),
                r"Foot_[LR]_(Roll|Yaw)": ControlInfo(
                    stiffness=80, damping=4, effort_limit=60,
                    velocity_limit=35,
                ),
                # waist (140 Nm class)
                r"(Twist|Backbone)_Joint": ControlInfo(
                    stiffness=250, damping=10, effort_limit=140,
                    velocity_limit=26,
                ),
                # shoulders + elbows (90 Nm class)
                r"Arm_[136]_[LR]_Joint": ControlInfo(
                    stiffness=120, damping=6, effort_limit=90,
                    velocity_limit=35,
                ),
                # upper-arm yaw (60 Nm class)
                r"Arm_4_[LR]_Joint": ControlInfo(
                    stiffness=80, damping=4, effort_limit=60,
                    velocity_limit=35,
                ),
                # wrists (30 Nm class)
                r"Arm_[789]_[LR]_Joint": ControlInfo(
                    stiffness=40, damping=2, effort_limit=30,
                    velocity_limit=35,
                ),
                # neck/head (25 Nm class)
                r"(Neck_2|Head)_Joint": ControlInfo(
                    stiffness=30, damping=1.5, effort_limit=25,
                    velocity_limit=35,
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
