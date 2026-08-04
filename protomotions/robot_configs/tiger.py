# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Tiger robot (fbx2robot import of the UE Animalia Tiger_M pack).

70 bodies / 207 hinges (<Body>_x/_y/_z, world-aligned bind frames),
4 digits x 2 segments on each paw (cats are digitigrade; the dewclaws
and claw tips erode at the 0.14 m prune),
200 kg, standing pelvis ~1.14 m (fbx2robot --min-bone-length 0.14 +
--drop-bones for whiskers/eyes/ears; anatomical densities: legs 70 kg,
tail 8 kg, torso 122 kg, COM over the four paws). Quadruped: all four ankles are contact feet;
front ankles double as the "hands" for the battle tables later (paw
swipes), jaw is the bite. PD gains follow the dog_v2 translation
(kp = 2*effort, kd = kp/10), efforts scaled for 200 kg.
See RAPTOR_TIGER_PLAN.md.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
)
from protomotions.robot_configs.dog_v2 import _pd


CONTROL_OVERRIDES = {
    # Catch-all FIRST (later matches override): digits, claws, tongue,
    # ears, whiskers and every other small bone of the full skeleton.
    r".*_[xyz]": _pd(10.0),
    r"RigSpine[0-9]_[xyz]": _pd(500.0),
    r"RigChest_[xyz]": _pd(500.0),
    r"RigNeck[0-9]_[xyz]": _pd(90.0),
    r"RigHead_[xyz]": _pd(60.0),
    r"RigJaw1_[xyz]": _pd(50.0),
    r"RigTail[0-9]_[xyz]": _pd(20.0),
    r"RigTail1_[xyz]": _pd(120.0),
    r"RigTail2_[xyz]": _pd(75.0),
    r"RigTail3_[xyz]": _pd(45.0),
    r"Rig[LR]BLeg1_[xyz]": _pd(280.0),   # hind hip
    r"Rig[LR]BLeg2_[xyz]": _pd(240.0),   # hind knee
    r"Rig[LR]BLeg3_[xyz]": _pd(160.0),   # hock
    r"Rig[LR]BLegAnkle_[xyz]": _pd(80.0),
    r"Rig[LR]FLegCollarbone_[xyz]": _pd(200.0),
    r"Rig[LR]ShoulderBlade1_[xyz]": _pd(200.0),
    r"Rig[LR]FLeg1_[xyz]": _pd(240.0),   # shoulder
    r"Rig[LR]FLeg2_[xyz]": _pd(200.0),   # elbow
    r"Rig[LR]FLeg3_[xyz]": _pd(140.0),   # carpus
    r"Rig[LR]FLegAnkle_[xyz]": _pd(80.0),
    # digits: light, like the raptor's toes/fingers
    r"Rig[LR][FB]LegDigit\d\d_[xyz]": _pd(8.0),
}


# NO armature, matching t800/dog_v2 (which train fine with armature=None).
# BUILT_IN_PD maps to IsaacLab's ImplicitActuatorCfg: its PD is integrated
# IMPLICITLY and is unconditionally stable, so the omega*dt criterion that
# motivated an earlier flat 0.02 (which applies to EXPLICIT integration)
# was never relevant. That 0.02 was 30x this robot's entire link inertia
# and ~90,000x a finger phalanx's, so every limb dragged a flywheel: the
# policy could stand but never accelerate a leg enough to walk or get up,
# and both ASE and AMP stalled with the discriminator above 90% accuracy.
# Creatures have no gearboxes, so zero is the physical answer too.
from dataclasses import replace as _replace
CONTROL_OVERRIDES = {
    _k: _replace(_v, armature=None) for _k, _v in CONTROL_OVERRIDES.items()
}


@dataclass
class TigerRobotConfig(RobotConfig):
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["RigLBLegAnkle"],
            "all_right_foot_bodies": ["RigRBLegAnkle"],
            # front paws play the "hands" role for battle tables later
            "all_left_hand_bodies": ["RigLFLegAnkle"],
            "all_right_hand_bodies": ["RigRFLegAnkle"],
            "head_body_name": ["RigHead"],
            "torso_body_name": ["RigChest"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "RigPelvis", "RigChest", "RigHead", "RigJaw1",
            "RigTail3", "RigTail5",
            "RigLBLeg1", "RigLBLeg2", "RigLBLegAnkle",
            "RigRBLeg1", "RigRBLeg2", "RigRBLegAnkle",
            "RigLFLeg1", "RigLFLeg2", "RigLFLegAnkle",
            "RigRFLeg1", "RigRFLeg2", "RigRFLegAnkle",
        ]
    )

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/tiger.xml",
            usd_asset_file_name="usd/tiger/tiger_flat.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/RigPelvis/",
            self_collisions=False,
        )
    )

    default_root_height: float = 1.14
    contact_bodies: List[str] = None

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info=dict(CONTROL_OVERRIDES),
        )
    )
