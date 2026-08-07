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
(kp = 2*effort, kd = kp/10), efforts scaled for 270 kg.
# Rescaled 2026-08-06 when the body was fitted to the mesh and
# scaled to a real large tiger: 3.05 m nose-to-tail, 270 kg at
# water density. Lengths moved by 0.9261 and mass by 270/200, so
# torque (m*g*L) moved by 1.2503 -- every effort below carries it.
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
    r".*_[xyz]": _pd(12.5),
    # Spine and neck are sized PER JOINT from the static cantilever load, not
    # from one blanket value. Measured on the 271 kg body: the mass forward of
    # RigSpine1 is 199 kg on a 91 cm lever, so merely HOLDING the pose wants
    # 1778 N.m against the 625 N.m it had -- a 0.35x margin, meaning the
    # actuator could not support the animal unaided, let alone accelerate it.
    # The legs carry most of that load on a quadruped, so it was not fatal,
    # but it left nothing for a sprint, and the sprint and jump clips ranked
    # worst in scan_actuator_feasibility.py.
    #
    # Each value is 1.5x its own static demand, so the margin is uniform down
    # the chain rather than generous at the tail and absent at the root:
    #   Spine1 199 kg @ 91 cm -> 1778 N.m    Neck1 40.5 kg @ 38 cm -> 150 N.m
    #   Spine2 174 kg @ 73 cm -> 1247 N.m    Neck2 35.0 kg @ 33 cm -> 113 N.m
    #   Spine3 148 kg @ 54 cm ->  789 N.m    Neck3 29.5 kg @ 28 cm ->  82 N.m
    # Chest, Neck4 and Head already cleared 1.5x and keep their values.
    # _pd() ties stiffness to effort (kp = 2 x effort); that is safe here
    # because BUILT_IN_PD maps to ImplicitActuatorCfg, whose PD is integrated
    # implicitly and is unconditionally stable at any gain.
    r"RigSpine[0-9]_[xyz]": _pd(625.1),
    r"RigSpine1_[xyz]": _pd(2667.1),
    r"RigSpine2_[xyz]": _pd(1870.1),
    r"RigSpine3_[xyz]": _pd(1183.1),
    r"RigChest_[xyz]": _pd(625.1),
    r"RigNeck[0-9]_[xyz]": _pd(112.5),
    r"RigNeck1_[xyz]": _pd(225.5),
    r"RigNeck2_[xyz]": _pd(170.0),
    r"RigNeck3_[xyz]": _pd(122.6),
    r"RigHead_[xyz]": _pd(75.0),
    r"RigJaw1_[xyz]": _pd(62.5),
    r"RigTail[0-9]_[xyz]": _pd(25.0),
    r"RigTail1_[xyz]": _pd(150.0),
    r"RigTail2_[xyz]": _pd(93.8),
    r"RigTail3_[xyz]": _pd(56.3),
    r"Rig[LR]BLeg1_[xyz]": _pd(350.1),   # hind hip
    r"Rig[LR]BLeg2_[xyz]": _pd(300.1),   # hind knee
    r"Rig[LR]BLeg3_[xyz]": _pd(200.0),   # hock
    r"Rig[LR]BLegAnkle_[xyz]": _pd(100.0),
    r"Rig[LR]FLegCollarbone_[xyz]": _pd(250.0),
    r"Rig[LR]ShoulderBlade1_[xyz]": _pd(250.0),
    r"Rig[LR]FLeg1_[xyz]": _pd(300.1),   # shoulder
    r"Rig[LR]FLeg2_[xyz]": _pd(250.0),   # elbow
    r"Rig[LR]FLeg3_[xyz]": _pd(175.0),   # carpus
    r"Rig[LR]FLegAnkle_[xyz]": _pd(100.0),
    # digits: light, like the raptor's toes/fingers
    r"Rig[LR][FB]LegDigit\d\d_[xyz]": _pd(10.0),
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

    # Bodies the AMP/ASE discriminator judges -- see raptor.py for why this
    # is separate from trackable_bodies_subset. The tiger needs it for the
    # same reason the raptor does: 32 digit segments the policy cannot
    # reproduce let the discriminator win on jitter instead of on gait.
    #
    # Differs from trackable_bodies_subset above in two ways that matter:
    #  - Leg3 is included. The chain is Leg1 -> Leg2 -> Leg3 -> Ankle, and
    #    the tracking list skips Leg3, which would leave a link unjudged --
    #    a paw position does not determine the leg's pose without it.
    #  - the collarbones and shoulder blades are included, so the front
    #    limbs are pinned at their base rather than floating off the chest.
    disc_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "RigPelvis", "RigSpine2", "RigChest", "RigNeck2",
            "RigHead", "RigJaw1", "RigTail3", "RigTail5",
            "RigLShoulderBlade1", "RigRShoulderBlade1",
            # hind limbs: every link
            "RigLBLeg1", "RigLBLeg2", "RigLBLeg3", "RigLBLegAnkle",
            "RigRBLeg1", "RigRBLeg2", "RigRBLeg3", "RigRBLegAnkle",
            # fore limbs: every link, from the collarbone down
            "RigLFLegCollarbone", "RigLFLeg1", "RigLFLeg2", "RigLFLeg3",
            "RigLFLegAnkle",
            "RigRFLegCollarbone", "RigRFLeg1", "RigRFLeg2", "RigRFLeg3",
            "RigRFLegAnkle",
            # toe TIPS only (tiger toes are two segments: Digit<n>1 -> <n>2)
            "RigLBLegDigit12", "RigLBLegDigit22", "RigLBLegDigit32",
            "RigLBLegDigit42",
            "RigRBLegDigit12", "RigRBLegDigit22", "RigRBLegDigit32",
            "RigRBLegDigit42",
            "RigLFLegDigit12", "RigLFLegDigit22", "RigLFLegDigit32",
            "RigLFLegDigit42",
            "RigRFLegDigit12", "RigRFLegDigit22", "RigRFLegDigit32",
            "RigRFLegDigit42",
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

    default_root_height: float = 1.0558
    contact_bodies: List[str] = None

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info=dict(CONTROL_OVERRIDES),
        )
    )
