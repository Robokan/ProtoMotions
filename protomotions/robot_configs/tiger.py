# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Tiger robot (fbx2robot import of the UE Animalia Tiger_M pack).

70 bodies / 207 hinges (<Body>_x/_y/_z, world-aligned bind frames),
4 digits x 2 segments on each paw (cats are digitigrade; the dewclaws
and claw tips erode at the 0.14 m prune). Quadruped: all four ankles are
contact feet; front ankles double as the "hands" for battle tables later
(paw swipes), jaw is the bite. See RAPTOR_TIGER_PLAN.md.

MASS / SCALE (2026-08-07)
  Caudal density bump (pelvis/spine1/hind 1540, tail 1600) put the body at
  323 kg with ~58% fore load. Then geometrically scaled to a 260 kg target
  via scale_robot_mjcf.py:

      SCALE = (260 / 323.44)^(1/3) = 0.929804
      torque  s^4 = 0.747    armature s^5 = 0.695

  Effort numbers below are the pre-scale (323 kg) peaks; _pd multiplies by
  TORQUE_SCALE. Motions must be Froude-scaled by the same LENGTH scale.

DAMPING / ARMATURE
  kd/kp follows utahraptor's Froude correction (≈0.13 on the pre-scale
  body, then *sqrt(SCALE) for this shrink). ControlInfo.armature stays
  None so IsaacLab reads the MJCF/USD value.
"""

from dataclasses import dataclass, field, replace as _replace
from typing import Dict, List

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
)
from protomotions.robot_configs.dog_v2 import VELOCITY_LIMIT


# Length scale: 323.44 kg (caudal-density) body → 260 kg target.
SCALE = 0.929804
TORQUE_SCALE = SCALE ** 4          # 0.747420  -- m*g*L
# Pre-scale kd/kp was 0.13; Froude wants one more sqrt(s) when shrinking.
DAMPING_RATIO = 0.13 * (SCALE ** 0.5)  # ≈ 0.12535


def _pd(effort: float) -> ControlInfo:
    """Peak-torque (pre-scale) → PD gains on the 260 kg body.

    effort is the 323 kg peak; TORQUE_SCALE brings it down with s^4.
    """
    e = effort * TORQUE_SCALE
    kp = 2.0 * e
    return ControlInfo(
        stiffness=kp,
        damping=DAMPING_RATIO * kp,
        effort_limit=e,
        # smaller animal cycles limbs faster (1/sqrt(s))
        velocity_limit=VELOCITY_LIMIT / (SCALE ** 0.5),
    )


CONTROL_OVERRIDES = {
    # Catch-all FIRST (later matches override): digits, claws, tongue,
    # ears, whiskers and every other small bone of the full skeleton.
    r".*_[xyz]": _pd(12.5),
    # Spine and neck sized from static cantilever load on the 271–323 kg
    # body (1.5x static demand). TORQUE_SCALE applied inside _pd.
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


# armature=None → IsaacLab keeps the MJCF/USD armature (scaled with s^5).
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
            # ShoulderBlade1 and FLegCollarbone are OMITTED. They have the
            # worst mass-to-effort ratios on this robot -- 421 and 314
            # N.m/kg against 32 for RigLFLeg1 -- meaning the actuator can
            # fling them far faster than physics would move them. That is the
            # same profile as the raptor's ForeArm (60 N.m/kg), whose
            # inclusion took its discriminator to 95.5% accuracy with a 0.058
            # style reward and destroyed a walk that had worked. A body the
            # policy cannot track is a free win for the discriminator, not a
            # constraint on gait.
            #
            # The load-bearing limb chain IS fully observed below them, so the
            # elbow/knee poses are still pinned; only the scapular attachment
            # is hidden, and it mostly just translates with the chest.
            # hind limbs: every link
            "RigLBLeg1", "RigLBLeg2", "RigLBLeg3", "RigLBLegAnkle",
            "RigRBLeg1", "RigRBLeg2", "RigRBLeg3", "RigRBLegAnkle",
            # fore limbs: every link below the collarbone
            "RigLFLeg1", "RigLFLeg2", "RigLFLeg3", "RigLFLegAnkle",
            "RigRFLeg1", "RigRFLeg2", "RigRFLeg3", "RigRFLegAnkle",
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

    # Was 1.0558 on the 323 kg body; * SCALE for the 260 kg shrink.
    default_root_height: float = 0.9817
    contact_bodies: List[str] = None

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info=dict(CONTROL_OVERRIDES),
        )
    )
