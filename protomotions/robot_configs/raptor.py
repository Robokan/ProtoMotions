# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Raptor robot (fbx2robot import of the UE RaptorDinosaur pack).

27 bodies / 78 hinges (<Body>_x/_y/_z, world-aligned bind frames),
40 kg, hips at 0.51 m. Bipedal theropod: legs carry locomotion, jaw and
feet are the strike surfaces, tail balances. PD gains follow the dog_v2
translation (kp = 2*effort, kd = kp/10), efforts scaled for 40 kg.
See RAPTOR_TIGER_PLAN.md.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
)
from protomotions.robot_configs.dog_v2 import _pd


CONTROL_OVERRIDES = {
    # Catch-all FIRST (later matches override): digits, tongue, eyes,
    # sockets and every other small bone of the full skeleton.
    r".*_[xyz]": _pd(6.0),
    r"Spine1?_[xyz]": _pd(90.0),
    r"Neck[0-9]?_[xyz]": _pd(35.0),
    r"Head_[xyz]": _pd(25.0),
    r"Jaw_[xyz]": _pd(20.0),
    r"Tail[0-9]_[xyz]": _pd(15.0),
    # Tail gains sized from measured gravity-hold torque x2 (the tail is a
    # 10 kg cantilever off the hips; the base needed 50 Nm against a 6 Nm
    # limit). Gradient from base to tip.
    r"Tail_[xyz]": _pd(100.0),
    r"Tail1_[xyz]": _pd(80.0),
    r"Tail2_[xyz]": _pd(55.0),
    r"Tail3_[xyz]": _pd(30.0),
    r"(Left|Right)UpLeg_[xyz]": _pd(120.0),
    r"(Left|Right)Leg_[xyz]": _pd(100.0),
    r"(Left|Right)Foot_[xyz]": _pd(60.0),
    r"(Left|Right)ToeBase_[xyz]": _pd(25.0),
    r"(Left|Right)Shoulder_[xyz]": _pd(30.0),
    r"(Left|Right)Arm_[xyz]": _pd(25.0),
    r"(Left|Right)ForeArm_[xyz]": _pd(20.0),
    r"(Left|Right)Hand_[xyz]": _pd(10.0),
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
class RaptorRobotConfig(RobotConfig):
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["LeftToeBase"],
            "all_right_foot_bodies": ["RightToeBase"],
            # forelimb claws play the "hands" role for battle tables later
            "all_left_hand_bodies": ["LeftHand"],
            "all_right_hand_bodies": ["RightHand"],
            "head_body_name": ["Head"],
            "torso_body_name": ["Spine1"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "Hips", "Spine1", "Head", "Jaw", "Tail3", "Tail5",
            "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
            "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
            "LeftArm", "LeftHand", "RightArm", "RightHand",
        ]
    )

    # Bodies the AMP/ASE discriminator judges. SEPARATE from
    # trackable_bodies_subset on purpose: that field means "tracking
    # targets" and is consumed by GPC and masked-mimic, and robots size it
    # for that job -- t800 lists six. Feeding six bodies to a discriminator
    # would leave almost every joint unjudged, so the two must not share a
    # field. Leave this None and the discriminator sees every body, which
    # is the right default for a robot without hard-to-actuate extremities.
    #
    # The raptor needs it: showing all 68 bodies let the discriminator win
    # on the 36 digit segments (~1.4 g, 6 N.m actuators, constantly hit by
    # ground contact) instead of on gait, and the policy learned to stand
    # and nothing more.
    #
    # ARMS ARE DELIBERATELY UNDER-OBSERVED: Arm and Hand only, no Shoulder
    # and no ForeArm. Adding those two per side (30 bodies -> 34) is exactly
    # what broke a working walk. Over 7828 epochs the discriminator went to
    # 95.5% accuracy and kept pulling away -- agent logit -4.60 -> -5.03 --
    # while style reward sat at 0.058, and episodes were being terminated at
    # ~4 s because nearly every transition scored under the 0.02 threshold.
    #
    # The reasoning that motivated adding them is sound in the abstract: an
    # end effector does not determine a limb's pose, so with only Arm and
    # Hand seen the elbow may sit anywhere on the circle around the
    # shoulder-to-hand axis at zero style cost. It is wrong HERE for the same
    # reason the digits are excluded: the raptor's forelimb bones are small
    # and light, the animation flicks them faster than these actuators can
    # follow, and a body whose motion the policy cannot reproduce is a free
    # win for the discriminator rather than a constraint on gait.
    #
    # If the elbow pose ever matters (fighting), pay for it with a cost term
    # or a torque-feasible reference, NOT by handing the discriminator a
    # body it can separate on.
    disc_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "Hips", "Spine1", "Head", "Jaw", "Tail3", "Tail5",
            "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
            "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
            "LeftArm", "LeftHand",
            "RightArm", "RightHand",
            # Digit TIPS by name. Intermediate phalanges stay hidden (they
            # are ~1.4 g on 6 N.m actuators and the policy cannot reproduce
            # them, which is what let the discriminator win on jitter), but
            # the tips must be visible or planting them costs nothing.
            "LeftHandIndex3", "LeftHandMiddle3", "LeftHandRing3",
            "RightHandIndex3", "RightHandMiddle3", "RightHandRing3",
            "LeftFootIndex3", "LeftFootMiddle3", "LeftFootRing3",
            "RightFootIndex3", "RightFootMiddle3", "RightFootRing3",
        ]
    )

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/raptor.xml",
            usd_asset_file_name="usd/raptor/raptor_flat.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/Hips/",
            self_collisions=False,
        )
    )

    default_root_height: float = 0.51
    contact_bodies: List[str] = None

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info=dict(CONTROL_OVERRIDES),
        )
    )
