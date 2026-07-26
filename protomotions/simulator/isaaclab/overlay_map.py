# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""SOMA23 -> Reallusion-CC-rig joint map for the skinned overlay.

Maps each SOMA23 rigid body to the character bone it drives. Character
bones not listed (twist chains, individual toes/fingers, face) hold their
bind pose — the same policy as the SOMA->BVH exporter's unmapped joints.

The CC bone names below match the omni.anim.people characters
(RL_BoneRoot skeleton family, see
`protomotions/data/assets/overlay/rl_character_skeleton.json`).
"""

# SOMA23 body name -> CC bone name (leaf name; resolved to full joint path
# at load time). Order does not matter; the driver walks the character
# hierarchy.
SOMA23_TO_CC = {
    "Hips": "Hip",
    "Spine1": "Waist",
    "Spine2": "Spine01",
    "Chest": "Spine02",
    "Neck1": "NeckTwist01",
    "Neck2": "NeckTwist02",
    "Head": "Head",
    "LeftShoulder": "L_Clavicle",
    "LeftArm": "L_Upperarm",
    "LeftForeArm": "L_Forearm",
    "LeftHand": "L_Hand",
    "RightShoulder": "R_Clavicle",
    "RightArm": "R_Upperarm",
    "RightForeArm": "R_Forearm",
    "RightHand": "R_Hand",
    "LeftLeg": "L_Thigh",
    "LeftShin": "L_Calf",
    "LeftFoot": "L_Foot",
    "LeftToeBase": "L_ToeBase",
    "RightLeg": "R_Thigh",
    "RightShin": "R_Calf",
    "RightFoot": "R_Foot",
    "RightToeBase": "R_ToeBase",
}

# Per-body frame constants c_b (wxyz, rel to Hips at the correspondence
# pose). Measured 2026-07-26 via MuJoCo FK: the soma23 MJCF zero pose is a
# T-pose with EVERY body frame world-aligned (identity rel-to-hips), and it
# matches the CC character's bind pose limb-for-limb (arms +-X, toes -Y,
# character faces -Y). So all constants are identity — the overlay's
# rest_rel.get(body, identity) fallback covers everything. (Motion-lib quats
# are XYZW; the overlay reorders to wxyz internally.)
SOMA23_REST_REL = {}

# soma23 T-pose (MJCF zero) body positions relative to Hips, meters — the
# robot side of the overlay auto-fit (uniform scale + offset onto the
# character's bind joint positions).
SOMA23_TPOSE_POS = {
    "Hips": (+0.0000, +0.0000, +0.0000),
    "Spine1": (+0.0000, +0.0005, +0.0500),
    "Spine2": (+0.0000, +0.0008, +0.1213),
    "Chest": (+0.0000, +0.0090, +0.1968),
    "Neck1": (-0.0018, +0.0145, +0.4599),
    "Neck2": (-0.0018, -0.0085, +0.5370),
    "Head": (-0.0018, -0.0280, +0.5983),
    "LeftShoulder": (+0.0162, -0.0421, +0.4292),
    "LeftArm": (+0.1654, +0.0129, +0.4292),
    "LeftForeArm": (+0.4528, +0.0129, +0.4292),
    "LeftHand": (+0.7237, +0.0129, +0.4292),
    "RightShoulder": (-0.0138, -0.0431, +0.4286),
    "RightArm": (-0.1642, +0.0124, +0.4286),
    "RightForeArm": (-0.4516, +0.0124, +0.4286),
    "RightHand": (-0.7229, +0.0124, +0.4286),
    "LeftLeg": (+0.1004, -0.0260, -0.0843),
    "LeftShin": (+0.1004, -0.0180, -0.5165),
    "LeftFoot": (+0.1004, +0.0168, -0.9381),
    "LeftToeBase": (+0.1004, -0.1155, -0.9887),
    "RightLeg": (-0.1005, -0.0262, -0.0830),
    "RightShin": (-0.1005, -0.0181, -0.5166),
    "RightFoot": (-0.1005, +0.0167, -0.9378),
    "RightToeBase": (-0.1005, -0.1161, -0.9886),
}

# soma23 kinematic parents (for per-segment limb-length matching).
SOMA23_PARENT = {
    "Spine1": "Hips", "Spine2": "Spine1", "Chest": "Spine2",
    "Neck1": "Chest", "Neck2": "Neck1", "Head": "Neck2",
    "LeftShoulder": "Chest", "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
    "RightShoulder": "Chest", "RightArm": "RightShoulder",
    "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "LeftLeg": "Hips", "LeftShin": "LeftLeg",
    "LeftFoot": "LeftShin", "LeftToeBase": "LeftFoot",
    "RightLeg": "Hips", "RightShin": "RightLeg",
    "RightFoot": "RightShin", "RightToeBase": "RightFoot",
}

# ---- Epic-skeleton (UE5) characters, e.g. the Red Samurai ----------------
# SOMA23 body -> UE bone. SOMA's 3 spine links spread over UE's 5 (spine_01
# and spine_03 hold bind). Fingers exist (index_01_l style names) but the
# fist synthesizer currently targets CC names only.
SOMA23_TO_UE = {
    "Hips": "pelvis",
    "Spine1": "spine_02",
    "Spine2": "spine_04",
    "Chest": "spine_05",
    "Neck1": "neck_01",
    "Neck2": "neck_02",
    "Head": "head",
    "LeftShoulder": "clavicle_l",
    "LeftArm": "upperarm_l",
    "LeftForeArm": "lowerarm_l",
    "LeftHand": "hand_l",
    "RightShoulder": "clavicle_r",
    "RightArm": "upperarm_r",
    "RightForeArm": "lowerarm_r",
    "RightHand": "hand_r",
    "LeftLeg": "thigh_l",
    "LeftShin": "calf_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "ball_l",
    "RightLeg": "thigh_r",
    "RightShin": "calf_r",
    "RightFoot": "foot_r",
    "RightToeBase": "ball_r",
}

# UE characters bind in an A-POSE (arms ~55 deg down), unlike the SOMA/CC
# T-pose: the arm chain carries the A-rotation as its rest constant
# (measured from red_samurai.usd bind segment directions; wxyz).
UE_REST_REL = {
    "LeftArm": (+0.887703, +0.000000, +0.460228, -0.013188),
    "LeftForeArm": (+0.856825, +0.000000, +0.351731, -0.377010),
    "LeftHand": (+0.856825, +0.000000, +0.351731, -0.377010),
    "RightArm": (+0.887703, +0.000000, -0.460228, +0.013188),
    "RightForeArm": (+0.856825, +0.000000, -0.351731, +0.377010),
    "RightHand": (+0.856825, +0.000000, -0.351731, +0.377010),
}

# UE parent map for limb-length matching (all mapped pairs are direct in the
# Epic hierarchy except the spread spine, which routes through held bones —
# the overlay compensates intermediates automatically).
SOMA23_PARENT_UE = {
    "Spine1": "Hips", "Spine2": "Spine1", "Chest": "Spine2",
    "Neck1": "Chest", "Neck2": "Neck1", "Head": "Neck2",
    "LeftShoulder": "Chest", "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
    "RightShoulder": "Chest", "RightArm": "RightShoulder",
    "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "LeftLeg": "Hips", "LeftShin": "LeftLeg",
    "LeftFoot": "LeftShin", "LeftToeBase": "LeftFoot",
    "RightLeg": "Hips", "RightShin": "RightLeg",
    "RightFoot": "RightShin", "RightToeBase": "RightFoot",
}
