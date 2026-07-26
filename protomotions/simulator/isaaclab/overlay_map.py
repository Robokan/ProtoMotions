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
