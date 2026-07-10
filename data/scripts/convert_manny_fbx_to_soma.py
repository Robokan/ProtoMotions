# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert UE5 Manny-skeleton FBX animations to SOMA23 ``.motion`` files.

Intake path for Reallusion/Unreal combat mocap retargeted to the standard
UE5 mannequin ("Manny"): FBX is read directly with ``ufbx`` (no Blender, no
BVH intermediate), bone world orientations are retargeted onto the SOMA
23-body skeleton via bind-pose offset calibration, and the repo's standard
motion builder produces the final ``.motion`` (FK, velocities, contacts).

Retarget math (world-rotation copy):
    offset(j)      = manny_bind_world(map(j))^-1            (calibration)
    soma_world(j)  = manny_world(map(j)) @ offset(j)
    soma_local(j)  = soma_world(parent(j))^-1 @ soma_world(j)

Copying WORLD orientations means Manny's five spine bones collapse onto
SOMA's three naturally — unmapped bones' rotations are absorbed by the
chain. The bind-pose offset also absorbs Manny's A-pose vs SOMA's T-pose.

Usage:
    python data/scripts/convert_manny_fbx_to_soma.py \
        --input-dir /path/to/fbx/ --output-dir motions/combat_reallusion \
        --output-fps 30 --mirror
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# SOMA23 body -> Manny bone.
# World-rotation retarget: Manny bones not listed (spine_01/04, twists, IK,
# fingers) are absorbed by the chain.
MANNY_MAP = {
    "Hips": "pelvis",
    "Spine1": "spine_02",
    "Spine2": "spine_03",
    "Chest": "spine_05",
    "Neck1": "neck_01",
    "Neck2": "neck_02",
    "Head": "head",
    "RightShoulder": "clavicle_r",
    "RightArm": "upperarm_r",
    "RightForeArm": "lowerarm_r",
    "RightHand": "hand_r",
    "LeftShoulder": "clavicle_l",
    "LeftArm": "upperarm_l",
    "LeftForeArm": "lowerarm_l",
    "LeftHand": "hand_l",
    "RightLeg": "thigh_r",
    "RightShin": "calf_r",
    "RightFoot": "foot_r",
    "RightToeBase": "ball_r",
    "LeftLeg": "thigh_l",
    "LeftShin": "calf_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "ball_l",
}

# Left/right swap for mirror augmentation
MIRROR_SWAP = {}
for _n in list(MANNY_MAP):
    if _n.startswith("Left"):
        MIRROR_SWAP[_n] = "Right" + _n[4:]
        MIRROR_SWAP["Right" + _n[4:]] = _n
    elif not _n.startswith("Right"):
        MIRROR_SWAP[_n] = _n


def _rot_from_matrix(m) -> np.ndarray:
    """ufbx.Matrix (columns c0..c3) -> orthonormal rotation [3,3].

    UE exports frequently carry uniform scale (unit conversion); strip it
    with a polar decomposition so pure rotations reach the retarget.
    """
    r = np.array(
        [
            [m.c0.x, m.c1.x, m.c2.x],
            [m.c0.y, m.c1.y, m.c2.y],
            [m.c0.z, m.c1.z, m.c2.z],
        ],
        dtype=np.float64,
    )
    u, _, vt = np.linalg.svd(r)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def _pos_from_matrix(m) -> np.ndarray:
    return np.array([m.c3.x, m.c3.y, m.c3.z], dtype=np.float64)


def load_fbx_frames(fbx_path: Path, output_fps: int):
    """Load Manny bone world rotations + pelvis position per output frame.

    Returns (world_rots {manny_bone: [T,3,3]}, root_pos [T,3],
    bind {manny_bone: [3,3]}). Meters, y-up right-handed (converted by ufbx).
    """
    import ufbx

    opts = ufbx.LoadOpts(
        target_axes=ufbx.axes_right_handed_y_up,
        target_unit_meters=1.0,
    )
    scene = ufbx.load_file(str(fbx_path), opts)

    wanted = set(MANNY_MAP.values())
    nodes = {n.name: n for n in scene.nodes if n.name in wanted}
    missing = wanted - set(nodes)
    if missing:
        raise ValueError(f"missing Manny bones: {sorted(missing)}")

    anim = scene.anim
    duration = max(float(anim.time_end) - float(anim.time_begin), 0.0)
    if duration <= 0:
        raise ValueError("no animation timeline")
    num_frames = int(round(duration * output_fps)) + 1
    if num_frames < 2:
        raise ValueError(f"animation too short: {duration:.3f}s")

    # Bind pose calibration: prefer an explicit bind pose, fall back to the
    # first animation frame (valid when clips start from the reference pose).
    bind = {}
    for pose in getattr(scene, "poses", []):
        if not getattr(pose, "is_bind_pose", False):
            continue
        for bone_pose in pose.bone_poses:
            bone_node = getattr(bone_pose, "bone_node", None)
            if bone_node is not None and bone_node.name in wanted:
                bind.setdefault(
                    bone_node.name, _rot_from_matrix(bone_pose.bone_to_world)
                )

    world_rots = {name: np.zeros((num_frames, 3, 3)) for name in nodes}
    root_pos = np.zeros((num_frames, 3))

    for f in range(num_frames):
        t = float(anim.time_begin) + min(f / output_fps, duration)
        eval_scene = ufbx.evaluate_scene(scene, anim, t)
        for name, node in nodes.items():
            enode = eval_scene.nodes[node.typed_id]
            world_rots[name][f] = _rot_from_matrix(enode.node_to_world)
            if name == "pelvis":
                root_pos[f] = _pos_from_matrix(enode.node_to_world)

    for name in nodes:
        bind.setdefault(name, world_rots[name][0].copy())

    return world_rots, root_pos, bind


def retarget_to_soma(world_rots, root_pos, bind, kinematic_info, mirror=False):
    """Manny world rotations -> SOMA23 local rotation matrices [T,23,3,3]."""
    body_names = kinematic_info.body_names
    parents = kinematic_info.parent_indices
    num_frames = root_pos.shape[0]

    if mirror:
        name_map = {soma: MANNY_MAP[MIRROR_SWAP[soma]] for soma in MANNY_MAP}
    else:
        name_map = dict(MANNY_MAP)

    # Mirror across the YZ plane (x -> -x in the y-up frame)
    S = np.diag([-1.0, 1.0, 1.0])

    soma_world = np.zeros((num_frames, len(body_names), 3, 3))
    for j, soma_name in enumerate(body_names):
        manny_name = name_map[soma_name]
        world = world_rots[manny_name] @ bind[manny_name].T
        if mirror:
            world = S[None] @ world @ S[None]
        soma_world[:, j] = world

    local = np.zeros_like(soma_world)
    for j in range(len(body_names)):
        p = parents[j]
        if p < 0:
            local[:, j] = soma_world[:, j]
        else:
            parent_inv = np.transpose(soma_world[:, p], (0, 2, 1))
            local[:, j] = parent_inv @ soma_world[:, j]

    out_root = root_pos.copy()
    if mirror:
        out_root[:, 0] = -out_root[:, 0]

    return (
        torch.from_numpy(local).float(),
        torch.from_numpy(out_root).float(),
    )


def quality_check(motion_dict, max_velocity=15.0, min_height=-0.05):
    """Final z-up motion sanity: velocity spikes and below-ground bodies."""
    vel = motion_dict["rigid_body_vel"]
    pos = motion_dict["rigid_body_pos"]
    if float(vel.norm(dim=-1).max()) > max_velocity:
        return "max velocity exceeded"
    if float(pos[..., 2].min()) < min_height:
        return "below-ground bodies"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-fps", type=int, default=30)
    parser.add_argument(
        "--mirror", action="store_true", help="Also emit L/R-mirrored clips"
    )
    parser.add_argument("--force-remake", action="store_true")
    parser.add_argument("--ignore-filter", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from protomotions.components.pose_lib import extract_kinematic_info
    from data.scripts.convert_soma23_to_proto import create_motion_from_soma23_data

    kinematic_info = extract_kinematic_info(
        "protomotions/data/assets/mjcf/soma23_humanoid.xml"
    )
    assert kinematic_info.num_bodies == 23

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fbx_files = sorted(
        set(args.input_dir.glob("**/*.fbx")) | set(args.input_dir.glob("**/*.FBX"))
    )
    print(f"{len(fbx_files)} FBX files")

    converted, failed = 0, []
    for fbx in fbx_files:
        try:
            world_rots, root_pos, bind = load_fbx_frames(fbx, args.output_fps)
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            failed.append((fbx.name, str(exc)))
            continue
        variants = [("", False)] + ([("_M", True)] if args.mirror else [])
        for suffix, mirror in variants:
            dst = args.output_dir / f"{fbx.stem}{suffix}.motion"
            if dst.exists() and not args.force_remake:
                continue
            try:
                local, root = retarget_to_soma(
                    world_rots, root_pos, bind, kinematic_info, mirror=mirror
                )
                motion = create_motion_from_soma23_data(
                    local, root, kinematic_info, fps=args.output_fps
                )
                motion_dict = (
                    motion.to_dict() if hasattr(motion, "to_dict") else motion
                )
                if not args.ignore_filter:
                    reason = quality_check(motion_dict)
                    if reason:
                        failed.append((dst.name, reason))
                        continue
                torch.save(motion_dict, dst)
                converted += 1
                print(f"  ok: {dst.name}")
            except Exception as exc:  # noqa: BLE001
                failed.append((dst.name, str(exc)))

    print(f"\nconverted: {converted} | failed: {len(failed)}")
    for name, reason in failed[:20]:
        print(f"  FAIL {name}: {reason[:90]}")


if __name__ == "__main__":
    main()
