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
    # This Reallusion export uses a REDUCED UE skeleton: 3 spine bones
    # (spine_01..03) and a single neck, not the full Manny's spine_01..05 +
    # neck_01/02. Map SOMA's 3 torso bones 1:1 onto the 3 spine bones and share
    # the single neck across SOMA's two neck bones (Neck2 local ~ identity).
    "Spine1": "spine_01",
    "Spine2": "spine_02",
    "Chest": "spine_03",
    "Neck1": "neck_01",
    "Neck2": "neck_01",
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


def _quat_to_mat(q) -> np.ndarray:
    """ufbx.Quat (x,y,z,w) -> rotation matrix [3,3]."""
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
            [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
            [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
        ]
    )


# Raw FBX space (Reallusion/UE: right-handed z-up) -> y-up right-handed, the
# frame create_motion_from_soma23_data expects: y_new = z_old, z_new = -y_old.
_ZUP_TO_YUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


# This ufbx binding SEGFAULTS when a Scene object is destroyed — keep every
# loaded scene alive for the process lifetime (callers should os._exit before
# interpreter teardown to skip the destructors entirely).
_SCENE_KEEPALIVE = []


def load_tpose_bind(tpose_path: Path):
    """Bind world rotations {manny_bone: [3,3]} from a T-pose reference FBX.

    The Reallusion combat clips store NO bind pose (scene.poses is empty) and
    start mid-stance, so calibrating from a clip is impossible — the shared
    MM_T_Pose.FBX export is the reference. Rotations are converted into the
    same y-up frame as load_fbx_frames so the retarget deltas cancel cleanly.
    """
    import ufbx

    scene = ufbx.load_file(str(tpose_path))
    _SCENE_KEEPALIVE.append(scene)
    wanted = set(MANNY_MAP.values())
    by_name = {n.name: n for n in scene.nodes}
    missing = wanted - set(by_name)
    if missing:
        raise ValueError(f"T-pose file missing bones: {sorted(missing)}")

    ordered = []
    seen = set()

    def _add(n):
        if n is None or id(n) in seen:
            return
        _add(n.parent)
        seen.add(id(n))
        ordered.append(n)

    for n in scene.nodes:
        _add(n)
    parent_idx = {id(n): (ordered.index(n.parent) if n.parent else -1)
                  for n in ordered}

    anim = scene.anim
    t0 = float(anim.time_begin)
    Rw = [None] * len(ordered)
    bind = {}
    for i, node in enumerate(ordered):
        tr = node.evaluate_transform(anim, t0)
        Rl = _quat_to_mat(tr.rotation)
        p = parent_idx[id(node)]
        Rw[i] = Rl if p < 0 else Rw[p] @ Rl
        if node.name in wanted:
            u, _, vt = np.linalg.svd(Rw[i])
            r = u @ vt
            if np.linalg.det(r) < 0:
                u[:, -1] *= -1
                r = u @ vt
            bind[node.name] = _ZUP_TO_YUP @ r
    return bind


def load_fbx_frames(fbx_path: Path, output_fps: int):
    """Load Manny bone world rotations + pelvis position per output frame.

    Returns (world_rots {manny_bone: [T,3,3]}, root_pos [T,3],
    bind {manny_bone: [3,3]}). Meters, y-up right-handed.

    Implementation notes (this ufbx binding):
    - ``scene.evaluate`` LEAKS unboundedly when called per frame (an OOM after
      ~100 calls) — so we evaluate per-node local transforms with
      ``node.evaluate_transform`` (fast, leak-free) and compose world
      transforms up the node hierarchy ourselves.
    - ``evaluate_transform`` returns RAW file-space transforms — LoadOpts axis/
      unit conversion does not apply — so the z-up/cm -> y-up/m conversion is
      done explicitly here (unit scale read from scene.settings).
    """
    import ufbx

    scene = ufbx.load_file(str(fbx_path))
    _SCENE_KEEPALIVE.append(scene)

    wanted = set(MANNY_MAP.values())
    by_name = {n.name: n for n in scene.nodes}
    missing = wanted - set(by_name)
    if missing:
        raise ValueError(f"missing Manny bones: {sorted(missing)}")

    # Unit scale to meters from the file settings (UE/Reallusion: cm = 0.01).
    unit_m = float(getattr(scene.settings, "unit_meters", 0.01) or 0.01)

    anim = scene.anim
    duration = max(float(anim.time_end) - float(anim.time_begin), 0.0)
    if duration <= 0:
        raise ValueError("no animation timeline")
    num_frames = int(round(duration * output_fps)) + 1
    if num_frames < 2:
        raise ValueError(f"animation too short: {duration:.3f}s")

    # Full node list in parent-before-child order for world composition.
    ordered = []
    seen = set()

    def _add(n):
        if n is None or id(n) in seen:
            return
        _add(n.parent)
        seen.add(id(n))
        ordered.append(n)

    for n in scene.nodes:
        _add(n)
    parent_idx = {id(n): (ordered.index(n.parent) if n.parent else -1)
                  for n in ordered}

    world_rots = {name: np.zeros((num_frames, 3, 3)) for name in wanted}
    root_pos = np.zeros((num_frames, 3))

    n_nodes = len(ordered)
    Rw = [None] * n_nodes
    tw = [None] * n_nodes
    for f in range(num_frames):
        t = float(anim.time_begin) + min(f / output_fps, duration)
        for i, node in enumerate(ordered):
            tr = node.evaluate_transform(anim, t)
            Rl = _quat_to_mat(tr.rotation) * np.array(
                [float(tr.scale.x), float(tr.scale.y), float(tr.scale.z)]
            )
            tl = np.array(
                [float(tr.translation.x), float(tr.translation.y),
                 float(tr.translation.z)]
            )
            p = parent_idx[id(node)]
            if p < 0:
                Rw[i], tw[i] = Rl, tl
            else:
                Rw[i] = Rw[p] @ Rl
                tw[i] = Rw[p] @ tl + tw[p]
        for i, node in enumerate(ordered):
            if node.name in wanted:
                # Orthonormalize (strip scale) then convert to y-up.
                u, _, vt = np.linalg.svd(Rw[i])
                r = u @ vt
                if np.linalg.det(r) < 0:
                    u[:, -1] *= -1
                    r = u @ vt
                world_rots[node.name][f] = _ZUP_TO_YUP @ r
                if node.name == "pelvis":
                    root_pos[f] = (_ZUP_TO_YUP @ tw[i]) * unit_m

    # Bind pose calibration from the file's own stored bind pose, when
    # present. Reallusion combat clips have NONE (scene.poses is empty) and
    # start mid-stance — for those the caller MUST supply a T-pose reference
    # (see load_tpose_bind); silently falling back to frame 0 produced
    # T-pose-started garbage retargets.
    bind = {}
    for pose in getattr(scene, "poses", []):
        if not getattr(pose, "is_bind_pose", False):
            continue
        for bone_pose in pose.bone_poses:
            bone_node = getattr(bone_pose, "bone_node", None)
            if bone_node is not None and bone_node.name in wanted:
                bind.setdefault(
                    bone_node.name,
                    _ZUP_TO_YUP @ _rot_from_matrix(bone_pose.bone_to_world),
                )

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
    parser.add_argument(
        "--min-height",
        type=float,
        default=-0.05,
        help="Quality filter: reject clips with bodies below this height (m). "
        "Retargeted strikes commonly dip a foot slightly below ground; "
        "-0.15 is a reasonable batch setting with visual review after.",
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=15.0,
        help="Quality filter: reject clips with body velocities above this "
        "(m/s). Fast spinning kicks can legitimately reach ~15-20.",
    )
    parser.add_argument(
        "--single-file",
        type=str,
        default=None,
        help="(internal) convert only this file name from --input-dir",
    )
    parser.add_argument(
        "--tpose-file",
        type=Path,
        default=None,
        help="T-pose reference FBX for bind calibration (e.g. MM_T_Pose.FBX). "
        "REQUIRED for exports whose clips carry no stored bind pose "
        "(Reallusion combat clips).",
    )
    args = parser.parse_args()

    # This ufbx binding SEGFAULTS on the second load_file within one process,
    # so batches run each file in an isolated child process (re-exec self with
    # --single-file). A child converts exactly one FBX directly.
    if args.single_file is None:
        import subprocess

        all_fbx = sorted(
            set(args.input_dir.glob("**/*.fbx"))
            | set(args.input_dir.glob("**/*.FBX"))
        )
        print(f"{len(all_fbx)} FBX files (isolated child per file)")
        ok = bad = 0
        for fbx in all_fbx:
            cmd = [sys.executable, __file__,
                   "--input-dir", str(args.input_dir),
                   "--output-dir", str(args.output_dir),
                   "--output-fps", str(args.output_fps),
                   "--min-height", str(args.min_height),
                   "--max-velocity", str(args.max_velocity),
                   "--single-file", fbx.name]
            if args.tpose_file is not None:
                cmd += ["--tpose-file", str(args.tpose_file)]
            if args.mirror:
                cmd.append("--mirror")
            if args.force_remake:
                cmd.append("--force-remake")
            if args.ignore_filter:
                cmd.append("--ignore-filter")
            r = subprocess.run(cmd, capture_output=True, text=True)
            out = (r.stdout or "") + (r.stderr or "")
            for line in out.splitlines():
                if line.strip().startswith(("ok:", "FAIL")):
                    print(f"  {line.strip()}")
            # Judge success by the artifact, not the exit code: this ufbx
            # binding often segfaults at interpreter teardown AFTER the
            # .motion files were saved successfully.
            if (args.output_dir / f"{fbx.stem}.motion").exists():
                ok += 1
            else:
                bad += 1
                if r.returncode != 0 and "ok:" not in out:
                    print(f"  CRASH {fbx.name} (exit {r.returncode})")
        print(f"\nfiles ok: {ok} | files failed/crashed: {bad}")
        return

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
    fbx_files = [f for f in fbx_files if f.name == args.single_file]

    tpose_bind = (
        load_tpose_bind(args.tpose_file) if args.tpose_file is not None else None
    )

    converted, failed = 0, []
    for fbx in fbx_files:
        try:
            world_rots, root_pos, bind = load_fbx_frames(fbx, args.output_fps)
            if tpose_bind is not None:
                bind = tpose_bind
            if not bind:
                raise ValueError(
                    "clip stores no bind pose — pass --tpose-file (e.g. "
                    "MM_T_Pose.FBX) for bind calibration"
                )
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
                # Ground alignment: retarget copies rotations but SOMA's leg
                # proportions differ from Manny's, so clips systematically ride
                # above/below z=0. Shift so the dominant contact level (median
                # of per-frame min body z — feet when standing, torso for
                # ground work) sits at 0, then rebuild the motion.
                zmin = motion_dict["rigid_body_pos"][..., 2].min(dim=1).values
                offset = float(zmin.median())
                # Bound the worst transient dip (kick follow-throughs, fall
                # impacts): after median alignment, lift further if any body
                # would still sink more than 8 cm below ground.
                worst = float(zmin.min()) - offset
                if worst < -0.08:
                    offset += worst + 0.08
                if abs(offset) > 0.01:
                    root = root.clone()
                    root[:, 1] -= offset  # y-up height, pre-conversion
                    motion = create_motion_from_soma23_data(
                        local, root, kinematic_info, fps=args.output_fps
                    )
                    motion_dict = (
                        motion.to_dict() if hasattr(motion, "to_dict") else motion
                    )
                if not args.ignore_filter:
                    reason = quality_check(
                        motion_dict,
                        max_velocity=args.max_velocity,
                        min_height=args.min_height,
                    )
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

    # Skip interpreter teardown: destroying kept-alive ufbx scenes segfaults
    # this binding (the source of the exit-code -11 "crashes" after successful
    # saves). Work is flushed; exit hard with a real code.
    import os
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
