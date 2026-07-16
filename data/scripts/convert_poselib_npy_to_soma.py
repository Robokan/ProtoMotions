# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert poselib SkeletonMotion .npy clips (ASE/IsaacLabASE lineage, UE
Manny skeleton) to SOMA23 ``.motion`` files.

The .npy stores LOCAL quaternions [T, J, 4] + root translation [T, 3] on the
68-node UE skeleton in z-up centimeters. World rotations are composed along
the parent chain in numpy, converted to the same y-up frame as the FBX
pipeline, and retargeted through the shared world-rotation-delta machinery
(MANNY_MAP + retarget_to_soma) with the bind pose taken from MM_T_Pose.FBX —
poselib skeletons carry no rest rotations, so the clip itself cannot provide
the bind.

Usage:
    python data/scripts/convert_poselib_npy_to_soma.py \\
        --input-dir /workspace/sparkpack/reallusion_fbx/drunken \\
        --output-dir data/motions/drunken_reallusion \\
        --tpose-file /workspace/sparkpack/reallusion_fbx/MM_T_Pose.FBX --mirror
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.scripts.convert_manny_fbx_to_soma import (  # noqa: E402
    MANNY_MAP,
    _ZUP_TO_YUP,
    load_tpose_bind,
    retarget_to_soma,
    quality_check,
)


def _quats_to_mats(q: np.ndarray) -> np.ndarray:
    """Quaternion array [..., 4] (x,y,z,w) -> rotation matrices [..., 3, 3]."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = (q * q).sum(-1)
    s = 2.0 / np.clip(n, 1e-12, None)
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - s * (y * y + z * z)
    m[..., 0, 1] = s * (x * y - w * z)
    m[..., 0, 2] = s * (x * z + w * y)
    m[..., 1, 0] = s * (x * y + w * z)
    m[..., 1, 1] = 1 - s * (x * x + z * z)
    m[..., 1, 2] = s * (y * z - w * x)
    m[..., 2, 0] = s * (x * z - w * y)
    m[..., 2, 1] = s * (y * z + w * x)
    m[..., 2, 2] = 1 - s * (x * x + y * y)
    return m


def _unwrap(v):
    """poselib fields may be {'arr': ..., 'context': ...} dicts."""
    if hasattr(v, "keys") and "arr" in v:
        return np.asarray(v["arr"])
    return np.asarray(v)


def load_npy_frames(npy_path: Path, output_fps: int):
    """Poselib npy -> (world_rots {manny_bone: [T,3,3]}, root_pos [T,3]).

    Output is meters, y-up — the frame the shared retarget expects.
    """
    data = np.load(npy_path, allow_pickle=True).item()
    if not data.get("is_local", False):
        raise ValueError("expected local rotations (is_local=True)")

    st = data["skeleton_tree"]
    names = list(st["node_names"])
    parents = np.asarray(_unwrap(st["parent_indices"]), dtype=np.int64)
    quats = _unwrap(data["rotation"]).astype(np.float64)  # [T, J, 4] local
    root_t = _unwrap(data["root_translation"]).astype(np.float64)  # [T, 3]
    fps = int(data.get("fps", 30))

    wanted = set(MANNY_MAP.values())
    missing = wanted - set(names)
    if missing:
        raise ValueError(f"missing bones: {sorted(missing)}")

    local = _quats_to_mats(quats)  # [T, J, 3, 3]
    T, J = local.shape[:2]
    world = np.empty_like(local)
    for j in range(J):  # parent_indices are topologically ordered in poselib
        p = parents[j]
        world[:, j] = local[:, j] if p < 0 else world[:, p] @ local[:, j]

    # Resample to output fps by nearest-frame (fps is usually already 30).
    if fps != output_fps:
        idx = np.clip(
            np.round(np.arange(0, T * output_fps / fps) * fps / output_fps),
            0, T - 1,
        ).astype(np.int64)
        world = world[idx]
        root_t = root_t[idx]

    name_to_j = {n: i for i, n in enumerate(names)}
    world_rots = {
        bone: _ZUP_TO_YUP[None] @ world[:, name_to_j[bone]] for bone in wanted
    }
    root_pos = (root_t @ _ZUP_TO_YUP.T) * 0.01  # cm z-up -> m y-up
    return world_rots, root_pos


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tpose-file", type=Path, required=True)
    parser.add_argument("--output-fps", type=int, default=30)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--force-remake", action="store_true")
    parser.add_argument("--ignore-filter", action="store_true")
    parser.add_argument("--min-height", type=float, default=-0.15)
    parser.add_argument("--max-velocity", type=float, default=20.0)
    parser.add_argument(
        "--single-file",
        type=str,
        default=None,
        help="(internal) convert only this file name from --input-dir",
    )
    args = parser.parse_args()

    # Batches crash natively after a few clips in one process (accumulation in
    # the motion-build path; single clips always succeed) — so, like the FBX
    # converter, run each file in an isolated child (re-exec with
    # --single-file) and judge success by the artifact on disk.
    if args.single_file is None:
        import os
        import subprocess

        all_npy = sorted(args.input_dir.glob("*.npy"))
        print(f"{len(all_npy)} npy files (isolated child per file)", flush=True)
        ok = bad = 0
        for f in all_npy:
            cmd = [sys.executable, __file__,
                   "--input-dir", str(args.input_dir),
                   "--output-dir", str(args.output_dir),
                   "--tpose-file", str(args.tpose_file),
                   "--output-fps", str(args.output_fps),
                   "--min-height", str(args.min_height),
                   "--max-velocity", str(args.max_velocity),
                   "--single-file", f.name]
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
                    print(f"  {line.strip()}", flush=True)
            if (args.output_dir / f"{f.stem}.motion").exists():
                ok += 1
            else:
                bad += 1
                if "FAIL" not in out:
                    print(f"  CRASH {f.name} (exit {r.returncode})", flush=True)
        print(f"\nfiles ok: {ok} | files failed/crashed: {bad}", flush=True)
        os._exit(0)

    from protomotions.components.pose_lib import extract_kinematic_info
    from data.scripts.convert_soma23_to_proto import create_motion_from_soma23_data

    kinematic_info = extract_kinematic_info(
        "protomotions/data/assets/mjcf/soma23_humanoid.xml"
    )
    bind = load_tpose_bind(args.tpose_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(args.input_dir.glob("*.npy"))
             if f.name == args.single_file]

    converted, failed = 0, []
    for f in files:
        try:
            world_rots, root_pos = load_npy_frames(f, args.output_fps)
        except Exception as exc:  # noqa: BLE001
            failed.append((f.name, str(exc)))
            continue
        variants = [("", False)] + ([("_M", True)] if args.mirror else [])
        for suffix, mirror in variants:
            dst = args.output_dir / f"{f.stem}{suffix}.motion"
            if dst.exists() and not args.force_remake:
                continue
            try:
                local, root = retarget_to_soma(
                    world_rots, root_pos, bind, kinematic_info, mirror=mirror
                )
                motion = create_motion_from_soma23_data(
                    local, root, kinematic_info, fps=args.output_fps
                )
                md = motion.to_dict() if hasattr(motion, "to_dict") else motion
                # Ground alignment + dip bound (same policy as the FBX path).
                zmin = md["rigid_body_pos"][..., 2].min(dim=1).values
                offset = float(zmin.median())
                worst = float(zmin.min()) - offset
                if worst < -0.08:
                    offset += worst + 0.08
                if abs(offset) > 0.01:
                    root = root.clone()
                    root[:, 1] -= offset
                    motion = create_motion_from_soma23_data(
                        local, root, kinematic_info, fps=args.output_fps
                    )
                    md = motion.to_dict() if hasattr(motion, "to_dict") else motion
                if not args.ignore_filter:
                    reason = quality_check(
                        md, max_velocity=args.max_velocity,
                        min_height=args.min_height,
                    )
                    if reason:
                        failed.append((dst.name, reason))
                        continue
                torch.save(md, dst)
                converted += 1
                print(f"  ok: {dst.name}")
            except Exception as exc:  # noqa: BLE001
                failed.append((dst.name, str(exc)))

    print(f"\nconverted: {converted} | failed: {len(failed)}")
    for name, reason in failed[:25]:
        print(f"  FAIL {name}: {reason[:90]}")

    # Skip interpreter teardown: destroying the (kept-alive) ufbx T-pose scene
    # segfaults this binding. All work is flushed; exit hard with a real code.
    import os
    sys.stdout.flush()
    os._exit(0 if converted > 0 else 1)


if __name__ == "__main__":
    main()
