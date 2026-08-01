# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert UE Manny-skeleton poselib .npy clips to SOMA23 ``.motion``.

Front-end to convert_manny_fbx_to_soma.py for npy sources (e.g. the
IsaacLabASE Template's manny_unreal_4 combat set): FK the poselib local
quats to world rotations, convert z-up -> y-up, then reuse the proven
world-rotation-copy retarget + motion builder. Downstream, the standard
chain applies (package lib -> convert_soma23_motion_to_bvh ->
retarget_headless) — the direct npy->GMR path (retarget_ue_npy) produces
CROUCHED robots on rotation-weighted rigs (frame-convention mismatch; see
GMR_Grab retarget_ue_fbx.py deprecation note).

Usage:
    python data/scripts/convert_manny_npy_to_soma.py \\
        --input-dir <staged npys> --output-dir <motions out> \\
        --tpose-file ~/eric/unreal_fbx_animations/drunken/MM_T_Pose.FBX
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_manny_fbx_to_soma import (  # noqa: E402
    MANNY_MAP,
    _ZUP_TO_YUP,
    load_tpose_bind,
    retarget_to_soma,
    quality_check,
)


def _unwrap(v):
    if hasattr(v, "keys") and "arr" in v:
        return np.asarray(v["arr"])
    return np.asarray(v)


def _quat_to_mat(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = (q * q).sum(-1)
    s = 2.0 / np.clip(n, 1e-12, None)
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - s * (y * y + z * z); m[..., 0, 1] = s * (x * y - w * z); m[..., 0, 2] = s * (x * z + w * y)
    m[..., 1, 0] = s * (x * y + w * z); m[..., 1, 1] = 1 - s * (x * x + z * z); m[..., 1, 2] = s * (y * z - w * x)
    m[..., 2, 0] = s * (x * z - w * y); m[..., 2, 1] = s * (y * z + w * x); m[..., 2, 2] = 1 - s * (x * x + y * y)
    return m


def load_npy_frames(npy_path: Path):
    """poselib npy -> (world_rots {manny_bone: [T,3,3]} y-up, root_pos [T,3]
    y-up meters, fps). Mirrors load_fbx_frames' output conventions."""
    d = np.load(npy_path, allow_pickle=True).item()
    st = d["skeleton_tree"]
    names = list(st["node_names"])
    parents = np.asarray(_unwrap(st["parent_indices"]), dtype=np.int64)
    offsets = _unwrap(st["local_translation"]).astype(np.float64)  # cm
    quats = _unwrap(d["rotation"]).astype(np.float64)              # local xyzw
    root_t = _unwrap(d["root_translation"]).astype(np.float64)     # cm
    fps = int(d.get("fps", 30))

    local = _quat_to_mat(quats)
    T, J = local.shape[:2]
    wrot = np.empty_like(local)
    wpos = np.empty((T, J, 3))
    for j in range(J):
        p = parents[j]
        if p < 0:
            wrot[:, j] = local[:, j]
            wpos[:, j] = root_t
        else:
            wrot[:, j] = wrot[:, p] @ local[:, j]
            wpos[:, j] = wpos[:, p] + np.einsum("tab,b->ta", wrot[:, p], offsets[j])

    wanted = set(MANNY_MAP.values())
    missing = wanted - set(names)
    if missing:
        raise ValueError(f"npy missing Manny bones: {sorted(missing)}")
    idx = {n: i for i, n in enumerate(names)}
    world_rots = {n: _ZUP_TO_YUP[None] @ wrot[:, idx[n]] for n in wanted}
    root_pos = (wpos[:, idx["pelvis"]] * 0.01) @ _ZUP_TO_YUP.T
    return world_rots, root_pos, fps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tpose-file", type=Path, required=True)
    p.add_argument("--output-fps", type=int, default=30)
    p.add_argument("--force-remake", action="store_true")
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--max-velocity", type=float, default=15.0)
    p.add_argument("--min-height", type=float, default=-0.05)
    p.add_argument("--ignore-filter", action="store_true")
    args = p.parse_args()

    from protomotions.components.pose_lib import extract_kinematic_info
    from data.scripts.convert_soma23_to_proto import create_motion_from_soma23_data

    kinematic_info = extract_kinematic_info(
        "protomotions/data/assets/mjcf/soma23_humanoid.xml"
    )
    bind = load_tpose_bind(args.tpose_file)
    # The kept-alive ufbx scene from load_tpose_bind segfaults when the GC
    # cycle collector traverses it mid-run ("Garbage-collecting" crash in
    # otherwise-pure-torch code). One scene, bounded run: disable GC.
    import gc
    gc.disable()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.npy"))
    print(f"{len(files)} npy files")

    converted, failed = 0, []
    for f in files:
        try:
            world_rots, root_pos, fps = load_npy_frames(f)
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
                    local, root, kinematic_info, fps=fps
                )
                md = motion.to_dict() if hasattr(motion, "to_dict") else motion
                zmin = md["rigid_body_pos"][..., 2].min(dim=1).values
                offset = float(zmin.median())
                worst = float(zmin.min()) - offset
                if worst < -0.08:
                    offset += worst + 0.08
                if abs(offset) > 0.01:
                    root = root.clone()
                    root[:, 1] -= offset
                    motion = create_motion_from_soma23_data(
                        local, root, kinematic_info, fps=fps
                    )
                    md = motion.to_dict() if hasattr(motion, "to_dict") else motion
                if not args.ignore_filter:
                    reason = quality_check(md, args.max_velocity, args.min_height)
                    if reason:
                        failed.append((dst.name, reason))
                        continue
                torch.save(md, dst)
                converted += 1
                print(f"  ok: {dst.name}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed.append((dst.name, str(exc)))

    print(f"\nconverted: {converted} | failed: {len(failed)}")
    for name, reason in failed[:20]:
        print(f"  FAIL {name}: {reason[:90]}")
    import os
    sys.stdout.flush()
    os._exit(0)  # ufbx scene teardown segfaults; exit hard


if __name__ == "__main__":
    main()
