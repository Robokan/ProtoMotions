# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert GMR-retargeted T800 motions (.npz/.pkl) to ProtoMotions .motion files.

GMR (``retarget_headless.py --robot t800``) saves
``{fps, root_pos [T,3], root_rot [T,4] xyzw, dof_pos [T,25]}``. The Proto MJCF
matches the GMR model (identity root, 25 hinge DOFs) — no root-bake or ankle
decomposition. Foot body origins sit on the sole plane at rest, so only a
small ground clearance offset is applied.

Usage:
    python data/scripts/convert_gmr_pkl_to_t800.py \\
        --input-dir data/motions/gmr_t800_npz --output-dir data/motions/gmr_t800
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

T800_NDOF = 25
T800_MJCF = "protomotions/data/assets/mjcf/t800.xml"
# Sole ≈ FOOT origin at qpos0; small clearance so contact labels stay clean.
T800_HEIGHT_OFFSET = 0.002


def gmr_to_qpos(path: Path):
    if path.suffix == ".npz":
        d = dict(np.load(path, allow_pickle=False))
    else:
        with open(path, "rb") as f:
            d = pickle.load(f)
    root_pos = np.asarray(d["root_pos"], dtype=np.float64)
    rr = np.asarray(d["root_rot"], dtype=np.float64)  # xyzw
    root_wxyz = rr[:, [3, 0, 1, 2]]
    dof = np.asarray(d["dof_pos"], dtype=np.float64)
    assert dof.shape[1] == T800_NDOF, (
        f"expected t800 dof_pos width {T800_NDOF}, got {dof.shape}"
    )
    qpos = np.concatenate([root_pos, root_wxyz, dof], axis=-1)
    assert qpos.shape[1] == 7 + T800_NDOF, qpos.shape
    return torch.from_numpy(qpos).float(), float(d.get("fps", 30.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--force-remake", action="store_true")
    args = ap.parse_args()

    from protomotions.components.pose_lib import (
        extract_kinematic_info,
        extract_transforms_from_qpos,
        extract_qpos_from_transforms,
        fk_from_transforms_with_velocities,
    )
    from contact_detection import compute_contact_labels_from_pos_and_vel

    kinematic_info = extract_kinematic_info(T800_MJCF)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        list(args.input_dir.glob("*.npz")) + list(args.input_dir.glob("*.pkl"))
    )
    print(f"{len(files)} gmr files")
    ok, failed = 0, []
    for f in files:
        dst = args.output_dir / f"{f.stem}.motion"
        if dst.exists() and not args.force_remake:
            continue
        try:
            qpos, fps = gmr_to_qpos(f)
            root_pos, joint_rot_mats = extract_transforms_from_qpos(
                kinematic_info, qpos
            )
            motion = fk_from_transforms_with_velocities(
                kinematic_info=kinematic_info,
                root_pos=root_pos,
                joint_rot_mats=joint_rot_mats,
                fps=int(round(fps)),
                compute_velocities=True,
                velocity_max_horizon=3,
            )
            q2 = extract_qpos_from_transforms(
                kinematic_info,
                root_pos,
                joint_rot_mats,
                multi_dof_decomposition_method="euler_xyz",
            )
            # GMR / euler extraction can wrap hinges across ±π between frames,
            # which looks like a limb teleport in the visualizer (Atlas is
            # smoother on the same BVHs). Unwrap each DOF over time.
            dof = q2[:, 7:].numpy()
            for j in range(dof.shape[1]):
                dof[:, j] = np.unwrap(dof[:, j])
            motion.dof_pos = torch.from_numpy(dof).float()
            dv = torch.zeros_like(motion.dof_pos)
            dv[1:] = (motion.dof_pos[1:] - motion.dof_pos[:-1]) * float(round(fps))
            motion.dof_vel = dv
            motion.fix_height(height_offset=T800_HEIGHT_OFFSET)
            motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
                motion.rigid_body_pos, motion.rigid_body_vel
            )
            motion.local_rigid_body_rot = None
            torch.save(
                motion.to_dict() if hasattr(motion, "to_dict") else motion, dst
            )
            ok += 1
            if ok <= 5 or ok % 200 == 0:
                print(f"  ok: {dst.name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((f.name, str(exc)))
    print(f"\nconverted: {ok} | failed: {len(failed)}")
    for n, r in failed[:20]:
        print(f"  FAIL {n}: {r[:120]}")


if __name__ == "__main__":
    main()
