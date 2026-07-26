# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert GMR-retargeted samurai motions (.npz) to ProtoMotions .motion.

Much simpler than the atlas converter: the samurai robot was GENERATED
(rig2mjcf) with the exact conventions ProtoMotions uses — same MJCF in GMR
and physics, all-hinge joints in identical order, no root bake, no ball
ankles. qpos maps 1:1 (root_rot arrives xyzw, mujoco wants wxyz).

Usage (isaacsim venv; needs dm_control):
    python data/scripts/convert_gmr_npz_to_samurai.py \
        --input-dir ~/sparkpack/output/samurai_npz_v6 \
        --output-dir data/motions/samurai_v6
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MJCF = "protomotions/data/assets/mjcf/samurai.xml"


def main():
    ap = argparse.ArgumentParser()
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

    kin = extract_kinematic_info(MJCF)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.npz"))
    print(f"{len(files)} npz files")
    ok, failed = 0, []
    for f in files:
        dst = args.output_dir / f"{f.stem}.motion"
        if dst.exists() and not args.force_remake:
            continue
        try:
            z = np.load(f, allow_pickle=True)
            fps = float(z["fps"])
            root_pos_in = z["root_pos"].astype(np.float64)
            root_rot = z["root_rot"].astype(np.float64)[:, [3, 0, 1, 2]]
            dof = z["dof_pos"].astype(np.float64)
            qpos = np.concatenate([root_pos_in, root_rot, dof], axis=1)
            qpos = torch.tensor(qpos, dtype=torch.float32)
            root_pos, joint_rot_mats = extract_transforms_from_qpos(kin, qpos)
            motion = fk_from_transforms_with_velocities(
                kinematic_info=kin,
                root_pos=root_pos,
                joint_rot_mats=joint_rot_mats,
                fps=int(round(fps)),
                compute_velocities=True,
                velocity_max_horizon=3,
            )
            # Continuity-aware euler_xyz decomposition. The library decomposer
            # picks a fixed branch, which flips (coupled +-pi jumps across the
            # triplet) at the y ~ +-90deg singularity — lying poses in getups
            # live there. Per frame, evaluate both euler solutions and keep
            # the one closest to the previous frame, then unwrap.
            rm = joint_rot_mats.numpy()[:, 1:]  # [T, J, 3, 3], skip free root
            T_, J_ = rm.shape[0], rm.shape[1]
            dp = np.zeros((T_, J_ * 3))
            prev = np.zeros(J_ * 3)
            for t in range(T_):
                for j in range(J_):
                    R_ = rm[t, j]
                    sy = np.clip(R_[0, 2], -1.0, 1.0)
                    y1 = np.arcsin(sy)
                    x1 = np.arctan2(-R_[1, 2], R_[2, 2])
                    z1 = np.arctan2(-R_[0, 1], R_[0, 0])
                    y2 = np.pi - y1
                    x2 = x1 + np.pi
                    z2 = z1 + np.pi
                    best = None
                    for cand in ((x1, y1, z1), (x2, y2, z2)):
                        c = np.array(cand)
                        c = prev[j*3:j*3+3] + np.mod(
                            c - prev[j*3:j*3+3] + np.pi, 2*np.pi) - np.pi
                        d_ = np.abs(c - prev[j*3:j*3+3]).sum()
                        if best is None or d_ < best[0]:
                            best = (d_, c)
                    dp[t, j*3:j*3+3] = best[1]
                prev = dp[t]
            motion.dof_pos = torch.tensor(dp, dtype=torch.float32)
            dv = torch.zeros_like(motion.dof_pos)
            dv[1:] = (motion.dof_pos[1:] - motion.dof_pos[:-1]) * float(
                round(fps))
            motion.dof_vel = dv
            # ground the lowest body per clip (feet capsule radius ~0.06)
            motion.fix_height(height_offset=0.06)
            motion.rigid_body_contacts = (
                compute_contact_labels_from_pos_and_vel(
                    motion.rigid_body_pos, motion.rigid_body_vel))
            motion.local_rigid_body_rot = None
            torch.save(
                motion.to_dict() if hasattr(motion, "to_dict") else motion,
                dst)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed.append((f.name, str(exc)))
    print(f"converted: {ok} | failed: {len(failed)}")
    for n, r in failed[:10]:
        print(f"  FAIL {n}: {r[:120]}")


if __name__ == "__main__":
    main()
