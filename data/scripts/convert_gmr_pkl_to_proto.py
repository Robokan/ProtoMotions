# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert GMR-retargeted Atlas motions (.pkl) to ProtoMotions .motion files.

GMR (scripts/soma_bvh_to_robot.py / retarget_headless.py, robot=atlas_fists)
saves {fps, root_pos [T,3], root_rot [T,4] xyzw, dof_pos [T,34]} in the GMR
rig's coordinate conventions. Two model deltas are handled here:

1. Root frame: the physics MJCF baked the rig's rotated Hip frame to identity
   (see retune_atlas_mjcf.py) — retargeted root quats are composed with
   ATLAS_ROOT_BAKE_QUAT^-1.
2. Ankles: the rig has passive BALL ankle joints (a wxyz quat inside dof_pos);
   the physics model has yaw/roll/pitch hinge triplets (declared z,y,x). The
   ball quat is decomposed as ZYX Tait-Bryan angles onto those hinges.

Usage:
    python data/scripts/convert_gmr_pkl_to_proto.py \\
        --input-dir data/motions/gmr_atlas_pkl --output-dir data/motions/gmr_atlas
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Root-frame bake constant from retune_atlas_mjcf.py (wxyz).
ATLAS_ROOT_BAKE_QUAT = np.array([0.67082, 0.74162, 0.0, 0.0])

# GMR atlas_fists dof_pos layout (qpos[7:], 34 dims):
#   0..21  hinges: Twist, Backbone, Neck_2, Head, ArmL x7, ArmR x7, LegL x4
#   22..25 Foot_L ball quat (wxyz)
#   26..29 LegR hinges x4
#   30..33 Foot_R ball quat (wxyz)
# Physics-model dof order (30): same hinges with each ball replaced by a
# CHAINED 2-DOF ankle [Pitch, Roll] (yaw dropped — real Atlas has none).
N_HINGE_BLOCK1 = 22  # Twist .. Leg_8_L
N_LEGR = 4


def _qmul(a, b):
    w1, x1, y1, z1 = a.T if a.ndim == 2 else a
    w2, x2, y2, z2 = b.T if b.ndim == 2 else b
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _quat_to_mat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
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


def _ball_to_xyz_hinges(quat_wxyz):
    """Ball quat [T,4] -> (pitch_x, roll_y, yaw_z) for hinges declared x,y,z.

    Matches the framework's euler_xyz convention: R = Rx(p) @ Ry(r) @ Rz(y).
    """
    m = _quat_to_mat(quat_wxyz)
    roll = np.arcsin(np.clip(m[..., 0, 2], -1.0, 1.0))
    yaw = np.arctan2(-m[..., 0, 1], m[..., 0, 0])
    pitch = np.arctan2(-m[..., 1, 2], m[..., 2, 2])
    return pitch, roll, yaw


def gmr_pkl_to_qpos(pkl_path: Path):
    """GMR pkl/npz -> (qpos [T, 39] for the physics model, fps)."""
    if pkl_path.suffix == ".npz":
        d = dict(np.load(pkl_path, allow_pickle=False))
    else:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
    root_pos = np.asarray(d["root_pos"], dtype=np.float64)
    rr = np.asarray(d["root_rot"], dtype=np.float64)  # xyzw
    root_wxyz = rr[:, [3, 0, 1, 2]]
    dof = np.asarray(d["dof_pos"], dtype=np.float64)
    T = root_pos.shape[0]
    assert dof.shape[1] in (30, 34), f"expected atlas_fists dof_pos 30 (hinge) or 34 (ball), got {dof.shape}"

    # Root: compose with the bake conjugate (identity root frame model).
    qb = ATLAS_ROOT_BAKE_QUAT / np.linalg.norm(ATLAS_ROOT_BAKE_QUAT)
    qb_conj = qb * np.array([1.0, -1.0, -1.0, -1.0])
    root_new = _qmul(root_wxyz, np.tile(qb_conj, (T, 1)))

    if dof.shape[1] == 30:
        # New GMR layout (hinge ankles): direct passthrough.
        qpos = np.concatenate([root_pos, root_new, dof], axis=-1)
    else:
        # Legacy GMR layout (ball ankles, 34 dof): decompose to pitch/roll.
        assert dof.shape[1] == 34, dof.shape
        hinges1 = dof[:, :N_HINGE_BLOCK1]
        ball_l = dof[:, 22:26]
        legr = dof[:, 26:30]
        ball_r = dof[:, 30:34]
        pl, rl, yl = _ball_to_xyz_hinges(ball_l)
        pr, rr_, yr = _ball_to_xyz_hinges(ball_r)
        qpos = np.concatenate([
            root_pos, root_new, hinges1,
            np.stack([pl, rl], axis=-1),   # Pitch(x), Roll(y) — yaw dropped
            legr,
            np.stack([pr, rr_], axis=-1),
        ], axis=-1)
    assert qpos.shape[1] == 37, qpos.shape
    return torch.from_numpy(qpos).float(), float(d.get("fps", 30.0))


def _rate_limit(x, dmax):
    """Per-DOF rate limiter: each step's delta (vs the limited trajectory)
    is clamped to +-dmax. Rejoins the source within a few frames."""
    out = x.clone()
    for t in range(1, x.shape[0]):
        out[t] = out[t - 1] + (x[t] - out[t - 1]).clamp(-dmax, dmax)
    return out


def clamp_dof_velocities(dof, limits, fps, scale):
    """Clamp |dof_vel| to scale*limits. Mean of a forward and a backward
    rate-limit pass — both feasible, so their mean is too, and the
    symmetric blend avoids the phase lag of a one-directional limiter."""
    dmax = limits * scale / fps
    fwd = _rate_limit(dof, dmax)
    bwd = torch.flip(_rate_limit(torch.flip(dof, [0]), dmax), [0])
    return 0.5 * (fwd + bwd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--force-remake", action="store_true")
    ap.add_argument("--clamp-dof-vel", type=float, default=0.0,
                    help="clamp dof velocities to this fraction of the atlas "
                         "actuator velocity_limit (0 = off)")
    args = ap.parse_args()

    from protomotions.components.pose_lib import (
        extract_kinematic_info,
        extract_transforms_from_qpos,
        extract_qpos_from_transforms,
        fk_from_transforms_with_velocities,
    )

    kinematic_info = extract_kinematic_info("protomotions/data/assets/mjcf/atlas.xml")

    vel_limits = None
    if args.clamp_dof_vel > 0.0:
        from protomotions.robot_configs.atlas import AtlasRobotConfig
        info = AtlasRobotConfig().control.control_info
        vel_limits = torch.tensor(
            [info[n].velocity_limit for n in kinematic_info.dof_names],
            dtype=torch.float32,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(list(args.input_dir.glob("*.pkl")) + list(args.input_dir.glob("*.npz")))
    print(f"{len(files)} pkl files")
    ok, failed = 0, []
    for f in files:
        dst = args.output_dir / f"{f.stem}.motion"
        if dst.exists() and not args.force_remake:
            continue
        try:
            qpos, fps = gmr_pkl_to_qpos(f)
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
                kinematic_info, root_pos, joint_rot_mats,
                multi_dof_decomposition_method="euler_xyz",
            )
            if vel_limits is not None:
                clamped = clamp_dof_velocities(
                    q2[:, 7:], vel_limits, float(round(fps)), args.clamp_dof_vel
                )
                if not torch.allclose(clamped, q2[:, 7:]):
                    q2 = torch.cat([q2[:, :7], clamped], dim=-1)
                    root_pos, joint_rot_mats = extract_transforms_from_qpos(
                        kinematic_info, q2
                    )
                    motion = fk_from_transforms_with_velocities(
                        kinematic_info=kinematic_info,
                        root_pos=root_pos,
                        joint_rot_mats=joint_rot_mats,
                        fps=int(round(fps)),
                        compute_velocities=True,
                        velocity_max_horizon=3,
                    )
            motion.dof_pos = q2[:, 7:]
            dv = torch.zeros_like(motion.dof_pos)
            dv[1:] = (motion.dof_pos[1:] - motion.dof_pos[:-1]) * float(round(fps))
            motion.dof_vel = dv
            # Atlas: the Foot BODY origin sits 0.076m above the sole plane
            # (measured at rest, mujoco qpos0). Generic offsets sank the sole
            # ~2-3cm under ground.
            # Sole-plane offsets measured on the 1.68 m asset; the asset
            # was rescaled x0.9048 to the real 1.52 m Atlas (2026-08-01).
            motion.fix_height_per_frame(height_offset=0.0507)
            motion.fix_height(height_offset=0.0688)
            from contact_detection import compute_contact_labels_from_pos_and_vel
            motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
                motion.rigid_body_pos, motion.rigid_body_vel
            )
            motion.local_rigid_body_rot = None
            torch.save(
                motion.to_dict() if hasattr(motion, "to_dict") else motion, dst
            )
            ok += 1
            print(f"  ok: {dst.name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((f.name, str(exc)))
    print(f"\nconverted: {ok} | failed: {len(failed)}")
    for n, r in failed[:10]:
        print(f"  FAIL {n}: {r[:100]}")


if __name__ == "__main__":
    main()
