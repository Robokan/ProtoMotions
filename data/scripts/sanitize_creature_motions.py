# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Clamp distal DOFs to realistic ranges and re-FK creature .motion clips.

Game-authored raptor attacks curl toes/arms/hands into poses that are
kinematically legal in the MJCF (±180° style ranges) but not anatomically
plausible. Those extremes poison AMP/ASE: the policy learns to chase
impossible references.

This pass:
  1. Identifies distal hinges (arms, hands, toe/foot digits).
  2. Clamps each to intersect(corpus p1–p99 ± margin, hard anatomical cap).
  3. Recomputes body poses/vels by FK from the clamped dof so stored
     rigid_body_* == sim-FK(dof).
  4. Recomputes contact labels from the new trajectories.

Core body DOFs (spine, neck, head, jaw, tail, hip, knee, ankle) are left
untouched.

Example:
    # Build distal limits from the current packed corpus, write raptor_v6
    python data/scripts/sanitize_creature_motions.py \
        --robot raptor \
        --in-dir data/motions/raptor_v5 \
        --out-dir data/motions/raptor_v6 \
        --limits-from data/raptor_pretrain_corpus_v9.pt

    # Same angle clamps on the Froude-scaled utah set (use utah MJCF for FK)
    python data/scripts/sanitize_creature_motions.py \
        --robot utahraptor \
        --in-dir data/motions/utahraptor \
        --out-dir data/motions/utahraptor \
        --backup-suffix _pre_sanitize \
        --limits-file data/motions/raptor_v6/_distal_limits.pt
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from protomotions.components.pose_lib import (  # noqa: E402
    extract_kinematic_info,
    extract_transforms_from_qpos,
    fk_from_transforms_with_velocities,
)
from data.scripts.contact_detection import (  # noqa: E402
    compute_contact_labels_from_pos_and_vel,
)

# Name substrings that mark distal hinges we sanitize. Shoulder / UpLeg /
# Leg / Foot (ankle) / axial skeleton are intentionally excluded.
_DISTAL_KEYS = (
    "ForeArm",
    "ToeBase",
    "FootIndex",
    "FootMiddle",
    "FootRing",
    "HandIndex",
    "HandMiddle",
    "HandRing",
)


def is_distal(name: str) -> bool:
    if any(k in name for k in _DISTAL_KEYS):
        return True
    stem = name.rsplit("_", 1)[0]
    # Upper arm + wrist only (Shoulder / axial / hindlimb stay core).
    return stem in ("LeftArm", "RightArm", "LeftHand", "RightHand")


def hard_cap_deg(name: str) -> tuple[float, float] | None:
    """Anatomical hard caps in degrees. None = percentile-only."""
    axis = name.rsplit("_", 1)[-1]
    if any(k in name for k in ("FootIndex", "FootMiddle", "FootRing")):
        return {"x": (-70.0, 90.0), "y": (-30.0, 30.0), "z": (-30.0, 30.0)}[axis]
    if "ToeBase" in name:
        return {"x": (-50.0, 100.0), "y": (-45.0, 45.0), "z": (-45.0, 45.0)}[axis]
    if any(k in name for k in ("HandIndex", "HandMiddle", "HandRing")):
        return {"x": (-90.0, 90.0), "y": (-75.0, 75.0), "z": (-75.0, 75.0)}[axis]
    if name.startswith("LeftHand_") or name.startswith("RightHand_"):
        return {"x": (-90.0, 90.0), "y": (-70.0, 70.0), "z": (-70.0, 70.0)}[axis]
    if "ForeArm" in name:
        return {"x": (-70.0, 100.0), "y": (-50.0, 50.0), "z": (-50.0, 50.0)}[axis]
    stem = name.rsplit("_", 1)[0]
    if stem in ("LeftArm", "RightArm"):
        return {"x": (-90.0, 90.0), "y": (-70.0, 70.0), "z": (-80.0, 80.0)}[axis]
    return None


def build_limits(
    dof_names: list[str],
    dps_deg: np.ndarray,
    lo_pct: float,
    hi_pct: float,
    margin_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lo_rad, hi_rad, distal_mask) for every dof."""
    n = len(dof_names)
    assert dps_deg.shape[1] == n
    p_lo = np.percentile(dps_deg, lo_pct, axis=0)
    p_hi = np.percentile(dps_deg, hi_pct, axis=0)
    lo = np.full(n, -math.pi, dtype=np.float64)
    hi = np.full(n, math.pi, dtype=np.float64)
    distal = np.zeros(n, dtype=bool)
    for i, name in enumerate(dof_names):
        if not is_distal(name):
            continue
        distal[i] = True
        a = p_lo[i] - margin_deg
        b = p_hi[i] + margin_deg
        hc = hard_cap_deg(name)
        if hc is not None:
            a = max(a, hc[0])
            b = min(b, hc[1])
            if a >= b:
                a, b = hc
        lo[i] = math.radians(a)
        hi[i] = math.radians(b)
    return lo, hi, distal


def sanitize_clip(
    d: dict,
    ki,
    lo: np.ndarray,
    hi: np.ndarray,
    distal: np.ndarray,
) -> tuple[dict, int, int]:
    """Clamp distal dof_pos, re-FK, recompute contacts. Returns (dict, n_clamped, n_distal_vals)."""
    dof = d["dof_pos"].clone()
    assert dof.shape[1] == len(lo)
    before = dof[:, distal].clone()
    lo_t = torch.as_tensor(lo, dtype=dof.dtype)
    hi_t = torch.as_tensor(hi, dtype=dof.dtype)
    # Only clamp distal columns; core stays identical.
    dof_clamped = dof.clone()
    dof_clamped[:, distal] = torch.maximum(
        torch.minimum(dof[:, distal], hi_t[distal]), lo_t[distal]
    )
    n_clamped = int((dof_clamped[:, distal] != before).sum().item())
    n_vals = int(before.numel())

    # Build MuJoCo-layout qpos: root pos, root quat WXYZ, hinges.
    root_pos = d["rigid_body_pos"][:, 0, :].contiguous()
    root_xyzw = d["rigid_body_rot"][:, 0, :].contiguous()
    root_wxyz = root_xyzw[:, [3, 0, 1, 2]]
    T = dof.shape[0]
    qpos = torch.zeros(T, 7 + dof.shape[1], dtype=dof.dtype)
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = root_wxyz
    qpos[:, 7:] = dof_clamped

    _, joint_mats = extract_transforms_from_qpos(ki, qpos)
    fps = float(d.get("fps", 30))
    fps_int = max(1, int(round(fps)))
    motion = fk_from_transforms_with_velocities(
        kinematic_info=ki,
        root_pos=qpos[:, 0:3],
        joint_rot_mats=joint_mats,
        fps=fps_int,
        compute_velocities=True,
        velocity_max_horizon=3,
    )

    # Preserve original fps (utah stores a non-integer Froude-scaled rate).
    out = dict(d)
    out["rigid_body_pos"] = motion.rigid_body_pos.contiguous()
    out["rigid_body_rot"] = motion.rigid_body_rot.contiguous()
    out["rigid_body_vel"] = motion.rigid_body_vel.contiguous()
    out["rigid_body_ang_vel"] = motion.rigid_body_ang_vel.contiguous()
    out["dof_pos"] = dof_clamped.contiguous()
    dv = torch.zeros_like(dof_clamped)
    if T > 1:
        dv[1:] = (dof_clamped[1:] - dof_clamped[:-1]) * fps
    out["dof_vel"] = dv.contiguous()
    out["fps"] = d.get("fps", fps_int)
    out["rigid_body_contacts"] = compute_contact_labels_from_pos_and_vel(
        out["rigid_body_pos"], out["rigid_body_vel"]
    )
    return out, n_clamped, n_vals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot", required=True, help="raptor | utahraptor | ...")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--limits-from",
        default=None,
        help="Packed MotionLib .pt whose dps define percentile limits",
    )
    ap.add_argument(
        "--limits-file",
        default=None,
        help="Reuse a previously saved _distal_limits.pt (skips --limits-from)",
    )
    ap.add_argument("--lo-pct", type=float, default=1.0)
    ap.add_argument("--hi-pct", type=float, default=99.0)
    ap.add_argument("--margin-deg", type=float, default=5.0)
    ap.add_argument(
        "--backup-suffix",
        default=None,
        help="If out-dir == in-dir, copy originals to <in-dir><suffix>/ first",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mjcf = REPO / f"protomotions/data/assets/mjcf/{args.robot}.xml"
    ki = extract_kinematic_info(str(mjcf))
    dof_names = list(ki.dof_names)
    print(f"robot={args.robot}  ndof={len(dof_names)}  bodies={ki.num_bodies}")

    if args.limits_file:
        lim = torch.load(args.limits_file, map_location="cpu", weights_only=False)
        lo = np.asarray(lim["lo"], dtype=np.float64)
        hi = np.asarray(lim["hi"], dtype=np.float64)
        distal = np.asarray(lim["distal"], dtype=bool)
        assert list(lim["dof_names"]) == dof_names, (
            "limits-file dof_names do not match this robot MJCF"
        )
        print(f"loaded limits from {args.limits_file}")
    else:
        if not args.limits_from:
            ap.error("need --limits-from or --limits-file")
        corpus = torch.load(args.limits_from, map_location="cpu", weights_only=False)
        dps = corpus["dps"]
        dps = dps.numpy() if hasattr(dps, "numpy") else np.asarray(dps)
        assert dps.shape[1] == len(dof_names), (
            f"corpus ndof {dps.shape[1]} != mjcf {len(dof_names)}"
        )
        lo, hi, distal = build_limits(
            dof_names, np.degrees(dps), args.lo_pct, args.hi_pct, args.margin_deg
        )
        print(
            f"limits from {args.limits_from}: "
            f"p{args.lo_pct:g}–p{args.hi_pct:g} ±{args.margin_deg:g}° "
            f"+ hard caps; distal dofs={int(distal.sum())}"
        )

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if out_dir == in_dir and args.backup_suffix:
        backup = Path(str(in_dir) + args.backup_suffix)
        if not args.dry_run:
            backup.mkdir(parents=True, exist_ok=True)
        print(f"backing up originals -> {backup}")
    else:
        backup = None

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Persist limits next to the output for utah reuse / audit.
        torch.save(
            {
                "dof_names": dof_names,
                "lo": lo,
                "hi": hi,
                "distal": distal,
                "lo_pct": args.lo_pct,
                "hi_pct": args.hi_pct,
                "margin_deg": args.margin_deg,
                "limits_from": args.limits_from or args.limits_file,
            },
            out_dir / "_distal_limits.pt",
        )

    paths = sorted(glob.glob(str(in_dir / "*.motion")))
    total_clamped = 0
    total_vals = 0
    worst = []
    for path in paths:
        name = os.path.basename(path)
        d = torch.load(path, map_location="cpu", weights_only=False)
        out, n_c, n_v = sanitize_clip(d, ki, lo, hi, distal)
        total_clamped += n_c
        total_vals += n_v
        frac = 100.0 * n_c / max(n_v, 1)
        worst.append((frac, n_c, name))
        if args.dry_run:
            continue
        if backup is not None:
            shutil.copy(path, backup / name)
        torch.save(out, out_dir / name)

    worst.sort(reverse=True)
    print(f"\nclips={len(paths)}  distal values touched="
          f"{total_clamped}/{total_vals} ({100*total_clamped/max(total_vals,1):.2f}%)")
    print("most-edited clips:")
    for frac, n_c, name in worst[:12]:
        print(f"  {frac:5.1f}%  ({n_c:5d} vals)  {name}")
    if args.dry_run:
        print("\n[dry-run] no files written")
    else:
        print(f"\nwrote {len(paths)} clips -> {out_dir}")


if __name__ == "__main__":
    main()
