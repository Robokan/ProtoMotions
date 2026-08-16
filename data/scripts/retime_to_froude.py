# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Retime locomotion clips to a target Froude number.

Fr = v^2 / (g * L), with L the mean root height. Animals walk at Fr
0.15-0.25; the utahraptor's whole walk library sits at Fr 0.038-0.157 --
below or at the bottom edge of that band, consistently, across every clip
and both mirrors (verified with the corpus's true length_starts offsets,
NOT reconstructed ones: accumulating rounded frame counts drifts and makes
later "clips" span junctions, which reads a between-clip teleport as
motion and invents speed differences between mirror pairs).

Retiming multiplies fps and every velocity field by k = sqrt(Fr_target /
Fr_actual); POSES ARE UNTOUCHED, so stride length, foot placement and
style are preserved exactly -- only the clock changes. Note this is NOT
Froude body-scaling (data/scripts/scale_motions.py, which maps a motion
onto a differently sized robot); the body is fixed here and only the
animated tempo moves.

Per-clip k is computed from that clip's own measured Fr, so a clip already
in band is left alone (k within --deadband of 1.0).

    python data/scripts/retime_to_froude.py --in-dir data/motions/utahraptor \\
        --out-dir data/motions/utahraptor_froude --pattern 'WalkFwd*' \\
        --target-fr 0.18 --dry-run
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil

import numpy as np
import torch

VEL_KEYS = ("dof_vel", "rigid_body_vel", "rigid_body_ang_vel")


def measure(mo):
    rp = mo["rigid_body_pos"].numpy()[:, 0]
    fps = float(mo.get("fps", 30))
    h = float(rp[:, 2].mean())
    if rp.shape[0] < 3:
        return None
    v = float(np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1).mean() * fps)
    return v, h, v * v / (9.81 * max(h, 1e-3)), fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--exclude", default="*Stop*,*Idle*,Run*",
                    help="comma-separated globs never retimed (non-locomotion "
                         "and clips already fast)")
    ap.add_argument("--target-fr", type=float, default=0.18)
    ap.add_argument("--deadband", type=float, default=0.05,
                    help="leave a clip alone if |k-1| is under this")
    ap.add_argument("--max-k", type=float, default=2.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    excl = [x for x in args.exclude.split(",") if x]
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)

    n_ret = n_kept = n_skip = 0
    for f in sorted(os.listdir(args.in_dir)):
        if not f.endswith(".motion"):
            continue
        src = f"{args.in_dir}/{f}"
        stem = f[:-7]
        selected = fnmatch.fnmatch(stem, args.pattern) and not any(
            fnmatch.fnmatch(stem, e) for e in excl)
        mo = torch.load(src, weights_only=False, map_location="cpu")
        got = measure(mo)
        if not selected or got is None:
            n_skip += 1
            if not args.dry_run:
                shutil.copy(src, f"{args.out_dir}/{f}")
            continue
        v, h, fr, fps = got
        k = float(np.sqrt(args.target_fr / max(fr, 1e-6)))
        k = min(k, args.max_k)
        if abs(k - 1.0) <= args.deadband:
            n_kept += 1
            print(f"  {stem[:34]:<34} Fr {fr:.3f} -> in band, unchanged")
            if not args.dry_run:
                shutil.copy(src, f"{args.out_dir}/{f}")
            continue
        mo["fps"] = fps * k
        for key in VEL_KEYS:
            if key in mo:
                mo[key] = mo[key] * k
        n_ret += 1
        print(f"  {stem[:34]:<34} Fr {fr:.3f} -> {fr*k*k:.3f}  "
              f"({v:.2f} -> {v*k:.2f} m/s, x{k:.2f})")
        if not args.dry_run:
            torch.save(mo, f"{args.out_dir}/{f}")

    print(f"\n{n_ret} retimed, {n_kept} already in band, {n_skip} passed "
          f"through untouched")
    if args.dry_run:
        print("dry run: nothing written")


if __name__ == "__main__":
    main()
