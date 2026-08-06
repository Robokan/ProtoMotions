# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Retarget a motion corpus onto a geometrically scaled robot.

Scaling the skeleton is not enough: a bigger animal CANNOT perform the
same motion at the same speed. Gravity does not scale, so a body 1.71x
longer takes sqrt(1.71) = 1.31x as long to fall its own height, and a
stride that was balanced at 40 kg is a faceplant at 200 kg.

The physically correct mapping is Froude (dynamic) similarity: preserve
v^2 / (g * L). With lengths scaled by s and g fixed, that forces

    length      s
    TIME        sqrt(s)      <-- the part that is easy to forget
    velocity    s / sqrt(s) = sqrt(s)
    ang. vel.   1 / sqrt(s)  (angles are dimensionless, time is not)
    joint angle 1            unchanged
    rotation    1            unchanged

Retiming is done by lowering the clip's fps rather than resampling: the
library stores motion_dt as a float (1/fps), so the clip simply plays
slower with no interpolation error at all.

Physically this says a Utahraptor does not sprint like a Velociraptor --
it covers ground faster in absolute terms (sqrt(s) x the speed) but its
limbs cycle more slowly, which is exactly what large animals do.

    python data/scripts/scale_motions.py --in-dir data/motions/raptor_v5 \
        --out-dir data/motions/utahraptor --scale 1.709260
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import torch

# tensors that are LENGTHS
_POS = ("rigid_body_pos",)
# tensors that are LENGTH / TIME
_LINVEL = ("rigid_body_vel",)
# tensors that are ANGLE / TIME
_ANGVEL = ("rigid_body_ang_vel", "dof_vel")
# unchanged: rigid_body_rot (orientation), dof_pos (angle), contacts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scale", type=float, required=True,
                    help="LENGTH scale s (mass scales s^3)")
    args = ap.parse_args()

    s = args.scale
    t = math.sqrt(s)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"length x{s:.6f}   time x{t:.6f}   "
          f"linear vel x{s/t:.6f}   angular vel x{1/t:.6f}")

    n = 0
    first = True
    for p in sorted(glob.glob(f"{args.in_dir}/*.motion")):
        d = torch.load(p, weights_only=False, map_location="cpu")
        for k in _POS:
            if k in d:
                d[k] = (d[k] * s).contiguous()
        for k in _LINVEL:
            if k in d:
                d[k] = (d[k] * (s / t)).contiguous()
        for k in _ANGVEL:
            if k in d:
                d[k] = (d[k] / t).contiguous()
        old_fps = float(d.get("fps", 30))
        d["fps"] = old_fps / t          # motion_dt = 1/fps, stored as float
        torch.save(d, f"{args.out_dir}/{os.path.basename(p)}")
        if first:
            first = False
            print(f"  e.g. {os.path.basename(p)}: fps {old_fps:.2f} -> "
                  f"{d['fps']:.3f}  (dt {1/old_fps*1000:.1f} -> "
                  f"{1/d['fps']*1000:.1f} ms), duration x{t:.3f}")
        n += 1
    print(f"\nscaled {n} clips -> {args.out_dir}")


main()
