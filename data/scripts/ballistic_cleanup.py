# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Make airborne phases of hand-animated clips physically valid.

A body in flight has nothing to push against, so its centre of mass MUST
follow a ballistic arc (constant -g). Animated creature clips routinely
violate this -- the raptor's jump attacks depart from gravity by up to
17 m/s^2 -- and a policy cannot reproduce that no matter how well it
learns, so those clips feed the AMP/ASE discriminator a free win.

For every airborne segment this rewrites the ROOT TRAJECTORY as the
ballistic arc that connects the segment's existing takeoff and landing
points over the same duration:

    v_xy = (p1_xy - p0_xy) / T
    v_z  = (p1_z - p0_z + 0.5 * g * T^2) / T
    p(t) = p0 + v * t + (0, 0, -0.5 * g * t^2)

Takeoff and landing are therefore preserved exactly -- the character
still leaves and lands where the animator intended, and the clip stays
continuous with the grounded frames on either side. Only the path
between them changes, and JOINT ANGLES ARE UNTOUCHED: the pose is the
animator's, it is the root path that was lying. Every body is shifted by
the same per-frame delta, then velocities are recomputed.

    python data/scripts/ballistic_cleanup.py --robot raptor \
        --in-dir data/motions/raptor_v3 --out-dir data/motions/raptor_v5
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

G = 9.81


def airborne_segments(low: np.ndarray, clearance: float, min_len: int,
                      max_len: int = 10 ** 9):
    """Contiguous frame ranges where no part of the body is near the ground.

    `clearance` is an ABSOLUTE height. Using each clip's own minimum instead
    is unsafe: a clip that crouches once (Dig bottoms out at 0.064 m while
    standing at 0.142 m) then reads as airborne throughout, and gets flung
    along a ten-metre parabola. Measured references: walking and idling keep
    the lowest body at 0.010-0.019 m, a death lies at 0.027-0.110 m, and a
    real jump attack reaches 0.725 m.
    """
    air = low > clearance
    segs, start = [], None
    for i, a in enumerate(air):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if min_len <= i - start <= max_len:
                segs.append((start, i - 1))
            start = None
    if start is not None and min_len <= len(air) - start <= max_len:
        segs.append((start, len(air) - 1))
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="raptor")
    ap.add_argument("--in-dir", default="data/motions/raptor_v3")
    ap.add_argument("--out-dir", default="data/motions/raptor_v5")
    ap.add_argument("--clearance", type=float, default=0.15,
                    help="ABSOLUTE height of the lowest body above which the "
                         "clip counts as airborne (walking sits at ~0.02 m)")
    ap.add_argument("--max-flight-s", type=float, default=1.2,
                    help="segments longer than this are detection failures, "
                         "not flight, and are left alone")
    ap.add_argument("--min-flight-frames", type=int, default=3)
    ap.add_argument("--max-airborne-frac", type=float, default=0.6,
                    help="if more than this fraction of a clip reads as "
                         "flight, treat it as a detection failure and leave "
                         "the clip alone")
    ap.add_argument("--report", type=int, default=10)
    args = ap.parse_args()

    from protomotions.robot_configs.factory import robot_config

    import mujoco

    rc = robot_config(args.robot)
    bn = list(rc.kinematic_info.body_names)
    mj = mujoco.MjModel.from_xml_path(
        f"protomotions/data/assets/mjcf/{args.robot}.xml")
    sim_names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i)
                 for i in range(mj.nbody)]
    mass = np.array([mj.body_mass[sim_names.index(b)] if b in sim_names else 0.0
                     for b in bn])
    mass_total = mass.sum()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(f"{args.in_dir}/*.motion")):
        name = os.path.basename(path)
        d = torch.load(path, weights_only=False, map_location="cpu")
        p = d["rigid_body_pos"].numpy().astype(np.float64)
        T, B = p.shape[0], p.shape[1]
        dt = 1.0 / float(d.get("fps", 30))
        if T < 4:
            shutil.copy(path, f"{args.out_dir}/{name}")
            continue

        # Airborne means NO part of the body is near the ground -- checking
        # only the feet would call a death or knockdown "flight" (the raptor
        # is lying down with its legs raised) and fling the corpse along a
        # parabola. The lowest body overall is the honest test.
        low = np.min(p[:, :, 2], axis=1)

        # A clip that NEVER touches down has no takeoff or landing to anchor a
        # parabola to, so "airborne" covers the whole thing and the correction
        # flings it away. Observed: the 20cm leg-offset additive layers sit at
        # 0.27 m throughout and were lifted to 1.20 m. Additive layers are not
        # standalone motions -- leave them exactly as they are.
        if low.min() > args.clearance:
            shutil.copy(path, f"{args.out_dir}/{name}")
            rows.append((name, 0, 0.0))
            continue

        segs = airborne_segments(
            low, args.clearance, args.min_flight_frames,
            int(round(args.max_flight_s / dt)))

        # Likewise, if most of the clip reads as flight the detector is wrong,
        # not the animator: a knockdown tumbles with its lowest body above the
        # threshold for 68% of its length and got shifted up 5 cm.
        if sum(b - a + 1 for a, b in segs) > args.max_airborne_frac * len(low):
            shutil.copy(path, f"{args.out_dir}/{name}")
            rows.append((name, 0, 0.0))
            continue

        if not segs:
            shutil.copy(path, f"{args.out_dir}/{name}")
            rows.append((name, 0, 0.0))
            continue

        # Constrain the CENTRE OF MASS, not the root: during a kick the
        # limbs swing, so a ballistic root still leaves the COM accelerating
        # off gravity (measured: root-only correction only took the jump
        # attacks from ~10-15 to ~5-10 m/s^2 of error).
        com = (p * mass[None, :, None]).sum(1) / mass_total
        delta = np.zeros_like(com)
        for (a, b) in segs:
            n = b - a
            if n < 1:
                continue
            flight_T = n * dt
            p0, p1 = com[a], com[b]
            v = np.zeros(3)
            v[:2] = (p1[:2] - p0[:2]) / flight_T
            v[2] = (p1[2] - p0[2] + 0.5 * G * flight_T ** 2) / flight_T
            for k in range(n + 1):
                t = k * dt
                want = p0 + v * t
                want[2] -= 0.5 * G * t * t
                delta[a + k] = want - com[a + k]

        moved = float(np.abs(delta).max())
        # shift every body by the same per-frame correction (joint angles,
        # and therefore body-relative geometry, are unchanged)
        p_new = p + delta[:, None, :]
        d["rigid_body_pos"] = torch.from_numpy(p_new).float()
        # velocities follow from the corrected positions
        vel = np.gradient(p_new, dt, axis=0)
        d["rigid_body_vel"] = torch.from_numpy(vel).float()
        torch.save(d, f"{args.out_dir}/{name}")
        rows.append((name, len(segs), moved))

    fixed = [r for r in rows if r[1] > 0]
    fixed.sort(key=lambda r: -r[2])
    print(f"scanned {len(rows)} clips; rewrote flight in {len(fixed)}")
    print(f"{'clip':<42}{'flights':>8}{'max shift':>12}")
    for name, nseg, moved in fixed[:args.report]:
        print(f"  {name:<40}{nseg:8d}{moved:11.3f} m")
    print(f"\nwritten to {args.out_dir}")


main()
