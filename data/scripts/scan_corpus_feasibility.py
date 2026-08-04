# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Flag physically infeasible clips in a hand-animated corpus.

These creature corpora are ANIMATED, not captured, so they can contain
motion no policy can reproduce: feet sliding while planted, bodies
floating (upward acceleration with no ground contact), or teleports. A
clip like that guarantees the AMP/ASE discriminator wins regardless of
how good the policy is, so it is worth finding before a long pretrain.

    python data/scripts/scan_corpus_feasibility.py --robot raptor \
        --motion-dir data/motions/raptor_v3
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="raptor")
    ap.add_argument("--motion-dir", default="data/motions/raptor_v3")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    import mujoco
    from protomotions.robot_configs.factory import robot_config

    rc = robot_config(args.robot)
    bn = list(rc.kinematic_info.body_names)
    m = mujoco.MjModel.from_xml_path(
        f"protomotions/data/assets/mjcf/{args.robot}.xml")
    sim_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                 for i in range(m.nbody)]
    mass = np.array([m.body_mass[sim_names.index(b)] if b in sim_names else 0.0
                     for b in bn])
    total = mass.sum()
    feet = [i for i, b in enumerate(bn)
            if b.endswith("ToeBase") or b.endswith("LegAnkle")]
    if not feet:
        feet = [i for i, b in enumerate(bn) if "Foot" in b]

    rows = []
    for path in sorted(glob.glob(f"{args.motion_dir}/*.motion")):
        d = torch.load(path, weights_only=False, map_location="cpu")
        p = d["rigid_body_pos"].numpy()
        T = p.shape[0]
        if T < 4:
            continue
        dt = 1.0 / float(d.get("fps", 30))
        # COM from mass-weighted body origins (good enough to spot floating)
        com = (p * mass[None, :, None]).sum(1) / total
        # --- foot slip while planted
        slip = 0.0
        for fi in feet:
            h = p[:, fi, 2]
            planted = h < (h.min() + 0.02)
            step = np.linalg.norm(np.diff(p[:, fi, :2], axis=0), axis=-1)
            both = planted[:-1] & planted[1:]
            slip += float(step[both].sum())
        travel = float(np.linalg.norm(p[-1, 0, :2] - p[0, 0, :2]))
        # --- floating: COM vertical accel while no foot is near the ground
        low = np.min(p[:, feet, 2], axis=1)
        air = low > (low.min() + 0.05)
        az = np.gradient(np.gradient(com[:, 2], dt), dt)
        float_err = float(np.abs(az[air] + 9.81).mean()) if air.any() else 0.0
        # --- teleports
        jump = float(np.linalg.norm(np.diff(p[:, 0], axis=0), axis=-1).max() / dt)
        rows.append(dict(name=os.path.basename(path), frames=T, slip=slip,
                         travel=travel, air=float(air.mean()),
                         float_err=float_err, jump=jump))

    def show(title, key, fmt, reverse=True, filt=None):
        sel = [r for r in rows if (filt is None or filt(r))]
        sel.sort(key=lambda r: r[key], reverse=reverse)
        print(f"\n=== {title}")
        for r in sel[:args.top]:
            print(f"  {r['name']:<40}{fmt(r)}")

    print(f"scanned {len(rows)} clips in {args.motion_dir}")
    show("worst FOOT SLIP (cm while planted; >20 cm is heavy)", "slip",
         lambda r: f"{r['slip']*100:6.1f} cm   travel {r['travel']:.2f} m")
    show("worst FLOATING (|COM accel + g| while airborne, m/s^2)", "float_err",
         lambda r: f"{r['float_err']:6.1f}   airborne {100*r['air']:.0f}% of frames",
         filt=lambda r: r["air"] > 0.05)
    show("worst TELEPORTS (max root speed, m/s)", "jump",
         lambda r: f"{r['jump']:6.1f} m/s")
    heavy = [r for r in rows if r["slip"] > 0.20 or r["float_err"] > 6.0
             or r["jump"] > 12.0]
    print(f"\nclips exceeding a heavy-artifact threshold: {len(heavy)}/{len(rows)}")
    if heavy:
        print("  " + ", ".join(r["name"].replace(".motion", "")
                               for r in heavy[:20]))


main()
