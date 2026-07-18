# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Scan a packed MotionLib for mangled retargets (kinematic sanity checks).

GMR IK occasionally produces garbage on a clip (limit-pegged limbs, joint
teleports between frames, bodies flying). This scores every clip on cheap
kinematic metrics and flags outliers so they can be dropped from training
and queued for a GMR fix later.

Per-clip metrics:
  dof_spike   max frame-to-frame |dof delta| * fps (rad/s) — IK flips/teleports
  body_speed  max rigid-body speed (m/s) — flying bodies
  sat_frac    worst per-joint fraction of frames within 2% of a joint limit
  sat_joints  number of joints saturated >30% of frames

Usage (host, CPU only — safe while training runs):
    python data/scripts/scan_motion_lib_quality.py \
        --lib data/atlas_tracker_stage2.pt --mjcf <atlas_physics.xml> \
        --out data/atlas_stage2_quality
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--lib", required=True)
    p.add_argument("--mjcf", required=True, help="robot MJCF for joint ranges")
    p.add_argument("--out", required=True, help="output prefix (.csv/.json/_bad.txt)")
    p.add_argument("--dof-spike-max", type=float, default=40.0, help="rad/s")
    p.add_argument("--body-speed-max", type=float, default=25.0, help="m/s")
    p.add_argument("--sat-joints-max", type=int, default=4,
                   help="flag if more joints than this are saturated >30%% of frames")
    args = p.parse_args()

    import mujoco

    m = mujoco.MjModel.from_xml_path(args.mjcf)
    lo, hi = [], []
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        lo.append(m.jnt_range[j][0])
        hi.append(m.jnt_range[j][1])
    lo, hi = np.array(lo), np.array(hi)
    span = np.maximum(hi - lo, 1e-6)

    d = torch.load(args.lib, map_location="cpu", weights_only=False)
    files = [str(f).split("/")[-1].replace(".motion", "") for f in d["motion_files"]]
    starts = d["length_starts"].long().numpy()
    nf = d["motion_num_frames"].long().numpy()
    dps = d["dps"].numpy()   # [T, ndof] dof positions
    gvs = d["gvs"].numpy()   # [T, nbodies, 3] body velocities
    assert dps.shape[1] == len(lo), (dps.shape, len(lo))

    rows = []
    for i, name in enumerate(files):
        s, e = starts[i], starts[i] + nf[i]
        fps = round(1.0 / float(d["motion_dt"][i]))
        q = dps[s:e]
        dq = np.abs(np.diff(q, axis=0)).max() * fps if e - s > 1 else 0.0
        speed = float(np.linalg.norm(gvs[s:e], axis=-1).max())
        sat = ((q - lo) / span < 0.02) | ((hi - q) / span < 0.02)
        sat_frac_per_joint = sat.mean(axis=0)
        rows.append(dict(
            name=name,
            frames=int(e - s),
            dof_spike=float(dq),
            body_speed=speed,
            sat_frac=float(sat_frac_per_joint.max()),
            sat_joints=int((sat_frac_per_joint > 0.30).sum()),
        ))

    bad = [r for r in rows if r["dof_spike"] > args.dof_spike_max
           or r["body_speed"] > args.body_speed_max
           or r["sat_joints"] > args.sat_joints_max]
    bad_names = sorted(r["name"] for r in bad)

    out = Path(args.out)
    with open(f"{out}.json", "w") as f:
        json.dump(dict(thresholds=vars(args), rows=rows), f)
    with open(f"{out}_bad.txt", "w") as f:
        f.write("\n".join(bad_names) + ("\n" if bad_names else ""))

    for key in ("dof_spike", "body_speed", "sat_joints"):
        v = np.array([r[key] for r in rows])
        print(f"{key}: p50={np.percentile(v,50):.2f} p95={np.percentile(v,95):.2f} "
              f"p99={np.percentile(v,99):.2f} max={v.max():.2f}")
    print(f"\nflagged {len(bad_names)}/{len(rows)} clips -> {out}_bad.txt")
    for r in sorted(bad, key=lambda r: -max(r["dof_spike"] / args.dof_spike_max,
                                            r["body_speed"] / args.body_speed_max))[:15]:
        print(f"  {r['name']}: spike={r['dof_spike']:.0f} rad/s "
              f"speed={r['body_speed']:.1f} m/s sat_joints={r['sat_joints']}")


if __name__ == "__main__":
    main()
