# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Zero-phase low-pass of retargeted .motion clips to remove retarget tremor.

DO NOT USE THIS IN THE ATLAS PIPELINE. It filters GMR *output*, which Eric
vetoed on 2026-07-24 -- it looked worse and was reverted, and the agreed
principle is to fix jitter upstream in BVH emission. That upstream fix already
exists and is already applied: lowpass_bvh.py (BVH euler channels, filtered in
quaternion space) and convert_manny_npy_to_soma.py --lowpass-hz 8, both landed
for corpus v9 and inherited by v10/v11. Verified on shared stems -- the v11
source clips carry 8.61% of dof_pos energy above 8 Hz against 8.55% for the
known-filtered atlas_seed_f8 clips, i.e. indistinguishable. Running this on top
is a SECOND filter over already-filtered data.

Kept only because the analysis in it is reusable for a source family that has
no upstream filter yet. Before reaching for it, check whether the corpus was
built through lowpass_bvh / --lowpass-hz first; for atlas the answer is yes.

A related caution: an earlier version of this file's docstring claimed GMR's IK
roughly halves the source tremor (17.4% -> 9.4%). That measurement was
CONFOUNDED -- it compared unfiltered bvh_combatviewer sources against clips
built from filtered sources, so the drop was mostly the upstream filter, not
the IK. GMR's contribution is unmeasured.


Eric spotted leg vibration in atlas_pretrain_corpus_v11 and correctly said it
predates the foot correction. Measured on the untouched retarget output
(135 clips, 30 fps):

    leg dofs: 6.94% of spectral energy above 8 Hz (worst clip 12.1%)
    arm dofs: 8.12%                              (worst clip 15.3%)
    noisiest dofs: Foot_R_Roll 2.785, Leg_8_L_Joint 2.646, Foot_L_Roll 2.507
                   (mean |2nd difference|, deg/frame^2)

The two noisiest DOFs are ankle ROLL, which the foot-pitch correction never
touches, so this is retarget noise, not a fix artifact. Arms are noisier than
legs; the eye just catches it in the legs because they are ground-contacting.

WHY FILTER THE VELOCITY FIELDS RATHER THAN RECOMPUTE THEM. Linear filtering
and differentiation commute: if v ~= dp/dt then filt(v) ~= d(filt(p))/dt. So
applying the SAME filtfilt to the velocity channels keeps them consistent with
the filtered positions while preserving the retarget's own velocity estimates,
which are smoother than anything finite differences would produce. Recomputing
by np.gradient instead would inject its own noise -- the mistake that made
fix_foot_clips' ankle dof_vel worse (20.8 -> 27.5 rad/s).

CUTOFF. 8 Hz at 30 fps (0.53 of Nyquist) -- a deliberately mild filter.
Gait fundamentals are 1-3 Hz and even fast strike transients sit below 8 Hz,
so the pass band holds the whole repertoire; the removed 8-15 Hz band is where
the tremor lives. Going lower would soften punches and kicks, which is the
opposite of what a combat corpus needs.

RUN THIS BEFORE fix_foot_clips.py, NEVER AFTER. Filtering moves the root and
every body, so it re-buries the feet; the foot correction must have the last
word on ground clearance.

    python data/scripts/lowpass_motion_clips.py \\
        --in-dir data/motions/atlas_v11_pre_footfix \\
        --out-dir data/motions/atlas_v11_smooth --cutoff 8
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from scipy.signal import butter, filtfilt

# Filtered as plain time series. Velocities are included on purpose (see the
# commute argument in the module docstring); quaternions need special handling
# and rigid_body_contacts is boolean, so both are excluded here.
LINEAR_KEYS = ("dof_pos", "dof_vel", "rigid_body_pos", "rigid_body_vel",
               "rigid_body_ang_vel")


def _filt(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """filtfilt along axis 0, flattening any trailing dims."""
    flat = x.reshape(x.shape[0], -1)
    pad = min(3 * max(len(a), len(b)), flat.shape[0] - 1)
    out = filtfilt(b, a, flat, axis=0, padlen=max(pad, 0))
    # filtfilt runs the filter backwards, so `out` carries negative strides and
    # torch.from_numpy would reject it.
    return np.ascontiguousarray(out).reshape(x.shape)


def _unwrap_quats(q: np.ndarray) -> np.ndarray:
    """Make a [T, B, 4] quaternion track sign-continuous in time.

    q and -q are the same rotation, and the retarget is free to emit either.
    An unflipped sign change is a 2x jump in every component, which a linear
    filter would smear across neighbouring frames as a real rotation.
    """
    q = q.copy()
    flip = np.cumprod(
        np.where((q[1:] * q[:-1]).sum(-1, keepdims=True) < 0, -1.0, 1.0), axis=0)
    q[1:] *= flip
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cutoff", type=float, default=8.0, help="Hz")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.in_dir}/*.motion"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no .motion files in {args.in_dir}")
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)

    tot = dict(clips=0, skipped=0, before=[], after=[])
    for p in paths:
        name = os.path.basename(p)
        mo = torch.load(p, weights_only=False, map_location="cpu")
        fps = float(mo.get("fps", 30))
        T = mo["dof_pos"].shape[0]
        nyq = 0.5 * fps
        if args.cutoff >= nyq:
            raise SystemExit(f"cutoff {args.cutoff} Hz >= Nyquist {nyq} Hz")
        b, a = butter(args.order, args.cutoff / nyq, btype="low")
        # filtfilt needs more frames than its padding; very short clips are
        # left alone rather than distorted at the edges.
        if T <= 3 * max(len(a), len(b)) + 1:
            tot["skipped"] += 1
            print(f"  {name[:58]:<58} SKIPPED (only {T} frames)")
            continue

        d0 = np.degrees(mo["dof_pos"].numpy().astype(np.float64))
        for k in LINEAR_KEYS:
            if k in mo:
                mo[k] = torch.from_numpy(
                    _filt(mo[k].numpy().astype(np.float64), b, a)).float()
        if "rigid_body_rot" in mo:
            q = _unwrap_quats(mo["rigid_body_rot"].numpy().astype(np.float64))
            q = _filt(q, b, a)
            q /= np.linalg.norm(q, axis=-1, keepdims=True)
            mo["rigid_body_rot"] = torch.from_numpy(q).float()

        d1 = np.degrees(mo["dof_pos"].numpy().astype(np.float64))
        c0 = float(np.abs(np.diff(d0, 2, axis=0)).mean())
        c1 = float(np.abs(np.diff(d1, 2, axis=0)).mean())
        tot["before"].append(c0)
        tot["after"].append(c1)
        tot["clips"] += 1
        if not args.dry_run:
            torch.save(mo, f"{args.out_dir}/{name}")
        print(f"  {name[:58]:<58} chatter {c0:6.3f} -> {c1:6.3f} "
              f"({100*(c1/max(c0,1e-9)-1):+5.1f}%)", flush=True)

    bo, af = np.array(tot["before"]), np.array(tot["after"])
    print(f"\n{tot['clips']} clips filtered at {args.cutoff} Hz "
          f"({tot['skipped']} too short, copied through)")
    if len(bo):
        print(f"  mean chatter {bo.mean():.3f} -> {af.mean():.3f} deg/frame^2 "
              f"({100*(af.mean()/bo.mean()-1):+.1f}%)")
    if args.dry_run:
        print("dry run: nothing written")
    else:
        print(f"written to {args.out_dir}")


if __name__ == "__main__":
    main()
