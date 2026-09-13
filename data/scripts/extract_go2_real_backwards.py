#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Cut the corpus's GENUINE backwards-walking segments into their own library.

The 8 `*_walk_backwards` clips are time-reversed forward walks: physically
impossible (a footfall's deceleration becomes an acceleration away from the
ground), so no policy can ever match them and the AMP discriminator keeps a
permanent gap. This finds the places where the source dog actually backed up,
which ARE achievable and therefore trainable.

Filter, and why each term is here:
  along-heading speed < -0.12 m/s   actually translating rearward
  |roll|,|pitch| < 25 deg           upright. WITHOUT this the sidestep
                                    extraction (2026-08) passed rearing and
                                    rolling clips -- a rearing dog does move
                                    its body backwards.
  rear hips not dropped             NOT redundant with pitch: a dog sits by
                                    FOLDING ITS REAR LEGS, so the trunk stays
                                    near level (max pitch across the first cut
                                    was only 16 deg) while the hindquarters
                                    sink. Eric spotted sitting clips that the
                                    pitch filter passed. Measuring front-hip
                                    minus rear-hip height catches it: standing
                                    is ~0, sitting reaches +14cm, and the data
                                    has a clean gap at 7.5..10.8cm.
  0.25 < root z < 0.45              standing, not lying down or leaping
  >= min-seconds                    long enough to contain a real step

Reversed clips are excluded by name so they can never leak back in.

    python data/scripts/extract_go2_real_backwards.py
    python data/scripts/extract_go2_real_backwards.py --apply
"""

import argparse

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

SRC = "data/motions/go2/go2_flat.pt"
DST = "data/motions/go2/go2_backwards_real.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=DST)
    ap.add_argument("--min-speed", type=float, default=-0.12,
                    help="along-heading speed must be below this (m/s)")
    ap.add_argument("--min-seconds", type=float, default=0.35)
    ap.add_argument("--max-tilt-deg", type=float, default=25.0)
    ap.add_argument("--max-rear-drop", type=float, default=0.06,
                    help="reject if front hips sit more than this above the "
                         "rear hips (m) -- the sitting/haunches signature")
    ap.add_argument("--min-clearance", type=float, default=0.26,
                    help="minimum trunk height above the feet (m). Standing is "
                         "0.287 (default_root_height); lying/crawling drops to "
                         "0.21. Root height alone cannot see this -- a lying dog "
                         "on raised ground still has a normal-looking root z.")
    ap.add_argument("--pad", type=int, default=3,
                    help="frames of context kept either side of a segment")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    from protomotions.robot_configs.factory import robot_config

    body_names = list(robot_config("go2").kinematic_info.body_names)
    FH = [body_names.index("FL_hip"), body_names.index("FR_hip")]
    RH = [body_names.index("RL_hip"), body_names.index("RR_hip")]
    FEET = [body_names.index(f + "_foot") for f in ("FL", "FR", "RL", "RR")]

    d = torch.load(args.src, map_location="cpu", weights_only=False)
    names = [str(x).split("/")[-1].replace(".motion", "") for x in d["motion_files"]]
    gts, grs = d["gts"].numpy(), d["grs"].numpy()
    starts = d["length_starts"].numpy()
    counts = d["motion_num_frames"].numpy()
    dts = d["motion_dt"].numpy()

    segments = []  # (source clip name, global start, global end)
    for i, name in enumerate(names):
        if "walk_backwards" in name:
            continue  # the reversed fakes
        s, k, dt = int(starts[i]), int(counts[i]), float(dts[i])
        fps = 1.0 / dt
        P = gts[s : s + k, 0]
        rpy = R.from_quat(grs[s : s + k, 0]).as_euler("zyx")
        yaw, pitch, roll = rpy[:, 0], rpy[:, 1], rpy[:, 2]

        vel = np.diff(P[:, :2], axis=0) / dt
        heading = np.stack([np.cos(yaw[:-1]), np.sin(yaw[:-1])], 1)
        along = (vel * heading).sum(1)

        seg_bodies = gts[s : s + k]
        rear_drop = (
            seg_bodies[:, FH, 2].mean(axis=1) - seg_bodies[:, RH, 2].mean(axis=1)
        )
        clearance = seg_bodies[:, 0, 2] - seg_bodies[:, FEET, 2].mean(axis=1)

        ok = (
            (along < args.min_speed)
            & (np.abs(np.degrees(roll[:-1])) < args.max_tilt_deg)
            & (np.abs(np.degrees(pitch[:-1])) < args.max_tilt_deg)
            & (P[:-1, 2] > 0.25)
            & (P[:-1, 2] < 0.45)
            & (rear_drop[:-1] < args.max_rear_drop)
            & (clearance[:-1] > args.min_clearance)
        )

        run_start = None
        for j, flag in enumerate(list(ok) + [False]):
            if flag and run_start is None:
                run_start = j
            elif not flag and run_start is not None:
                if (j - run_start) / fps >= args.min_seconds:
                    a = max(0, run_start - args.pad)
                    b = min(k, j + args.pad)
                    segments.append((name, s + a, s + b, (b - a) / fps,
                                     float(np.median(along[run_start:j]))))
                run_start = None

    total_s = sum(x[3] for x in segments)
    print(f"{len(segments)} segments, {total_s:.1f}s total")
    for nm, a, b, dur, sp in sorted(segments, key=lambda x: -x[3])[:10]:
        print(f"    {nm:34s} {dur:5.2f}s  {sp:+.3f} m/s")
    if len(segments) > 10:
        print(f"    ... +{len(segments)-10} more")

    if not args.apply:
        print("\n  (report only -- re-run with --apply)")
        return

    frame_idx = np.concatenate([np.arange(a, b) for _, a, b, _, _ in segments])
    n_src = int(d["gts"].shape[0])
    out = {}
    for key, v in d.items():
        if torch.is_tensor(v) and v.shape and v.shape[0] == n_src:
            out[key] = v[torch.from_numpy(frame_idx)]

    lens = torch.tensor([b - a for _, a, b, _, _ in segments], dtype=torch.long)
    out["motion_num_frames"] = lens
    shifted = lens.roll(1)
    shifted[0] = 0
    out["length_starts"] = shifted.cumsum(0)
    # Every segment inherits its source clip's frame rate.
    src_dt = []
    for nm, _, _, _, _ in segments:
        src_dt.append(float(dts[names.index(nm)]))
    out["motion_dt"] = torch.tensor(src_dt, dtype=torch.float32)
    out["motion_lengths"] = (lens - 1).float() * out["motion_dt"]
    out["motion_weights"] = torch.ones(len(segments), dtype=torch.float32)
    out["motion_files"] = tuple(
        f"{nm}_back{i:02d}.motion" for i, (nm, _, _, _, _) in enumerate(segments)
    )

    torch.save(out, args.out)

    chk = torch.load(args.out, map_location="cpu", weights_only=False)
    assert int(chk["gts"].shape[0]) == int(chk["motion_num_frames"].sum())
    P = chk["gts"].numpy()[:, 0]
    print(f"\n  wrote {args.out}")
    print(f"    {len(chk['motion_files'])} clips, {int(chk['gts'].shape[0])} frames, "
          f"{float(chk['motion_lengths'].sum()):.1f}s (verified on reload)")
    print(f"    root height {P[:,2].min():.3f}..{P[:,2].max():.3f} m")


if __name__ == "__main__":
    main()
