#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Balance a motion library's left- and right-turning clips by count.

A policy captured without heading control has a preferred turn direction, so a
corpus harvested from it is lopsided -- go2_reversed_mocap_v3 is 89% left / 10%
right. An LLC trained on that turns one way well and the other way barely.

Clips are classified by MEAN yaw rate over the clip (net yaw / duration), so a
clip that wanders both ways nets out as straight rather than counting as both.
The majority side is then cut down to the minority's count. Straight clips are
always kept -- they are the scarce material, not the surplus.

Which majority clips survive is not a random draw: each is paired with the
minority clip closest to it in |yaw rate|, so the kept set MIRRORS the minority
side's distribution of turn rates instead of piling up at the majority's mode.
A random subsample of the same size would keep the count balanced and leave the
rates lopsided.

    python data/scripts/balance_motion_lib_by_turn.py \
        --in-lib  data/motions/go2/go2_reversed_mocap_v3.pt \
        --out-lib data/motions/go2/go2_reversed_mocap_v3_balanced.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

PER = ["gts", "grs", "gvs", "gavs", "dvs", "dps"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-lib", required=True)
    ap.add_argument("--out-lib", required=True)
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="[rad/s] |mean yaw rate| above this is a turn; "
                         "below it the clip counts as straight")
    ap.add_argument("--target-total", type=int, default=0,
                    help="approximate final clip count. 0 = cut the majority "
                         "side down to the minority's size and keep everything "
                         "else. Straight clips are kept first (they are the "
                         "scarce material), then the remainder is split evenly "
                         "between left and right.")
    ap.add_argument("--apply", action="store_true", default=True)
    args = ap.parse_args()

    m = torch.load(args.in_lib, weights_only=False, map_location="cpu")
    starts = m["length_starts"].numpy().astype(int)
    counts = m["motion_num_frames"].numpy().astype(int)
    dts = m["motion_dt"].numpy()
    weights = m["motion_weights"].numpy()
    files = [str(f) for f in m["motion_files"]]
    grs = m["grs"].numpy()

    yaw_rate = np.zeros(len(counts))
    for i, (s, k) in enumerate(zip(starts, counts)):
        y = np.unwrap(R.from_quat(grs[s : s + k, 0]).as_euler("zyx")[:, 0])
        yaw_rate[i] = (y[-1] - y[0]) / max((k - 1) * float(dts[i]), 1e-9)

    left = np.flatnonzero(yaw_rate > args.threshold)
    right = np.flatnonzero(yaw_rate < -args.threshold)
    straight = np.flatnonzero(np.abs(yaw_rate) <= args.threshold)
    print(f"{len(files)} clips: {len(left)} left, {len(right)} right, "
          f"{len(straight)} straight (|yaw| <= {args.threshold})")

    major, minor = (left, right) if len(left) >= len(right) else (right, left)
    side = "left" if len(left) >= len(right) else "right"

    keep_straight = straight
    n_side = len(minor)
    if args.target_total:
        keep_straight = straight[: max(0, args.target_total - 2)]
        n_side = max(1, (args.target_total - len(keep_straight)) // 2)
        n_side = min(n_side, len(minor))

    # Spread the minority picks across its |yaw rate| range via evenly spaced
    # quantiles. Taking the first N, or a uniform random draw, would cluster at
    # the mode and throw away the sharp turns -- which are the whole reason to
    # keep turning clips.
    mo = minor[np.argsort(np.abs(yaw_rate[minor]))]
    idx = np.unique(np.linspace(0, len(mo) - 1, n_side).round().astype(int))
    keep_minor = mo[idx]

    # pair each kept minority clip with the nearest-magnitude unused majority clip
    pool = list(major)
    keep_major = []
    for j in keep_minor[np.argsort(-np.abs(yaw_rate[keep_minor]))]:
        if not pool:
            break
        k = min(pool, key=lambda c: abs(abs(yaw_rate[c]) - abs(yaw_rate[j])))
        pool.remove(k)
        keep_major.append(k)

    keep = np.sort(np.concatenate([
        np.array(keep_major, dtype=int),
        np.asarray(keep_minor, dtype=int),
        np.asarray(keep_straight, dtype=int),
    ]))
    other = "right" if side == "left" else "left"
    print(f"cutting {side}: {len(major)} -> {len(keep_major)}")
    print(f"cutting {other}: {len(minor)} -> {len(keep_minor)}")
    print(f"straight: {len(straight)} -> {len(keep_straight)}")
    print(f"keeping {len(keep)} clips total")
    print(f"  |yaw rate| kept {side:5s}: "
          f"{np.abs(yaw_rate[keep_major]).min():.3f}.."
          f"{np.abs(yaw_rate[keep_major]).max():.3f}")
    print(f"  |yaw rate| kept {other:5s}: "
          f"{np.abs(yaw_rate[keep_minor]).min():.3f}.."
          f"{np.abs(yaw_rate[keep_minor]).max():.3f}")

    out = {k: [] for k in PER}
    nf, dt_l, ln, w, nm = [], [], [], [], []
    for i in keep:
        s, k = int(starts[i]), int(counts[i])
        for key in PER:
            out[key].append(m[key][s : s + k])
        nf.append(k)
        dt_l.append(float(dts[i]))
        ln.append((k - 1) * float(dts[i]))
        w.append(float(weights[i]))
        nm.append(files[i])

    packed = {k: torch.cat(v, dim=0).contiguous() for k, v in out.items()}
    nf_t = torch.tensor(nf, dtype=m["motion_num_frames"].dtype)
    st = nf_t.roll(1)
    st[0] = 0
    packed["length_starts"] = st.cumsum(0).to(m["length_starts"].dtype)
    packed["motion_num_frames"] = nf_t
    packed["motion_dt"] = torch.tensor(dt_l, dtype=m["motion_dt"].dtype)
    packed["motion_lengths"] = torch.tensor(ln, dtype=m["motion_lengths"].dtype)
    packed["motion_weights"] = torch.tensor(w, dtype=m["motion_weights"].dtype)
    packed["motion_files"] = tuple(nm)
    torch.save(packed, args.out_lib)

    chk = torch.load(args.out_lib, weights_only=False, map_location="cpu")
    assert int(chk["gts"].shape[0]) == int(chk["motion_num_frames"].sum())
    g = chk["gts"].numpy()
    s2 = chk["length_starts"].numpy()
    c2 = chk["motion_num_frames"].numpy()
    worst = max(
        (float(np.abs(np.diff(g[a : a + b, 0, :2], axis=0)).max())
         for a, b in zip(s2, c2) if b > 1),
        default=0.0,
    )
    assert worst < 0.5, f"teleport leaked in: {worst:.2f} m single-frame step"
    yr2 = np.array([
        (lambda y: (y[-1] - y[0]) / max((b - 1) * float(d), 1e-9))(
            np.unwrap(R.from_quat(g_[:, 0]).as_euler("zyx")[:, 0]))
        for g_, b, d in ((chk["grs"].numpy()[a : a + b], b, d)
                         for a, b, d in zip(s2, c2, chk["motion_dt"].numpy()))
    ])
    print(f"\n  wrote {args.out_lib}  (verified on reload)")
    print(f"    {len(chk['motion_files'])} clips, {g.shape[0]} frames, "
          f"{float(chk['motion_lengths'].sum()):.1f}s")
    print(f"    left {int((yr2 > args.threshold).sum())}  "
          f"right {int((yr2 < -args.threshold).sum())}  "
          f"straight {int((np.abs(yr2) <= args.threshold).sum())}")
    print(f"    yaw rate median {np.median(yr2):+.3f} rad/s, "
          f"largest single-frame step {worst:.4f} m")


if __name__ == "__main__":
    main()
