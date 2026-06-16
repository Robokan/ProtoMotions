# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Split support clips into flat and support sub-clips.

Many motions that climb onto a platform spend most of their length walking or
jumping on flat ground and only briefly stand on the structure (e.g. anymal
41_clip_4: 35s long, support only 5.2-9.8s; 20_clip_1: a flat walk + a hop onto
a low platform + a separate flat jump). Training the whole clip on a support
cell wastes the flat travel (confined to an oversized cell) and the flat jump
rides along on terrain it doesn't need.

This tool reads the support manifest's ``support_segments`` and splits each
flagged clip at those boundaries:

  * support sub-clip(s)  -> the supported window (+ a small pad for the climb
    up/down transition); kept in the support list with boxes recomputed for
    that window against the clip's original ground baseline.
  * flat sub-clip(s)     -> the remaining travel; emitted as ordinary flat
    motions (no terrain) so they train on flat ground with everything else.

Flat clips pass through unchanged. The result is a new motion lib (.pt) plus a
new support manifest keyed by the sub-clip names.

Usage:
    python data/scripts/split_support_clips.py \
        --motion-lib data/motions/anymal_d/anymal_d_full.pt \
        --manifest   data/motions/anymal_d/support_manifest.yaml \
        --out-lib    data/motions/anymal_d/anymal_d_split.pt \
        --out-manifest data/motions/anymal_d/support_manifest_split.yaml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

# reuse the exact stance/box logic from the scanner so split boxes match
import importlib.util

_scan_path = Path(__file__).with_name("scan_clip_support_geometry.py")
_spec = importlib.util.spec_from_file_location("scan_support", _scan_path)
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)

FEET = [4, 8, 12, 16]
# Pad each support window by >=1s on both sides so the support sub-clip includes
# a full second of the robot ON THE GROUND before it jumps up and after it lands
# back down -- the climb-on / step-off transitions must be trainable, not clipped
# mid-air the instant the feet leave / before they settle.
PAD_SEC = 1.0  # seconds of ground to keep before jump-up and after jump-down
MIN_FLAT_SEC = 0.75  # drop flat fragments shorter than this (too short to train)
PER = ["gts", "grs", "gvs", "gavs", "dvs", "dps"]


def _merge(intervals):
    """Merge overlapping/adjacent (start, end) frame intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _segment_boxes(gts, gvs, a, b, ground_z):
    """Boxes for the support window [a, b) using the clip's original ground."""
    foot_pos = gts[:, FEET, :]
    foot_speed = np.linalg.norm(gvs[:, FEET, :], axis=-1)
    # elevated planted stances within the window, per foot
    pts = []
    for k in range(len(FEET)):
        pts += scan.foot_elevated_stances(
            foot_pos[a:b], foot_speed[a:b], k, ground_z,
            scan.STANCE_DETECT_SPEED, scan.STANCE_DETECT_MIN_FRAMES,
            scan.STANCE_DETECT_STD_MAX, scan.ELEVATION_THRESHOLD,
        )
    elevated = np.array(pts) if pts else np.empty((0, 3))
    # carve against the whole clip's foot trajectory (avoid clip-through)
    return scan.build_boxes(elevated, foot_pos.reshape(-1, 3), ground_z)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion-lib", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-lib", required=True, type=Path)
    ap.add_argument("--out-manifest", required=True, type=Path)
    ap.add_argument("--pad-sec", type=float, default=PAD_SEC,
                    help="ground time kept before jump-up / after jump-down")
    args = ap.parse_args()

    m = torch.load(args.motion_lib, weights_only=False, map_location="cpu")
    manifest = yaml.safe_load(open(args.manifest))
    starts = m["length_starts"].numpy().astype(int)
    nframes = m["motion_num_frames"].numpy().astype(int)
    dts = m["motion_dt"].numpy()
    weights = m["motion_weights"].numpy()
    files = [str(f) for f in m["motion_files"]]

    new = {k: [] for k in PER}
    out_files, out_nf, out_dt, out_len, out_w = [], [], [], [], []
    out_manifest = {}
    n_split = n_flat_subclips = n_support_subclips = 0

    def emit(name, gts, grs, gvs, gavs, dvs, dps, dt, weight):
        n = gts.shape[0]
        new["gts"].append(gts); new["grs"].append(grs)
        new["gvs"].append(gvs); new["gavs"].append(gavs)
        new["dvs"].append(dvs); new["dps"].append(dps)
        out_files.append(name)
        out_nf.append(n); out_dt.append(dt)
        out_len.append((n - 1) * dt); out_w.append(weight)

    for i, path in enumerate(files):
        base = os.path.basename(path)
        s0, n = int(starts[i]), int(nframes[i])
        dt = float(dts[i]); fps = 1.0 / dt
        sl = slice(s0, s0 + n)
        gts = m["gts"][sl].numpy(); grs = m["grs"][sl].numpy()
        gvs = m["gvs"][sl].numpy(); gavs = m["gavs"][sl].numpy()
        dvs = m["dvs"][sl].numpy(); dps = m["dps"][sl].numpy()
        entry = manifest.get(base, {})

        segs = entry.get("support_segments") if entry.get(
            "classification") == "needs_support" else None
        if not segs:
            # flat clip (or needs_support w/o segments): pass through unchanged
            emit(base, gts, grs, gvs, gavs, dvs, dps, dt, float(weights[i]))
            continue

        n_split += 1
        ground_z = float(np.percentile(gts[:, FEET, 2], 5))
        pad = int(round(args.pad_sec * fps))

        # Airborne "events": maximal runs where all 4 feet are off the floor (a
        # jump arc OR a climb-on/stand/step-off on a structure -- the robot keeps
        # all feet up the whole time). Cuts are placed only in the GROUNDED gaps
        # BETWEEN events, never through one, so no jump is split: each jump stays
        # whole in whichever sub-clip owns it.
        foot_elev = gts[:, FEET, 2] - ground_z
        airborne = (foot_elev > scan.ELEVATION_THRESHOLD).all(axis=1)
        events = []
        es = None
        for f, v in enumerate(np.append(airborne, False)):
            if v and es is None:
                es = f
            elif not v and es is not None:
                events.append((es, f))
                es = None
        seg_f = [(int(round(a * fps)), int(round(b * fps))) for a, b in segs]
        ev_support = [
            any(s < bb and e > aa for aa, bb in seg_f) for s, e in events
        ]
        nE = len(events)

        # Per-event clip bounds: extend up to `pad` ground frames each side, but
        # only to the midpoint of a gap shared with a neighbouring event (so the
        # neighbour's jump is not eaten). Outer ends extend by up to `pad`.
        sup = []
        for idx, (s, e) in enumerate(events):
            if not ev_support[idx]:
                continue
            prev_e = events[idx - 1][1] if idx > 0 else 0
            next_s = events[idx + 1][0] if idx < nE - 1 else n
            if idx == 0:
                lo = max(0, s - pad)
            elif (s - prev_e) <= 2 * pad:
                lo = (prev_e + s) // 2  # share the short gap with prev event
            else:
                lo = s - pad
            if idx == nE - 1:
                hi = min(n, e + pad)
            elif (next_s - e) <= 2 * pad:
                hi = (e + next_s) // 2  # share the short gap with next event
            else:
                hi = e + pad
            sup.append((lo, hi))
        sup = _merge(sup)
        # complement -> flat intervals (each starts/ends grounded, between events,
        # so any flat jump it contains is whole)
        flat, cur = [], 0
        for a, b in sup:
            if a > cur:
                flat.append((cur, a))
            cur = b
        if cur < n:
            flat.append((cur, n))

        stem = base[:-len(".motion")] if base.endswith(".motion") else base
        k = 0

        def sub(a, b):
            return (gts[a:b], grs[a:b], gvs[a:b], gavs[a:b], dvs[a:b], dps[a:b])

        for a, b in sup:
            name = f"{stem}__s{k}_support.motion"; k += 1
            emit(name, *sub(a, b), dt, float(weights[i]))
            n_support_subclips += 1
            boxes = _segment_boxes(gts, gvs, a, b, ground_z)
            rxy = gts[a:b, 0, :2]
            out_manifest[name] = {
                "classification": "needs_support",
                "ground_z": round(ground_z, 3),
                "duration_s": round((b - a) / fps, 2),
                "root_xy_min": [round(float(v), 3) for v in rxy.min(axis=0)],
                "root_xy_max": [round(float(v), 3) for v in rxy.max(axis=0)],
                "support_boxes": boxes,
                "from_clip": base,
            }
        for a, b in flat:
            if (b - a) < int(round(MIN_FLAT_SEC * fps)):
                continue  # too short to train on
            name = f"{stem}__s{k}_flat.motion"; k += 1
            emit(name, *sub(a, b), dt, float(weights[i]))
            n_flat_subclips += 1

    # pack
    cursor = 0
    new_starts = []
    for nf in out_nf:
        new_starts.append(cursor); cursor += nf
    out = {k: torch.from_numpy(np.concatenate(new[k], axis=0)) for k in PER}
    out["length_starts"] = torch.tensor(new_starts, dtype=m["length_starts"].dtype)
    out["motion_num_frames"] = torch.tensor(out_nf, dtype=m["motion_num_frames"].dtype)
    out["motion_dt"] = torch.tensor(out_dt, dtype=m["motion_dt"].dtype)
    out["motion_lengths"] = torch.tensor(out_len, dtype=m["motion_lengths"].dtype)
    out["motion_weights"] = torch.tensor(out_w, dtype=m["motion_weights"].dtype)
    out["motion_files"] = tuple(out_files)

    torch.save(out, args.out_lib)
    with open(args.out_manifest, "w") as f:
        yaml.safe_dump(out_manifest, f, sort_keys=True)

    print(f"input clips: {len(files)}  ->  output clips: {len(out_files)}")
    print(f"  split {n_split} support clips into "
          f"{n_support_subclips} support + {n_flat_subclips} flat sub-clips")
    print(f"  support sub-clips in new manifest: {len(out_manifest)}")
    print(f"wrote {args.out_lib}")
    print(f"wrote {args.out_manifest}")


if __name__ == "__main__":
    main()
