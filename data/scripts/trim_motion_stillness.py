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
"""Trim long static (standing-still) stretches out of motion clips.

Retargeted mocap (e.g. the dm_control dog from the MANN BVH library) contains
long stretches where the character just stands still. Training on a library
dominated by stillness wastes samples and biases the policy toward doing
nothing. This tool finds those stretches and cuts them out.

Stillness criterion: a frame is "still" when EVERY joint is barely moving, i.e.
``max_j |dof_vel[f, j]| < --still-joint-vel`` (after light smoothing). Because the
max is over ALL joints, a clip where the body is planted but the HEAD is moving
keeps a high max and is NOT counted as still -- that motion is useful.

For each run of stillness lasting at least ``--still-min-sec``:
  * the clip is SPLIT at the run, and
  * only ``--keep-sec`` of stillness is retained on each side (the first
    keep-sec stays on the end of the left sub-clip, the last keep-sec is
    prepended to the right sub-clip); the middle is removed.
So a 5 s still stretch with keep-sec=1 leaves 1 s + 1 s and drops the middle 3 s,
splitting the clip in two. Leading/trailing stillness is trimmed to keep-sec
(no split); a clip that is still throughout is dropped entirely.

Usage:
    python data/scripts/trim_motion_stillness.py \
        --motion-lib data/motions/dog_v2/dog_full.pt \
        --out-lib    data/motions/dog_v2/dog_trimmed.pt \
        --still-joint-vel 0.5 --still-min-sec 2.0 --keep-sec 1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

PER = ["gts", "grs", "gvs", "gavs", "dvs", "dps"]


def still_mask(dof_vel: np.ndarray, fps: float, thr: float, smooth_sec: float):
    """Per-frame boolean: True where ALL joints are ~still.

    Metric = max over joints of |joint velocity|, lightly smoothed so brief
    sensor spikes don't fragment a still run.
    """
    metric = np.abs(dof_vel).max(axis=1)  # (N,) -- max |joint vel| over all joints
    w = max(1, int(round(smooth_sec * fps)))
    if w > 1:
        metric = np.convolve(metric, np.ones(w) / w, mode="same")
    return metric < thr


def close_gaps(mask: np.ndarray, bridge_frames: int) -> np.ndarray:
    """Fill brief non-still gaps (< bridge_frames) between two still regions.

    Standing-still has occasional one-frame jitter spikes above threshold that
    would otherwise chop a long still stretch into many sub-threshold pieces.
    Bridging those momentary spikes lets a low ('all joints ~0') threshold still
    recognise the sustained stillness. Leading/trailing gaps are never filled.
    """
    if bridge_frames <= 0:
        return mask
    mask = mask.copy()
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            if 0 < i and j < n and (j - i) < bridge_frames:
                mask[i:j] = True  # gap between two still regions -> close it
            i = j
        else:
            i += 1
    return mask


def long_still_runs(mask: np.ndarray, min_frames: int, bridge_frames: int = 0):
    """Contiguous runs of `mask` (True=still) lasting >= min_frames, as [s, e)."""
    mask = close_gaps(mask, bridge_frames)
    runs = []
    s = None
    for i, v in enumerate(np.append(mask, False)):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= min_frames:
                runs.append((s, i))
            s = None
    return runs


def subclip_ranges(n, runs, keep, min_frames, mask):
    """Frame ranges [a, b) of the kept sub-clips after trimming/splitting.

    Each long still run splits the clip: the left sub-clip ends `keep` frames
    into the run, the right sub-clip starts `keep` frames before the run ends,
    the middle is dropped. Leading stillness is trimmed (no left clip); the
    trailing remainder is dropped if it has no motion.
    """
    subs = []
    cur = 0
    for s, e in runs:
        if s <= 0:  # leading stillness: keep only its last `keep`, no split
            cur = max(0, e - keep)
            continue
        subs.append((cur, min(n, s + keep)))  # close left sub-clip (+1s tail)
        cur = max(0, e - keep)  # right sub-clip starts 1s before still ends
    subs.append((cur, n))  # trailing remainder
    # keep only sub-clips long enough AND containing at least one moving frame
    return [
        (a, b) for a, b in subs
        if b - a >= min_frames and not mask[a:b].all()
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion-lib", required=True, type=Path)
    ap.add_argument("--out-lib", required=True, type=Path)
    ap.add_argument("--still-joint-vel", type=float, default=0.5,
                    help="[rad/s] a frame is still if max_j|dof_vel_j| is below this")
    ap.add_argument("--still-min-sec", type=float, default=2.0,
                    help="minimum stillness duration that triggers a split")
    ap.add_argument("--keep-sec", type=float, default=1.0,
                    help="stillness kept on each side of a split")
    ap.add_argument("--min-subclip-sec", type=float, default=1.0,
                    help="drop resulting sub-clips shorter than this")
    ap.add_argument("--smooth-sec", type=float, default=0.1,
                    help="smoothing window for the stillness metric")
    ap.add_argument("--bridge-sec", type=float, default=0.5,
                    help="close non-still gaps shorter than this (jitter spikes) "
                         "so sustained stillness isn't fragmented")
    args = ap.parse_args()

    m = torch.load(args.motion_lib, weights_only=False, map_location="cpu")
    starts = m["length_starts"].numpy().astype(int)
    nframes = m["motion_num_frames"].numpy().astype(int)
    dts = m["motion_dt"].numpy()
    weights = m["motion_weights"].numpy()
    files = [str(f) for f in m["motion_files"]]

    new = {k: [] for k in PER}
    out_files, out_nf, out_dt, out_len, out_w = [], [], [], [], []
    in_frames = out_frames = 0
    n_split = n_dropped = 0

    for i, path in enumerate(files):
        s0, n = int(starts[i]), int(nframes[i])
        dt = float(dts[i]); fps = 1.0 / dt
        in_frames += n
        sl = slice(s0, s0 + n)
        clip = {k: m[k][sl].numpy() for k in PER}

        mask = still_mask(clip["dvs"], fps, args.still_joint_vel, args.smooth_sec)
        mask = close_gaps(mask, int(round(args.bridge_sec * fps)))
        runs = long_still_runs(mask, int(round(args.still_min_sec * fps)))
        if not runs:
            ranges = [(0, n)]  # nothing to trim
        else:
            ranges = subclip_ranges(
                n, runs, int(round(args.keep_sec * fps)),
                int(round(args.min_subclip_sec * fps)), mask,
            )
            if len(ranges) != 1 or ranges[0] != (0, n):
                n_split += 1

        if not ranges:
            n_dropped += 1
            continue

        stem = path[:-len(".motion")] if path.endswith(".motion") else path
        multi = len(ranges) > 1
        for k, (a, b) in enumerate(ranges):
            name = f"{stem}__t{k}.motion" if multi else path
            for key in PER:
                new[key].append(clip[key][a:b])
            out_files.append(name)
            out_nf.append(b - a)
            out_dt.append(dt)
            out_len.append((b - a - 1) * dt)
            out_w.append(float(weights[i]))
            out_frames += b - a

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

    fps0 = 1.0 / float(dts[0])
    print(f"clips: {len(files)} -> {len(out_files)} "
          f"({n_split} split, {n_dropped} dropped as all-still)")
    print(f"duration: {in_frames / fps0:.0f}s -> {out_frames / fps0:.0f}s "
          f"(removed {(in_frames - out_frames) / fps0:.0f}s of stillness, "
          f"{100 * (in_frames - out_frames) / in_frames:.0f}%)")
    print(f"wrote {args.out_lib}")


if __name__ == "__main__":
    main()
