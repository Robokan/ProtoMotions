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
"""Repair short orientation-flip glitches in retargeted poselib .npy dog clips.

The dm_control dog retarget produced occasional bad frames where a body's
*world* orientation jumps ~pi for a few frames and flips back -- e.g. clip 44's
Neck flips at frame 11 and returns at frame 14. These are glitches in the source
rotations (not gimbal/decomposition artifacts) and are what make the dog's
head/neck/tail "flip over" mid-clip (and likely drove training non-finite).

Why world space: a body and its child often flip together (the child's local
rotation compensates so the child's *world* stays smooth). Detecting/bridging in
*world* space and then recomputing locals handles that coupling automatically --
repairing the neck without breaking the head.

Algorithm (per body, on world orientation built by FK from the local quats):
    1. find frame-to-frame geodesic steps > THR (flip transitions)
    2. pair an opening flip at i1 with the next flip at i2 (i2-i1 <= MAX_RUN)
       when the clean frames bounding the region agree (the motion continues
       smoothly across the glitch): geodesic(W[i1], W[i2+1]) < THR
    3. replace the flipped interior frames W[i1+1 .. i2] with slerp(W[i1],W[i2+1])
Then recompute every body's local quat from the repaired world orientations and
write the clip back out (originals are left untouched in --in-dir).

Usage:
    python data/scripts/clean_source_quat_spikes.py \
        --in-dir  data/motions/dog_v2/npy \
        --out-dir data/motions/dog_v2/npy_clean
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ---- quaternion helpers (wxyz) -------------------------------------------------


def _norm(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _conj(q):
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _geo(a, b):
    dot = np.abs((a * b).sum(-1)).clip(-1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _slerp(q0, q1, t):
    q0 = _norm(q0)
    q1 = _norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        out = q0[None] + t[:, None] * (q1 - q0)[None]
        return _norm(out)
    th0 = np.arccos(d)
    s0 = np.sin((1.0 - t) * th0) / np.sin(th0)
    s1 = np.sin(t * th0) / np.sin(th0)
    return s0[:, None] * q0[None] + s1[:, None] * q1[None]


# ---- repair --------------------------------------------------------------------


def fk_world(local, parents):
    """local (T,B,4) wxyz relative-to-parent -> world (T,B,4)."""
    T, B, _ = local.shape
    world = np.empty_like(local)
    for b in range(B):
        p = int(parents[b])
        world[:, b] = local[:, b] if p < 0 else _mul(world[:, p], local[:, b])
    return world


def repair_body(W_b, thr, max_run):
    """W_b (T,4) one body's world quats -> (repaired copy, n_frames_fixed)."""
    W = W_b.copy()
    T = W.shape[0]
    steps = _geo(W[1:], W[:-1])  # steps[i] is between frame i and i+1
    bigs = list(np.where(steps > thr)[0])
    n_fixed = 0
    used = 0
    while used < len(bigs):
        i1 = bigs[used]
        # find the paired closing flip
        matched = False
        for k in range(used + 1, len(bigs)):
            i2 = bigs[k]
            if i2 - i1 > max_run:
                break
            if i2 + 1 >= T:
                continue
            # Out-and-back glitch test: the clean frames bounding the region must
            # be much closer to each other than to the flipped interior. This
            # catches short pi flips that happen DURING genuine fast motion
            # (where the bounding frames don't agree in absolute terms) while
            # leaving sustained real motion (no return) untouched.
            ends = _geo(W[i1], W[i2 + 1])
            flip_mag = max(_geo(W[i1], W[i1 + 1]), _geo(W[i2], W[i2 + 1]))
            if ends < thr or ends < 0.5 * flip_mag:
                run = i2 - i1
                ts = (np.arange(1, run + 1) / (run + 1)).astype(W.dtype)
                W[i1 + 1 : i2 + 1] = _slerp(W[i1], W[i2 + 1], ts).astype(W.dtype)
                n_fixed += run
                used = k + 1
                matched = True
                break
        if not matched:
            used += 1
    return W, n_fixed


def clean_clip(local, parents, thr, max_run):
    world = fk_world(local, parents)
    total = 0
    for b in range(world.shape[1]):
        world[:, b], n = repair_body(world[:, b], thr, max_run)
        total += n
    # recompute locals from repaired world
    new_local = np.empty_like(local)
    for b in range(world.shape[1]):
        p = int(parents[b])
        new_local[:, b] = (
            world[:, b] if p < 0 else _mul(_conj(world[:, p]), world[:, b])
        )
    return _norm(new_local).astype(local.dtype), total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--thr", type=float, default=1.2, help="flip threshold (rad)")
    ap.add_argument(
        "--max-run", type=int, default=8, help="max flipped frames to bridge"
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.in_dir.glob("*.npy"))
    total = 0
    touched = 0
    for p in files:
        d = np.load(p, allow_pickle=True).item()
        local = np.asarray(d["rotation"]["arr"])  # (T,B,4) wxyz local
        parents = np.asarray(d["skeleton_tree"]["parent_indices"]["arr"])
        cleaned, n = clean_clip(local, parents, args.thr, args.max_run)
        if n:
            d["rotation"]["arr"] = cleaned
            touched += 1
            total += n
            print(f"  {p.name}: fixed {n} flipped frames")
        np.save(args.out_dir / p.name, np.array(d, dtype=object), allow_pickle=True)
    print(f"Repaired {total} flipped frames across {touched}/{len(files)} clips")
    print(f"Wrote cleaned clips to {args.out_dir}")


if __name__ == "__main__":
    main()
