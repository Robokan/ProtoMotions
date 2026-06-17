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
"""Low-pass the retargeted dog clips to remove the high-frequency 'chihuahua
tremor' (a fast shake, most visible when the dog stands still) while keeping the
real gait.

A zero-phase Butterworth filter (filtfilt -> no lag) is applied per clip to:
  - each body's local quaternion (sign-made-continuous, filtered component-wise,
    then renormalized -- valid because the tremor is a small perturbation), and
  - the root translation.
Real locomotion is well below the cutoff (gait fundamentals ~1-3 Hz); the tremor
is much faster, so an ~8 Hz cutoff removes it without softening the motion.

Usage:
    python data/scripts/smooth_motion_tremor.py \
        --in-dir data/motions/dog_v2/npy_clean \
        --out-dir data/motions/dog_v2/npy_smooth --cutoff 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt


def _lp(x, b, a):
    # filtfilt needs length > 3*max(len(a),len(b)); short clips are passed through
    if x.shape[0] <= 3 * max(len(a), len(b)):
        return x
    return filtfilt(b, a, x, axis=0)


def smooth_clip(quat, root, fps, cutoff, order=4):
    b, a = butter(order, cutoff / (0.5 * fps), btype="low")
    # quaternions: enforce sign continuity, filter components, renormalize
    q = quat.copy()
    for j in range(1, q.shape[0]):
        flip = (q[j] * q[j - 1]).sum(-1) < 0  # per body
        q[j][flip] = -q[j][flip]
    q = _lp(q, b, a)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    r = _lp(root.copy(), b, a)
    return q.astype(quat.dtype), r.astype(root.dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cutoff", type=float, default=3.0, help="low-pass cutoff (Hz)")
    ap.add_argument("--order", type=int, default=4, help="Butterworth order")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.in_dir.glob("*.npy"))
    for p in files:
        d = np.load(p, allow_pickle=True).item()
        quat = np.asarray(d["rotation"]["arr"])           # (T, B, 4) wxyz local
        root = np.asarray(d["root_translation"]["arr"])   # (T, 3)
        fps = float(d["fps"])
        q, r = smooth_clip(quat, root, fps, args.cutoff, args.order)
        d["rotation"]["arr"] = q
        d["root_translation"]["arr"] = r
        np.save(args.out_dir / p.name, np.array(d, dtype=object), allow_pickle=True)
    print(f"smoothed {len(files)} clips (cutoff {args.cutoff} Hz) -> {args.out_dir}")


if __name__ == "__main__":
    main()
