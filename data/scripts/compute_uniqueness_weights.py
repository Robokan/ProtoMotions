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
"""Re-weight motion clips by uniqueness.

MotionLib samples clips proportionally to ``motion_weights``. A flat (uniform)
or hand-tuned weighting over-trains on the redundant bulk of a mocap library
(e.g. dozens of near-identical forward-walk clips) and under-trains on the rare
behaviours (a jump, a spin, a climb) that the policy most needs the extra
exposure to learn.

This script derives a data-driven weight per clip:

  1. Build a fixed-length, duration-invariant DESCRIPTOR per clip from summary
     statistics of root motion + joint pose/activity (see ``clip_descriptor``).
  2. z-score normalise each descriptor dimension across the library.
  3. Estimate each clip's local DENSITY as the sum of a Gaussian kernel over its
     distance to every other clip (bandwidth = median pairwise distance). A clip
     surrounded by many similar clips has high density; an isolated/unique clip
     has low density.
  4. weight ~ 1 / density  ->  redundant clips down, unique clips up. Ratios are
     clamped (``--max-ratio``) so a lone outlier can't dominate sampling, then
     normalised to sum to 1.

Optionally ``--blend-existing`` multiplies the uniqueness weight by the clip's
current weight (preserves a meaningful hand-tuned importance prior, e.g. Go2's)
before normalising.

Usage:
    python data/scripts/compute_uniqueness_weights.py \
        --motion-lib data/motions/anymal_d/anymal_d_full.pt \
        --feet 4 8 12 16
    # in-place by default; --output writes a copy, --dry-run only reports
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _per_clip_slices(m: dict):
    starts = m["length_starts"].numpy().astype(int)
    nframes = m["motion_num_frames"].numpy().astype(int)
    return [slice(int(s), int(s) + int(n)) for s, n in zip(starts, nframes)]


def clip_descriptor(m: dict, sl: slice, fps: float, feet: list[int]) -> np.ndarray:
    """Duration-invariant summary vector for one clip.

    Captures *what kind of motion* it is (not how long): how fast/which way the
    root travels, how much it bobs/turns/jumps, the joint pose distribution, and
    how active the joints are. Quantities are per-second where rate-like so clips
    of different length are comparable.
    """
    gts = m["gts"][sl].numpy()  # (T, B, 3) global body positions
    gvs = m["gvs"][sl].numpy()  # (T, B, 3) linear velocities
    gavs = m["gavs"][sl].numpy()  # (T, B, 3) angular velocities
    dps = m["dps"][sl].numpy()  # (T, D) dof positions
    dvs = m["dvs"][sl].numpy()  # (T, D) dof velocities

    root_z = gts[:, 0, 2]
    root_v = gvs[:, 0, :]
    root_w = gavs[:, 0, :]
    ground = np.percentile(gts[:, feet, 2], 5)

    feats = []
    # root height: posture (crouch / rear / normal) and its variability (bob/jump)
    feats += [root_z.mean(), root_z.std(), root_z.max() - root_z.min()]
    # planar travel speed
    sp = np.linalg.norm(root_v[:, :2], axis=1)
    feats += [sp.mean(), sp.std(), sp.max()]
    # signed travel components -> direction (backwards / sideways locomotion)
    feats += [root_v[:, 0].mean(), root_v[:, 1].mean()]
    # vertical motion -> hops / jumps
    feats += [np.abs(root_v[:, 2]).mean(), np.abs(root_v[:, 2]).max()]
    # turning (yaw rate) and overall angular agitation
    feats += [np.abs(root_w[:, 2]).mean(), np.linalg.norm(root_w, axis=1).mean()]
    # feet elevation above ground -> climbing / stepping up
    foot_elev = gts[:, feet, 2] - ground
    feats += [foot_elev.max(), foot_elev.mean()]
    # joint pose distribution (per-dof mean & std) -> the "shape" of the motion
    feats += list(dps.mean(axis=0))
    feats += list(dps.std(axis=0))
    # joint activity (per-dof mean |velocity|) -> how vigorously each joint moves
    feats += list(np.abs(dvs).mean(axis=0))
    return np.asarray(feats, dtype=np.float64)


def uniqueness_weights(desc: np.ndarray, max_ratio: float) -> np.ndarray:
    """Inverse-density weights from a (N, F) descriptor matrix."""
    # z-score each dimension (guard zero-variance dims)
    mu = desc.mean(axis=0)
    sd = desc.std(axis=0)
    sd[sd < 1e-8] = 1.0
    z = (desc - mu) / sd

    # pairwise euclidean distances
    diff = z[:, None, :] - z[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))  # (N, N)

    # bandwidth = median of off-diagonal distances (robust scale)
    iu = np.triu_indices(len(z), k=1)
    sigma = np.median(dist[iu])
    if sigma < 1e-8:
        sigma = 1.0

    # local density = sum of Gaussian affinities (includes self=1)
    density = np.exp(-(dist ** 2) / (2.0 * sigma ** 2)).sum(axis=1)
    w = 1.0 / density

    # clamp dynamic range so a lone outlier can't monopolise sampling
    lo = w.min()
    w = np.minimum(w, lo * max_ratio)
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion-lib", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None,
                    help="write here instead of in-place")
    ap.add_argument("--feet", type=int, nargs="+", required=True,
                    help="foot body indices (anymal/go2: 4 8 12 16)")
    ap.add_argument("--max-ratio", type=float, default=20.0,
                    help="max weight / min weight ratio (clamp outliers)")
    ap.add_argument("--blend-existing", action="store_true",
                    help="multiply uniqueness by current weights before normalising")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write")
    args = ap.parse_args()

    m = torch.load(args.motion_lib, weights_only=False, map_location="cpu")
    fps = float(m["motion_dt"][0] ** -1) if "motion_dt" in m else 60.0
    slices = _per_clip_slices(m)
    files = m.get("motion_files", [f"clip_{i}" for i in range(len(slices))])

    desc = np.stack([clip_descriptor(m, sl, fps, args.feet) for sl in slices])
    w = uniqueness_weights(desc, args.max_ratio)

    if args.blend_existing and "motion_weights" in m:
        old = m["motion_weights"].numpy().astype(np.float64)
        w = w * old
        w = w / w.sum()

    old = m.get("motion_weights")
    old = old.numpy() if old is not None else np.full(len(w), 1.0 / len(w))

    # report extremes
    order = np.argsort(w)
    name = lambda i: Path(str(files[i])).stem  # noqa: E731
    print(f"clips: {len(w)}   weight range: {w.min():.5f} .. {w.max():.5f} "
          f"(ratio {w.max() / w.min():.1f}x)")
    print("\nMOST UNIQUE (highest weight):")
    for i in order[::-1][:12]:
        print(f"  {name(i):28s} {w[i]:.5f}   (was {old[i]:.5f})")
    print("\nMOST REDUNDANT (lowest weight):")
    for i in order[:12]:
        print(f"  {name(i):28s} {w[i]:.5f}   (was {old[i]:.5f})")

    if args.dry_run:
        print("\n[dry-run] not written")
        return

    m["motion_weights"] = torch.tensor(w, dtype=torch.float32)
    out = args.output or args.motion_lib
    torch.save(m, out)
    print(f"\nwrote uniqueness weights -> {out}")


if __name__ == "__main__":
    main()
