# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Zero-phase low-pass filter for BVH motion channels.

Upstream jitter fix for SEED/Reallusion sources. Each joint's euler
triplet is filtered in QUATERNION space (sign-continuous, renormalized)
and converted back with the joint's own rotation order — filtering euler
channels independently distorts rotations near +-180 crossings (turn
clips) and was observed to throw the downstream GMR IK onto a wrong
branch for a frame. Position channels are filtered directly. Hierarchy
passes through verbatim, so the output stays a drop-in for GMR
retarget_headless.

Usage:
    python data/scripts/lowpass_bvh.py --input-dir <bvhs> --output-dir <out> \\
        --cutoff-hz 8
"""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation


def filter_bvh(src: Path, dst: Path, cutoff_hz: float):
    text = src.read_text()
    m = re.search(r"Frame Time:\s*([0-9.eE+-]+)\s*\n", text)
    if not m:
        raise ValueError("no Frame Time line")
    fps = 1.0 / float(m.group(1))
    head, frames_txt = text[: m.end()], text[m.end():]
    rows = [r for r in frames_txt.splitlines() if r.strip()]
    data = np.array([[float(v) for v in r.split()] for r in rows])

    # Channel order per BVH spec appears in HIERARCHY; positions are the
    # channels named Xposition/Yposition/Zposition (root, usually first 3).
    chan_names = re.findall(r"CHANNELS\s+\d+\s+([^\n]+)", text)
    names = [c for grp in chan_names for c in grp.split()]
    if len(names) != data.shape[1]:
        raise ValueError(f"channel mismatch {len(names)} != {data.shape[1]}")
    is_pos = np.array([n.endswith("position") for n in names])

    nyq = fps / 2.0
    if cutoff_hz >= nyq or data.shape[0] < 10:
        dst.write_text(text)
        return
    b, a = butter(2, cutoff_hz / nyq, btype="low")
    out = data.copy()
    out[:, is_pos] = filtfilt(b, a, data[:, is_pos], axis=0)

    # Rotation triplets, one per joint, filtered as quaternions.
    idx = 0
    while idx < len(names):
        if names[idx].endswith("position"):
            idx += 1
            continue
        axes = "".join(n[0] for n in names[idx:idx + 3])
        if len(axes) != 3 or not all(
            n.endswith("rotation") for n in names[idx:idx + 3]
        ):
            raise ValueError(f"unexpected channel run at {idx}: {names[idx:idx+3]}")
        q = Rotation.from_euler(axes.upper(), data[:, idx:idx + 3], degrees=True).as_quat()
        flip = (q[1:] * q[:-1]).sum(-1) < 0.0
        sign = np.cumprod(np.where(flip, -1.0, 1.0), axis=0)
        q[1:] *= sign[..., None]
        q = filtfilt(b, a, q, axis=0)
        q /= np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-8)
        out[:, idx:idx + 3] = Rotation.from_quat(q).as_euler(axes.upper(), degrees=True)
        idx += 3

    body = "\n".join(" ".join(f"{v:.6f}" for v in row) for row in out)
    dst.write_text(head + body + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--bvh-list", type=Path, help="text file of BVH paths")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cutoff-hz", type=float, default=8.0)
    args = p.parse_args()

    if args.bvh_list:
        files = [Path(l.strip()) for l in args.bvh_list.read_text().splitlines() if l.strip()]
    else:
        files = sorted(args.input_dir.glob("*.bvh"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []
    for f in files:
        try:
            filter_bvh(f, args.output_dir / f.name, args.cutoff_hz)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed.append((f.name, str(exc)))
    print(f"filtered: {ok} | failed: {len(failed)}")
    for n, r in failed[:10]:
        print(f"  FAIL {n}: {r[:90]}")


if __name__ == "__main__":
    main()
