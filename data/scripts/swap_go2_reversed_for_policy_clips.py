#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Swap the go2 corpus's time-reversed walk clips for policy-captured ones.

The ASE corpus (go2_flat.pt) carries 8 `*_walk_backwards` clips: forward walks
played backwards, which are dynamically impossible (a footfall's deceleration
becomes an acceleration away from the ground). They were also deliberately
BOOSTED -- 0.25 each, 1.95 of the corpus's 7.108 total weight, so 27.4% of all
sampling comes from 4.2% of the duration. Backward locomotion is meant to be
well represented; the clips themselves were just unachievable.

This replaces them with clips captured from the AMP policy that learned from
them, which ARE physically achievable because a real robot produced them under
real physics.

WEIGHTING. The incoming clips carry weight 1.0 each from the capture pipeline.
Merging them at face value would hand 39 clips a weight of 39 against a base
that sums to ~5.2 -- backward walking would become 88% of all sampling. Instead
the removed clips' TOTAL weight is divided across the incoming ones, so the
corpus keeps exactly the backward-motion share it was tuned to have.

    python data/scripts/swap_go2_reversed_for_policy_clips.py \\
        --base data/motions/go2/go2_flat.pt \\
        --add  data/motions/go2/go2_reversed_mocap_balanced.pt \\
        --out  data/motions/go2/go2_flat_policy_backwards.pt
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch


def _load_merge_module():
    """Reuse merge_motion_libs' subset/merge rather than reimplementing them."""
    p = Path(__file__).with_name("merge_motion_libs.py")
    spec = importlib.util.spec_from_file_location("_mml", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/motions/go2/go2_flat.pt")
    ap.add_argument("--add", default="data/motions/go2/go2_reversed_mocap_balanced.pt")
    ap.add_argument("--out", default="data/motions/go2/go2_flat_policy_backwards.pt")
    ap.add_argument("--drop-substring", default="walk_backwards")
    args = ap.parse_args()

    mml = _load_merge_module()

    base = torch.load(args.base, weights_only=False, map_location="cpu")
    add = torch.load(args.add, weights_only=False, map_location="cpu")

    files = [str(f) for f in base["motion_files"]]
    drop = [i for i, f in enumerate(files) if args.drop_substring in f]
    keep = [i for i in range(len(files)) if i not in set(drop)]
    if not drop:
        raise SystemExit(f"no base clip matches {args.drop_substring!r}")

    wb = base["motion_weights"].float()
    dropped_w = float(wb[drop].sum())
    print(f"base {args.base}: {len(files)} clips, "
          f"{float(base['motion_lengths'].sum()):.0f}s, weight sum {float(wb.sum()):.3f}")
    print(f"  dropping {len(drop)} '{args.drop_substring}' clips "
          f"({float(base['motion_lengths'].numpy()[drop].sum()):.1f}s, "
          f"weight {dropped_w:.3f} = {100*dropped_w/float(wb.sum()):.1f}% of sampling)")
    for i in drop:
        print(f"    - {files[i].split('/')[-1]}  w={float(wb[i]):.3f}")

    trimmed = mml.subset(base, keep)

    # hand the incoming clips the removed group's TOTAL weight, split evenly
    n_add = len(add["motion_files"])
    per_clip = dropped_w / n_add
    add = dict(add)
    add["motion_weights"] = torch.full((n_add,), per_clip, dtype=torch.float32)
    print(f"add  {args.add}: {n_add} clips, "
          f"{float(add['motion_lengths'].sum()):.1f}s")
    print(f"  per-clip weight {per_clip:.5f} (= {dropped_w:.3f} / {n_add})")

    merged = mml.merge(trimmed, add, match_weight_of=None)
    torch.save(merged, args.out)

    # verify on reload, at the point of consumption
    chk = torch.load(args.out, weights_only=False, map_location="cpu")
    w = chk["motion_weights"].numpy()
    nm = [str(f) for f in chk["motion_files"]]
    nf = chk["motion_num_frames"].numpy()
    st = chk["length_starts"].numpy()
    assert int(chk["gts"].shape[0]) == int(nf.sum())
    assert int(st[-1]) + int(nf[-1]) == int(chk["gts"].shape[0])
    assert not any(args.drop_substring in f for f in nm), "reversed clips survived"
    assert abs(w.sum() - 1.0) < 1e-5, f"weights not normalized: {w.sum()}"
    new_idx = [i for i, f in enumerate(nm) if "policy_reversed" in f]
    print(f"\n  wrote {args.out}  (verified on reload)")
    print(f"    {len(nm)} clips, {chk['gts'].shape[0]} frames, "
          f"{float(chk['motion_lengths'].sum()):.0f}s")
    print(f"    policy clips: {len(new_idx)}, "
          f"{100*w[new_idx].sum():.1f}% of sampling weight "
          f"(was {100*dropped_w/float(wb.sum()):.1f}% for the reversed clips)")
    print(f"    weight range {w.min():.6f}..{w.max():.6f}, sum {w.sum():.6f}")
    print(f"    dt values present: {np.unique(np.round(chk['motion_dt'].numpy(),5))}")


if __name__ == "__main__":
    main()
