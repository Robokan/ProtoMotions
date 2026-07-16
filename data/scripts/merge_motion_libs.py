# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Merge two packed MotionLib .pt files into one.

Concatenates per-frame tensors, offsets ``length_starts``, and concatenates
per-motion fields. The incoming lib's clips can be assigned a per-clip weight
matched to a reference group in the base lib (e.g. "every added clip samples
like a shadow-boxing clip"); everything renormalizes at the end.

Example::

    python data/scripts/merge_motion_libs.py \\
        --base data/soma_combat_viewer.pt \\
        --add data/soma_combat_reallusion.pt \\
        --match-weight-of shadow_boxing \\
        --output data/soma_combat_viewer.pt
"""

import argparse
from pathlib import Path

import torch

PER_FRAME = ["gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts", "lrs"]
PER_MOTION = ["motion_lengths", "motion_dt", "motion_num_frames", "motion_weights"]


def merge(base: dict, add: dict, match_weight_of: str | None) -> dict:
    out = {}
    n_base_frames = base["gts"].shape[0]

    for key in PER_FRAME:
        if key in base and key in add:
            out[key] = torch.cat([base[key], add[key]], dim=0)
        elif key in base or key in add:
            raise ValueError(f"field {key!r} present in only one lib")

    for key in PER_MOTION:
        out[key] = torch.cat([base[key].float(), add[key].float()])
    out["motion_num_frames"] = out["motion_num_frames"].long()

    out["length_starts"] = torch.cat(
        [base["length_starts"], add["length_starts"] + n_base_frames]
    )
    out["motion_files"] = tuple(base["motion_files"]) + tuple(add["motion_files"])

    # Weights: keep the base lib's (possibly curated) distribution; added clips
    # each get the base's per-clip weight of the reference group.
    wb = base["motion_weights"].float()
    if match_weight_of is not None:
        files = [str(f) for f in base["motion_files"]]
        ref = [i for i, f in enumerate(files) if match_weight_of in f]
        if not ref:
            raise ValueError(f"no base clip matches {match_weight_of!r}")
        per_clip = float(wb[ref].sum() / len(ref))
        wa = torch.full((len(add["motion_files"]),), per_clip)
    else:
        wa = add["motion_weights"].float()
    w = torch.cat([wb, wa])
    out["motion_weights"] = w / w.sum()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--add", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--match-weight-of",
        default=None,
        help="Base-lib filename substring; every added clip gets that group's "
        "per-clip weight (renormalized). Omit to keep the added lib's own "
        "weights (rescaled into the merged total).",
    )
    args = p.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    add = torch.load(args.add, map_location="cpu", weights_only=False)
    merged = merge(base, add, args.match_weight_of)

    n_b, n_a = len(base["motion_files"]), len(add["motion_files"])
    w = merged["motion_weights"]
    print(f"merged: {n_b} + {n_a} = {len(merged['motion_files'])} motions, "
          f"{merged['gts'].shape[0]} frames")
    print(f"added-clips sampling mass: {float(w[n_b:].sum()) * 100:.1f}%")
    if args.output.exists():
        bak = args.output.with_suffix(args.output.suffix + ".pre_merge_bak")
        args.output.replace(bak)
        print(f"backed up existing output -> {bak}")
    torch.save(merged, args.output)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
