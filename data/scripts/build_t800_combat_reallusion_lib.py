# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Build T800 combat+reallusion training lib (no drunken), Atlas-uniform weights.

Atlas ``atlas_combat_viewer.pt`` is 184 clips = 102 combat-only + 82 reallusion
with uniform per-clip weights (all 1.0). This rebuilds the same composition for
T800, swapping in the retuned reallusion pack.
"""

from pathlib import Path

import torch

from data.scripts.merge_motion_libs import merge, subset


def main() -> None:
    viewer = torch.load(
        "data/t800_combat_viewer.pt", map_location="cpu", weights_only=False
    )
    reallusion = torch.load(
        "data/t800_reallusion.pt", map_location="cpu", weights_only=False
    )

    r_names = {Path(f).name for f in reallusion["motion_files"]}
    keep = [
        i
        for i, f in enumerate(viewer["motion_files"])
        if Path(f).name not in r_names
    ]
    n_viewer = len(viewer["motion_files"])
    n_r = len(reallusion["motion_files"])
    print(f"viewer={n_viewer} keep_non_reallusion={len(keep)} reallusion={n_r}")
    if len(keep) != 102 or n_r != 82:
        raise SystemExit(f"unexpected counts keep={len(keep)} reallusion={n_r}")

    base = subset(viewer, keep)
    base["motion_weights"] = torch.ones(len(base["motion_files"]))
    reallusion = dict(reallusion)
    reallusion["motion_weights"] = torch.ones(len(reallusion["motion_files"]))

    out = merge(base, reallusion, match_weight_of=None)
    n = len(out["motion_files"])
    # Match atlas_combat_viewer: uniform ones (not sum-normalized)
    out["motion_weights"] = torch.ones(n)

    out_path = Path("data/t800_combat_reallusion.pt")
    torch.save(out, out_path)

    w = out["motion_weights"].float()
    files = [Path(f).name for f in out["motion_files"]]
    r_mask = torch.tensor([name in r_names for name in files])
    print(f"saved {out_path} n={n}")
    print(
        f"reallusion mass={float(w[r_mask].sum() / w.sum()):.4f} n={int(r_mask.sum())}"
    )
    print(
        f"combat mass={float(w[~r_mask].sum() / w.sum()):.4f} n={int((~r_mask).sum())}"
    )

    av = torch.load(
        "data/atlas_combat_viewer.pt", map_location="cpu", weights_only=False
    )
    ar = torch.load(
        "data/atlas_reallusion_combat.pt", map_location="cpu", weights_only=False
    )
    ar_names = {Path(f).name for f in ar["motion_files"]}
    av_files = [Path(f).name for f in av["motion_files"]]
    print(f"name parity with atlas_combat_viewer: {set(files) == set(av_files)}")


if __name__ == "__main__":
    main()
