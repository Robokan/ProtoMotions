# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Build T800 tracker lib: full corpus + combat/reallusion, no drunken.

Mirrors Atlas ``atlas_tracker_stage4.pt`` composition (uniform per-clip
weights) but drops drunken. Swaps in retuned reallusion clips from
``data/t800_reallusion.pt`` when present.
"""

from pathlib import Path

import torch

from data.scripts.merge_motion_libs import merge, subset


def main() -> None:
    stage = torch.load(
        "data/t800_tracker_stage1.pt", map_location="cpu", weights_only=False
    )
    reallusion = torch.load(
        "data/t800_reallusion.pt", map_location="cpu", weights_only=False
    )
    drunken = torch.load(
        "data/t800_drunken.pt", map_location="cpu", weights_only=False
    )
    # Atlas drunken set is the authoritative 20-name list (t800_drunken is filtered)
    atlas_stage1 = torch.load(
        "data/atlas_combat_stage1.pt", map_location="cpu", weights_only=False
    )
    atlas_drunk = {
        Path(f).name
        for f in atlas_stage1["motion_files"]
        if "drunken" in str(f)
    }
    drunk_names = {Path(f).name for f in drunken["motion_files"]} | atlas_drunk
    r_names = {Path(f).name for f in reallusion["motion_files"]}

    keep = [
        i
        for i, f in enumerate(stage["motion_files"])
        if Path(f).name not in drunk_names and Path(f).name not in r_names
    ]
    base = subset(stage, keep)
    print(
        f"stage={len(stage['motion_files'])} "
        f"drop_drunken={len(drunk_names)} drop_old_reallusion={len(r_names)} "
        f"keep={len(keep)}"
    )

    # Uniform weights like Atlas stage packs
    base["motion_weights"] = torch.ones(len(base["motion_files"]))
    reallusion = dict(reallusion)
    reallusion["motion_weights"] = torch.ones(len(reallusion["motion_files"]))

    out = merge(base, reallusion, match_weight_of=None)
    n = len(out["motion_files"])
    out["motion_weights"] = torch.ones(n) / n  # Atlas stage4 style (sum≈1)

    # Sanity: no drunken left
    leftover = [Path(f).name for f in out["motion_files"] if Path(f).name in drunk_names]
    if leftover:
        raise SystemExit(f"drunken still present: {leftover}")

    out_path = Path("data/t800_tracker_nodrunken.pt")
    torch.save(out, out_path)

    files = [Path(f).name for f in out["motion_files"]]
    r_mask = torch.tensor([name in r_names for name in files])
    w = out["motion_weights"].float()
    print(f"saved {out_path} n={n}")
    print(
        f"reallusion mass={float(w[r_mask].sum() / w.sum()):.4f} n={int(r_mask.sum())}"
    )
    print(
        f"other mass={float(w[~r_mask].sum() / w.sum()):.4f} n={int((~r_mask).sum())}"
    )
    print(f"weight min/max={float(w.min()):.8f}/{float(w.max()):.8f}")


if __name__ == "__main__":
    main()
