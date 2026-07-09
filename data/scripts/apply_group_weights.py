# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Assign sampling-mass fractions to path-matched clip groups in a MotionLib.

Multi-group generalization of ``weight_combat_motions.py``: each group is a
path substring plus a target fraction of the total sampling mass; clips
matching no group share the remainder. Relative weights *within* every group
are preserved.

Example (the combat-emphasized prior mix)::

    python data/scripts/apply_group_weights.py \\
        --motion-lib data/soma_seed_curated.pt \\
        --group /combat/=0.35 --group /adjacent/=0.25
    # /breadth/ (unmatched) keeps the remaining 0.40
"""

import argparse
from pathlib import Path

import torch


def apply_group_weights(data: dict, groups: dict) -> dict:
    """Mutate data['motion_weights'] so each pattern gets its mass fraction."""
    if "motion_files" not in data or "motion_weights" not in data:
        raise ValueError(
            "Packed motion lib must contain 'motion_files' and 'motion_weights'"
        )
    total_fraction = sum(groups.values())
    if not 0.0 < total_fraction < 1.0:
        raise ValueError(
            f"Group fractions must sum to within (0, 1); got {total_fraction} "
            "(unmatched clips receive the remainder)"
        )

    files = [str(f) for f in data["motion_files"]]
    weights = data["motion_weights"].clone().float()
    num = len(files)
    if num != weights.shape[0]:
        raise ValueError("motion_files / motion_weights length mismatch")

    masks = {}
    assigned = torch.zeros(num, dtype=torch.bool)
    for pattern in groups:
        mask = torch.tensor([pattern in f for f in files], dtype=torch.bool)
        mask &= ~assigned  # first pattern wins on overlap
        if not mask.any():
            raise ValueError(f"No clips match group pattern '{pattern}'")
        masks[pattern] = mask
        assigned |= mask
    if not (~assigned).any():
        raise ValueError("Every clip matched a group; nothing left for remainder")

    remainder_fraction = 1.0 - total_fraction
    stats = {}
    for pattern, fraction in groups.items():
        mask = masks[pattern]
        group_mass = weights[mask].sum()
        weights[mask] *= fraction / group_mass
        stats[pattern] = {"clips": int(mask.sum()), "fraction": fraction}
    rem_mass = weights[~assigned].sum()
    weights[~assigned] *= remainder_fraction / rem_mass
    stats["<remainder>"] = {
        "clips": int((~assigned).sum()),
        "fraction": remainder_fraction,
    }

    # Normalize for readability (multinomial is scale-invariant)
    weights = weights / weights.sum() * num
    data["motion_weights"] = weights
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--motion-lib", required=True)
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="PATTERN=FRACTION",
        help="Path substring and its sampling-mass fraction; repeatable",
    )
    parser.add_argument("--output", default=None, help="Default: in place")
    args = parser.parse_args()

    groups = {}
    for spec in args.group:
        pattern, _, fraction = spec.partition("=")
        groups[pattern] = float(fraction)

    data = torch.load(args.motion_lib, map_location="cpu", weights_only=False)
    stats = apply_group_weights(data, groups)
    output = Path(args.output or args.motion_lib)
    torch.save(data, output)
    for pattern, info in stats.items():
        print(f"  {pattern:15s} {info['clips']:6d} clips -> {info['fraction']:.0%} of samples")
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
