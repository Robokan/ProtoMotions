# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Bias a packed MotionLib's sampling weights toward combat clips.

SOMA_GPC_COMBAT_PLAN Phase 3 trains the GPC prior on a mixed
combat + BONES-SEED library "with combat upweighted": the frozen tracker
already covers every skill as tokens, so the prior's clip-sampling
distribution is what shapes its behavioral preferences. Training and reset
sampling draw clips via ``multinomial(motion_weights)``, so boosting the
combat clips' weights directly increases how much fighting the prior sees.

Clips are matched by a substring of their source path (default ``/combat/``,
matching the layout ``prepare_soma_combat_dataset.sh`` produces).

Two ways to specify the emphasis:

``--combat-fraction 0.5``
    Rescale so combat clips receive exactly this fraction of the total
    sampling mass (recommended: interpretable and independent of how many
    clips each side has -- 142K SEED clips vs a few hundred combat clips
    would otherwise drown the combat set).

``--boost 10``
    Simply multiply combat clip weights by a constant.

Example::

    python data/scripts/weight_combat_motions.py \\
        --motion-lib data/soma_combat_seed.pt \\
        --combat-fraction 0.5 \\
        --output data/soma_combat_seed_weighted.pt
"""

import argparse
from pathlib import Path

import torch


def apply_combat_weights(
    data: dict,
    pattern: str,
    combat_fraction: float = None,
    boost: float = None,
) -> dict:
    """Return stats after mutating ``data['motion_weights']`` in place."""
    if "motion_files" not in data or "motion_weights" not in data:
        raise ValueError(
            "Packed motion lib must contain 'motion_files' and 'motion_weights' "
            f"(has: {sorted(data.keys())})"
        )
    files = list(data["motion_files"])
    weights = data["motion_weights"].clone().float()
    if len(files) != weights.shape[0]:
        raise ValueError(
            f"motion_files ({len(files)}) and motion_weights "
            f"({weights.shape[0]}) length mismatch"
        )

    combat_mask = torch.tensor(
        [pattern in str(f) for f in files], dtype=torch.bool
    )
    num_combat = int(combat_mask.sum())
    if num_combat == 0:
        raise ValueError(
            f"No clips match pattern '{pattern}'. Sample paths: {files[:3]}"
        )
    if num_combat == len(files):
        raise ValueError(
            f"Every clip matches pattern '{pattern}' — nothing to reweight "
            "against. Use the combined (seed + combat) library."
        )

    if (combat_fraction is None) == (boost is None):
        raise ValueError("Specify exactly one of --combat-fraction / --boost")

    if boost is not None:
        weights[combat_mask] *= boost
    else:
        if not 0.0 < combat_fraction < 1.0:
            raise ValueError("--combat-fraction must be in (0, 1)")
        combat_mass = weights[combat_mask].sum()
        other_mass = weights[~combat_mask].sum()
        if combat_mass <= 0 or other_mass <= 0:
            raise ValueError("Both clip groups need positive total weight")
        # Scale combat mass so combat / (combat + other) == combat_fraction
        scale = (combat_fraction / (1.0 - combat_fraction)) * (
            other_mass / combat_mass
        )
        weights[combat_mask] *= scale

    # Normalize for readability (multinomial is scale-invariant)
    weights = weights / weights.sum() * len(files)
    data["motion_weights"] = weights

    combat_share = float(weights[combat_mask].sum() / weights.sum())
    return {
        "num_clips": len(files),
        "num_combat": num_combat,
        "combat_sampling_share": combat_share,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--motion-lib", required=True, help="Packed .pt library")
    parser.add_argument(
        "--pattern",
        default="/combat/",
        help="Path substring identifying combat clips (default: /combat/)",
    )
    parser.add_argument(
        "--combat-fraction",
        type=float,
        default=None,
        help="Fraction of total sampling mass for combat clips (recommended)",
    )
    parser.add_argument(
        "--boost", type=float, default=None, help="Plain multiplier instead"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pt (default: overwrite --motion-lib in place)",
    )
    args = parser.parse_args()

    data = torch.load(args.motion_lib, map_location="cpu", weights_only=False)
    stats = apply_combat_weights(
        data,
        pattern=args.pattern,
        combat_fraction=args.combat_fraction,
        boost=args.boost,
    )

    output = Path(args.output or args.motion_lib)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    print(
        f"{stats['num_combat']}/{stats['num_clips']} clips matched "
        f"'{args.pattern}'; combat now receives "
        f"{stats['combat_sampling_share']:.1%} of sampling mass -> {output}"
    )


if __name__ == "__main__":
    main()
