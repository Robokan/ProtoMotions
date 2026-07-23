# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Curate a combat-focused prior/tracker corpus from a full motion lib.

Selects (by clip name, robot-agnostic):
  - SEED combat tier (from the staging dir created by
    curate_seed_combat_subset.py) — every clip,
  - Reallusion combat clips — every clip (upweighted),
  - fight-support: getups, falls/knockdowns, dodges, rolls, stumble/balance
    recoveries, fighting stances/ducks,
  - core locomotion (walk/jog/run/turn/strafe/sidestep/idle), capped per
    family so 3k neutral walks can't dominate.

Excludes anything in --exclude-list (per-robot unconvertible clips).

Usage (host or container):
    python data/scripts/curate_prior_corpus.py \\
        --lib data/atlas_tracker_stage5.pt \\
        --reallusion-lib data/atlas_reallusion_combat.pt \\
        --exclude-list data/atlas_unconvertible_motions.txt \\
        --out data/atlas_prior_corpus.pt
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_motion_libs import subset  # noqa: E402

COMBAT_STAGING = Path("/home/evaughan/sparkpack/bones-seed/staging/combat")

SUPPORT_PATTERNS = [
    r"come_up", r"get_?up", r"stand_?up", r"lying", r"knock",
    r"\bfall", r"stumbl", r"\btrip", r"\bslip", r"balance",
    r"dodge", r"duck", r"avoid_bump",
    r"\broll\b", r"side_roll", r"forward_roll", r"shoulder_roll",
    r"crouch",  # low guards / level changes — fight-relevant
    r"spar", r"box", r"punch", r"kick", r"strike", r"block", r"attack",
    r"fight", r"guard",
]
LOCOMOTION_PATTERNS = [
    r"^walk", r"^jog", r"^run", r"^turn", r"^strafe", r"^side_?step",
    r"^idle", r"^stand_",
]
LOCOMOTION_CAP = 40  # per family stem
SUPPORT_CAP = 25  # per family stem — SEED has 200+ takes of some crouches


def family(n: str) -> str:
    return re.split(r"_\d|__A", n)[0][:24]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--lib", required=True)
    p.add_argument("--reallusion-lib", required=True)
    p.add_argument("--exclude-list", default=None)
    p.add_argument(
        "--exclude-families",
        default=None,
        help="Comma-separated substrings; any clip whose name contains one is "
        "dropped (e.g. 'crouch,come_up' — families that retarget badly on a "
        "given robot; T800 crouches pop, come_up steps onto phantom props).",
    )
    p.add_argument("--reallusion-weight-mult", type=float, default=4.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    combat_tier = {f.stem for f in COMBAT_STAGING.rglob("*.bvh")}
    real_names = {
        str(f).split("/")[-1].replace(".motion", "")
        for f in torch.load(args.reallusion_lib, map_location="cpu",
                            weights_only=False)["motion_files"]
    }
    exclude = set()
    if args.exclude_list:
        exclude = {l.strip() for l in open(args.exclude_list)
                   if l.strip() and not l.startswith("#")}

    d = torch.load(args.lib, map_location="cpu", weights_only=False)
    names = [str(f).split("/")[-1].replace(".motion", "") for f in d["motion_files"]]

    sup_re = re.compile("|".join(SUPPORT_PATTERNS))
    loc_re = re.compile("|".join(LOCOMOTION_PATTERNS))
    keep, reason = [], Counter()
    loco_per_family = Counter()
    sup_per_family = Counter()
    for i, n in enumerate(names):
        if n in exclude:
            reason["excluded"] += 1
            continue
        if args.exclude_families and any(
            fam in n.lower() for fam in args.exclude_families.split(",")
        ):
            reason["excluded_family"] += 1
            continue
        low = n.lower()
        if n in real_names:
            keep.append(i); reason["reallusion"] += 1
        elif n in combat_tier:
            keep.append(i); reason["seed_combat"] += 1
        elif sup_re.search(low):
            fam = family(n)
            if sup_per_family[fam] < SUPPORT_CAP:
                sup_per_family[fam] += 1
                keep.append(i); reason["fight_support"] += 1
            else:
                reason["support_capped"] += 1
        elif loc_re.search(low):
            fam = family(n)
            if loco_per_family[fam] < LOCOMOTION_CAP:
                loco_per_family[fam] += 1
                keep.append(i); reason["locomotion"] += 1
            else:
                reason["locomotion_capped"] += 1
        else:
            reason["dropped"] += 1

    out = subset(d, keep)
    kept_names = [names[i] for i in keep]
    w = out["motion_weights"].float()
    mean_w = float(w.mean())
    for i, n in enumerate(kept_names):
        if n in real_names:
            w[i] = args.reallusion_weight_mult * mean_w
    out["motion_weights"] = w / w.sum()
    torch.save(out, args.out)

    v = torch.load(args.out, map_location="cpu", weights_only=False)
    assert v["gts"].shape[0] == int(v["motion_num_frames"].long().sum())
    wv = v["motion_weights"].float()
    rmass = sum(float(wv[i]) for i, n in enumerate(kept_names) if n in real_names)
    print(f"selection: {dict(reason)}")
    print(f"curated: {len(kept_names)} motions, {v['gts'].shape[0]} frames, "
          f"reallusion mass {rmass*100:.1f}%  -> {args.out}")


if __name__ == "__main__":
    main()
