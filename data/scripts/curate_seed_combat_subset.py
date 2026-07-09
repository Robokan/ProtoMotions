# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Curate a combat-emphasized subset of BONES-SEED for GPC prior training.

Classifies every clip in the SEED metadata into three tiers and stages the
selected BVH files (symlinks) into per-tier directories, so the converted
``.motion`` files carry ``/combat/``, ``/adjacent/`` or ``/breadth/`` in their
paths for group weighting:

- **combat**: strict matches — martial arts category, shadow boxing,
  punches, sparring, strikes-as-attacks, combat-tagged clips.
- **adjacent**: fight-relevant support skills — falls, get-ups, stunts,
  dodges/avoids, fighting stances, ducking — the material that makes
  knockdown recovery and defensive movement exist in the token vocabulary.
- **breadth**: a stratified sample of everything else (locomotion, dances,
  gestures...), capped per category so 33K neutral-walk clips can't dominate.

Usage:
    python data/scripts/curate_seed_combat_subset.py \\
        --seed-root /workspace/sparkpack/bones-seed \\
        --staging-dir /workspace/sparkpack/bones-seed/staging \\
        --breadth-total 8000
"""

import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path

COMBAT_PATTERNS = [
    r"shadow.?boxing",
    r"\bboxing\b",
    r"\bpunch",
    r"\bspar(ring)?\b",
    r"\buppercut",
    r"\bjab\b",
    r"kickboxing",
    r"\bkarate|kung.?fu|taekwondo|martial",
    r"\bcombat\b",
    r"(front|side|high|spinning|round(house)?)\s+kick",
    r"kick.*(face|head|opponent|attacker)",
    r"\bheadbutt",
    r"elbow strike|knee strike",
    r"\bbrawl|wrestl|grappl",
    r"defend.*(attack|punch)|block.*(punch|attack|strike)",
]

ADJACENT_PATTERNS = [
    r"\bfall(s|ing|en)?\b",
    r"get(ting)?.?up\b",
    r"stand(ing)?.?up (from|off)",
    r"rise (from|off)",
    r"knock(ed)?\b",
    r"\bstumbl|\btrip(s|ped|ping)?\b|\bcollaps",
    r"lying (on|down)|\bprone\b|\bsupine\b",
    r"roll(ing)? (on|over|across).*(ground|floor)",
    r"\bdodg|\bavoid|\bduck(s|ing)?\b|\bevad",
    r"fighting stance",
    r"\bcrouch",
]

TEXT_FIELDS = [
    "move_name",
    "content_short_description",
    "content_short_description_2",
    "content_type_of_movement",
    "content_uniform_style",
    "content_technical_description",
    "content_natural_desc_1",
]

# Ball-sport kicks etc. that the kick patterns must not capture
COMBAT_EXCLUDE = re.compile(r"\bball\b|soccer|football|dancecard", re.I)


def classify(row) -> str:
    text = " ".join(row.get(k) or "" for k in TEXT_FIELDS)
    category = row.get("category") or ""

    if category == "Martial Arts":
        return "combat"
    # Dance choreography is never combat, even when described with fight
    # vocabulary ("kick it", punch-styled moves): letting those through
    # concentrated ~14% of prior-training samples on hip-hop moves.
    if row.get("package") != "Dances" and not COMBAT_EXCLUDE.search(text):
        for pattern in COMBAT_PATTERNS:
            if re.search(pattern, text, re.I):
                return "combat"
    if category == "Stunts":
        return "adjacent"
    for pattern in ADJACENT_PATTERNS:
        if re.search(pattern, text, re.I):
            return "adjacent"
    return "breadth"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed-root", required=True, help="bones-seed directory")
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--breadth-total", type=int, default=8000)
    parser.add_argument(
        "--breadth-category-cap",
        type=float,
        default=0.15,
        help="Max fraction of breadth-total any one (package, category) can take",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_root = Path(args.seed_root)
    staging = Path(args.staging_dir)
    rng = random.Random(args.seed)

    tiers = {"combat": [], "adjacent": [], "breadth": []}
    missing = 0
    with open(seed_root / "metadata/seed_metadata_v004.csv") as f:
        for row in csv.DictReader(f):
            rel = row.get("move_soma_uniform_path")
            if not rel:
                continue
            bvh = seed_root / rel
            if not bvh.exists():
                missing += 1
                continue
            tiers[classify(row)].append((bvh, row))

    # Stratified breadth sample with a per-category cap
    by_cat = collections.defaultdict(list)
    for item in tiers["breadth"]:
        by_cat[(item[1]["package"], item[1]["category"])].append(item)
    cap = max(1, int(args.breadth_total * args.breadth_category_cap))
    sampled = []
    for cat, items in sorted(by_cat.items()):
        rng.shuffle(items)
        sampled.extend(items[:cap])
    rng.shuffle(sampled)
    tiers["breadth"] = sampled[: args.breadth_total]

    summary = {"missing_bvh": missing}
    for tier, items in tiers.items():
        tier_dir = staging / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        for bvh, _row in items:
            link = tier_dir / bvh.name
            if not link.exists():
                link.symlink_to(bvh.resolve())
        summary[tier] = len(items)
        cats = collections.Counter(
            f"{r['package']}/{r['category']}" for _, r in items
        )
        summary[f"{tier}_categories"] = dict(cats.most_common(8))

    (staging / "curation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
