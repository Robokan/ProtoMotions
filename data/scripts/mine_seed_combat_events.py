# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Mine combat moments out of BONES-SEED's temporal event labels.

Many SEED captures are long multi-activity clips whose clip-level description
never mentions fighting, but whose per-event annotations do ("The fighter
performs a straight jab...", "shadowboxing in stance"). This script finds
those events with a strict combat filter and carves each time window out of
the converted ``.motion`` file into a standalone sub-clip under the
``combat/`` tier, growing the combat pool beyond what clip-level curation
can see.

Two-pass usage (sources that aren't converted yet get staged for the
standard converter first):

    # Pass 1: reports hits; symlinks unconverted sources into --stage-dir
    python data/scripts/mine_seed_combat_events.py --seed-root ... --trim

    # Convert staged sources (if any were reported), then rerun with --trim
    python data/scripts/convert_soma23_bvh_to_proto.py \\
        --input-dir <stage-dir> --output-dir <motions-root>/event_sources ...
"""

import argparse
import csv
import json
import re
from pathlib import Path

import torch

STRONG_COMBAT = re.compile(
    r"\b(punch(es|ing|ed)?\b|jab(s|bing)?\b|uppercut|boxing|boxes (the air|an imaginary)|"
    r"spar(s|ring)\b|shadow.?box|kickbox|martial|karate|"
    r"fight(ing)? (stance|pose|position|move)|throws? (a|an|some) (punch|jab|hook|uppercut|kick)|"
    r"kicks? (in the air|the air|forward and punches|at an imaginary|towards? (a |an )?(person|opponent|imaginary))|"
    r"(high|roundhouse|spinning) kick|"
    r"blocks? (a |an |the )?(punch|kick|strike|blow|attack)|"
    r"dodges? (a |an |the )?(punch|kick|strike|blow|attack))",
    re.I,
)
EXCLUDE = re.compile(
    r"ball|soccer|trash|obstacle|can\b|door|box(es)? (on|off|up)|cartwheel|"
    r"lying|resting|scissor|swim",
    re.I,
)

FRAME_KEYS = [
    "dof_pos",
    "dof_vel",
    "rigid_body_pos",
    "rigid_body_rot",
    "rigid_body_vel",
    "rigid_body_ang_vel",
    "rigid_body_contacts",
    "local_rigid_body_rot",
]


def scan_events(seed_root: Path, min_dur: float, max_dur: float):
    meta = {}
    with open(seed_root / "metadata/seed_metadata_v004.csv") as f:
        for row in csv.DictReader(f):
            meta[row["filename"]] = row["package"]

    hits = []
    with open(seed_root / "metadata/seed_metadata_v002_temporal_labels.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if meta.get(rec["filename"]) == "Dances":
                continue
            for ev in rec["events"]:
                desc = ev["description"]
                if STRONG_COMBAT.search(desc) and not EXCLUDE.search(desc):
                    dur = ev["end_time"] - ev["start_time"]
                    if min_dur <= dur <= max_dur:
                        hits.append(
                            (rec["filename"], ev["start_time"], ev["end_time"], desc)
                        )
    return hits


def find_motion(motions_root: Path, name: str):
    for tier in ("combat", "adjacent", "breadth", "event_sources"):
        candidate = motions_root / tier / f"{name}.motion"
        if candidate.exists():
            return candidate, tier
    return None, None


def trim_motion(
    src: Path,
    dst: Path,
    start: float,
    end: float,
    pad: float,
    max_velocity: float = 20.0,
    min_height: float = -0.05,
) -> bool:
    """Cut the window and quality-check it.

    Long multi-activity captures often fail the converter's whole-clip
    filter because of one bad stretch elsewhere; the window itself is what
    must be clean, so the velocity/underground checks run on the slice.
    """
    data = torch.load(src, map_location="cpu", weights_only=False)
    fps = float(data["fps"])
    num_frames = data["dof_pos"].shape[0]
    lo = max(0, int((start - pad) * fps))
    hi = min(num_frames, int((end + pad) * fps) + 1)
    if hi - lo < int(1.0 * fps):  # keep at least a second
        return False

    window_pos = data["rigid_body_pos"][lo:hi]
    window_vel = data["rigid_body_vel"][lo:hi]
    if float(window_vel.norm(dim=-1).max()) > max_velocity:
        return False
    if float(window_pos[..., 2].min()) < min_height:
        return False

    for key in FRAME_KEYS:
        if key in data:
            data[key] = data[key][lo:hi].clone()
    torch.save(data, dst)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed-root", required=True)
    parser.add_argument("--motions-root", required=True)
    parser.add_argument("--stage-dir", default=None, help="Where to symlink unconverted sources")
    parser.add_argument("--pad-seconds", type=float, default=0.3)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--trim", action="store_true", help="Write trimmed sub-clips")
    args = parser.parse_args()

    seed_root = Path(args.seed_root)
    motions_root = Path(args.motions_root)
    combat_dir = motions_root / "combat"

    hits = scan_events(seed_root, args.min_duration, args.max_duration)
    print(f"combat events: {len(hits)} across {len({h[0] for h in hits})} clips")

    missing, trimmed, skipped_combat_tier, too_short = [], 0, 0, 0
    for idx, (name, start, end, _desc) in enumerate(hits):
        src, tier = find_motion(motions_root, name)
        if src is None:
            missing.append(name)
            continue
        if tier == "combat":
            skipped_combat_tier += 1  # whole clip already fully combat-weighted
            continue
        if args.trim:
            dst = combat_dir / f"{name}__evt{int(start * 10):04d}.motion"
            if dst.exists():
                continue
            if trim_motion(src, dst, start, end, args.pad_seconds):
                trimmed += 1
            else:
                too_short += 1

    missing = sorted(set(missing))
    print(
        f"trimmed: {trimmed} | already-combat sources skipped: {skipped_combat_tier} | "
        f"too short: {too_short} | unconverted sources: {len(missing)}"
    )
    if missing and args.stage_dir:
        stage = Path(args.stage_dir)
        stage.mkdir(parents=True, exist_ok=True)
        meta_paths = {}
        with open(seed_root / "metadata/seed_metadata_v004.csv") as f:
            for row in csv.DictReader(f):
                meta_paths[row["filename"]] = row["move_soma_uniform_path"]
        staged = 0
        for name in missing:
            rel = meta_paths.get(name)
            if not rel:
                continue
            bvh = seed_root / rel
            link = stage / bvh.name
            if bvh.exists() and not link.exists():
                link.symlink_to(bvh.resolve())
                staged += 1
        print(f"staged {staged} BVH sources into {stage} — convert them, then rerun")


if __name__ == "__main__":
    main()
