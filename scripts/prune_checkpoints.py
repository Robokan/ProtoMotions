#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Reclaim disk by deleting intermediate epoch_*.ckpt snapshots.

Every run writes epoch_<N>.ckpt every save_epoch_checkpoint_every epochs on
top of last.ckpt. Those intermediates are only useful for retreating to an
earlier point in a run that later degraded; once a run is finished or
abandoned they are dead weight (measured 2026-08-25: 138 GB of 179 GB in
results/ was intermediate epoch checkpoints).

SAFETY -- this script never touches:
  * last.ckpt                (resume point, and what --llc-checkpoint refs use)
  * the NEWEST epoch_*.ckpt  (kept as a milestone per run)
  * anything under a league/ directory (battle league members reference these
    by path; deleting them breaks league restore)
  * env_*.ckpt / resolved_configs.*

Dry run by default. Add --apply to actually delete.

    python scripts/prune_checkpoints.py                 # report only
    python scripts/prune_checkpoints.py --apply         # delete
    python scripts/prune_checkpoints.py --keep 3        # keep newest 3 each
    python scripts/prune_checkpoints.py --exclude atlas_v17_physx
"""

import argparse
import os
import re
from pathlib import Path

EPOCH_RE = re.compile(r"^epoch_(\d+)\.ckpt$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="results directory")
    ap.add_argument("--keep", type=int, default=1,
                    help="how many newest epoch checkpoints to keep per run")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="run names to leave completely alone")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default is a dry run)")
    args = ap.parse_args()

    root = Path(args.results)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")

    total_bytes = 0
    total_files = 0
    rows = []

    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        if run.name in args.exclude:
            continue
        # Collect epoch checkpoints anywhere under the run EXCEPT league dirs.
        found = []
        for path in run.rglob("epoch_*.ckpt"):
            if "league" in path.parts:
                continue
            m = EPOCH_RE.match(path.name)
            if m:
                found.append((int(m.group(1)), path))
        if len(found) <= args.keep:
            continue
        found.sort()                      # ascending by epoch number
        doomed = found[: -args.keep] if args.keep > 0 else found
        size = sum(p.stat().st_size for _, p in doomed)
        rows.append((size, run.name, len(doomed), found[-1][0]))
        total_bytes += size
        total_files += len(doomed)
        if args.apply:
            for _, p in doomed:
                try:
                    p.unlink()
                except OSError as exc:
                    print(f"  ! could not delete {p}: {exc}")

    rows.sort(reverse=True)
    print(f"{'run':44s} {'delete':>7s} {'keep@':>9s} {'frees':>9s}")
    for size, name, n, newest in rows:
        print(f"{name:44s} {n:7d} {newest:9d} {size/2**30:8.2f}G")
    verb = "DELETED" if args.apply else "would delete (dry run)"
    print(f"\n{verb}: {total_files} files, {total_bytes/2**30:.1f} GB")
    if not args.apply:
        print("re-run with --apply to actually remove them")


if __name__ == "__main__":
    main()
