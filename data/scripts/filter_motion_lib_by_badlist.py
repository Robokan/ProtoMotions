# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Rebuild a MotionLib directory of .motion files excluding a bad-name list, then pack."""

import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion-dir", type=Path, required=True)
    p.add_argument("--bad-list", type=Path, required=True)
    p.add_argument("--good-dir", type=Path, required=True)
    p.add_argument("--output-pt", type=Path, required=True)
    p.add_argument("--min-keep", type=int, default=100)
    args = p.parse_args()

    bad = set()
    if args.bad_list.exists():
        bad = {l.strip() for l in args.bad_list.read_text().splitlines() if l.strip()}

    args.good_dir.mkdir(parents=True, exist_ok=True)
    for old in args.good_dir.glob("*.motion"):
        old.unlink()

    kept = 0
    for src in sorted(args.motion_dir.glob("*.motion")):
        if src.stem in bad:
            continue
        dst = args.good_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        kept += 1

    print(f"kept {kept} / skipped {len(bad)} bad names -> {args.good_dir}")
    if kept < args.min_keep:
        raise SystemExit(f"only {kept} clips kept; refusing to pack")

    from protomotions.components.motion_lib import MotionLib, MotionLibConfig

    lib = MotionLib(
        config=MotionLibConfig(motion_file=str(args.good_dir)),
        device="cpu",
    )
    lib.save_to_file(str(args.output_pt))
    print(f"saved {args.output_pt} motions={len(lib.motion_files)}")


if __name__ == "__main__":
    main()
