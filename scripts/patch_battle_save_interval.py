#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Set save_last_checkpoint_every on a run's pickled resolved config.

RESUME reads resolved_configs.pt, never the experiment file, so this is the only
way to change the cadence for an existing run without starting a new one.

Verification reloads the file from disk and reports the value the trainer will
actually read -- re-reading an in-memory object proves nothing.

    python scripts/patch_battle_save_interval.py <run_dir> --epochs 240
"""

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--epochs", type=int, required=True,
                    help="write last.ckpt every N epochs")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    run = Path(args.run_dir)
    targets = [p for p in (run / "resolved_configs.pt",
                           run / "resolved_configs_inference.pt") if p.exists()]
    if not targets:
        raise SystemExit(f"no resolved_configs*.pt under {run}")

    for p in targets:
        cfg = torch.load(p, map_location="cpu", weights_only=False)
        before = getattr(cfg["agent"], "save_last_checkpoint_every", None)
        if not args.apply:
            print(f"{p.name}: save_last_checkpoint_every {before} -> {args.epochs} (dry run)")
            continue
        cfg["agent"].save_last_checkpoint_every = args.epochs
        torch.save(cfg, p)

        reloaded = torch.load(p, map_location="cpu", weights_only=False)
        got = getattr(reloaded["agent"], "save_last_checkpoint_every", None)
        assert got == args.epochs, f"patch did not persist: {got!r}"
        print(f"{p.name}: save_last_checkpoint_every {before} -> {got} (verified on reload)")


if __name__ == "__main__":
    main()
