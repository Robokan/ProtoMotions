#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Set battle reward weights on a run's pickled resolved config.

RESUME reads resolved_configs.pt, not the experiment file, so an existing run's
weights can only be changed here.

TWO THINGS THIS GETS RIGHT, both learned the hard way:

1. Weights live in ``component.static_params['weight']``. Setting
   ``component.weight`` creates a new attribute that MdpComponent never reads,
   so the patch silently no-ops (cost a weekend, 2026-08).

2. Verification reloads from disk AND diffs against
   ``default_battle_reward_components``. Confirming a value is merely *live*
   catches typos but not wrong numbers -- battle_win sat at 50 against a
   documented default of 500 for a 7-day run because "it's in the config" was
   the only check performed (2026-09-02).

    python scripts/patch_battle_reward_weights.py <run_dir>                 # report
    python scripts/patch_battle_reward_weights.py <run_dir> --sync-defaults --apply
    python scripts/patch_battle_reward_weights.py <run_dir> --set battle_win=500 --apply
"""

import argparse
from pathlib import Path

import torch

from protomotions.envs.battle.factories import default_battle_reward_components


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE")
    ap.add_argument("--sync-defaults", action="store_true",
                    help="set every weight to its factory default")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    overrides = {}
    for item in args.set:
        k, v = item.split("=", 1)
        overrides[k] = float(v)

    defaults = {k: v.static_params.get("weight")
                for k, v in default_battle_reward_components(1.0).items()}

    run = Path(args.run_dir)
    targets = [p for p in (run / "resolved_configs.pt",
                           run / "resolved_configs_inference.pt") if p.exists()]
    if not targets:
        raise SystemExit(f"no resolved_configs*.pt under {run}")

    for p in targets:
        cfg = torch.load(p, map_location="cpu", weights_only=False)
        comps = cfg["env"].reward_components
        changes = []
        for name, comp in comps.items():
            cur = comp.static_params.get("weight")
            want = overrides.get(name)
            if want is None and args.sync_defaults:
                want = defaults.get(name)
            if want is None or cur == want:
                continue
            changes.append((name, cur, want))
            if args.apply:
                comp.static_params["weight"] = want

        print(f"\n{p.name}:")
        if not changes:
            print("  no changes needed")
        for name, cur, want in changes:
            print(f"  {name:26s} {cur:9.1f} -> {want:9.1f}"
                  + ("" if args.apply else "   (dry run)"))
        if not args.apply:
            continue

        torch.save(cfg, p)

        # Reload from disk and audit against the factory defaults.
        chk = torch.load(p, map_location="cpu", weights_only=False)
        live = chk["env"].reward_components
        print("  verified on reload:")
        bad = 0
        for name in sorted(defaults):
            if name not in live:
                print(f"    {name:26s} MISSING from run"); bad += 1
                continue
            got = live[name].static_params.get("weight")
            dflt = defaults[name]
            mark = "" if got == dflt else f"  <-- differs from default {dflt:.1f}"
            if got != dflt:
                bad += 1
            print(f"    {name:26s} {got:9.1f}{mark}")
        # A deliberate divergence is fine, but it must be visible.
        print(f"  {bad} weight(s) differ from factory defaults"
              + (" -- confirm each is intentional" if bad else ""))


if __name__ == "__main__":
    main()
