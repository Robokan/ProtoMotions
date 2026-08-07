# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Remove ArticulationRootAPI from the converter's `worldBody` prim.

convert_robot_mjcf_to_usda.py emits TWO prims carrying
UsdPhysics.ArticulationRootAPI in <robot>_flat_physics.usd: the real root
(e.g. /..._cleaned/RigPelvis/RigPelvis) and a `worldBody` standing in for the
MJCF <worldbody>. The composed .usda deactivates worldBody, so most code paths
never notice -- but IsaacLab resolves articulations against the spawned prim
tree and finds both:

    RuntimeError: Failed to find a single articulation when resolving
    '/World/envs/env_0/Robot'. Found multiple
    '[.../Robot/worldBody, .../Robot/RigPelvis/RigPelvis]'

worldBody is the STATIC WORLD. It has no joints and cannot be an articulation,
so the API is meaningless there; removing it makes the resolution unambiguous
without touching the real root. Run after convert, alongside
apply_mjcf_contact_excludes.py.

    python usd_convert/strip_worldbody_articulation.py \
        --usd protomotions/data/assets/usd/tiger/configuration/tiger_flat_physics.usd
"""
from __future__ import annotations

import argparse

from pxr import Usd, UsdPhysics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.usd)
    roots = [p for p in stage.Traverse()
             if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print(f"articulation roots found: {[p.GetName() for p in roots]}")

    victims = [p for p in roots if p.GetName() == "worldBody"]
    keep = [p for p in roots if p.GetName() != "worldBody"]
    if not victims:
        print("no worldBody articulation root; nothing to do")
        return
    if not keep:
        # Never leave the asset with zero roots -- that fails just as hard,
        # and less legibly.
        print("REFUSING: worldBody is the ONLY articulation root, so removing "
              "it would leave the asset with none. Check the conversion.")
        raise SystemExit(1)

    for p in victims:
        print(f"  removing ArticulationRootAPI from {p.GetPath()}")
        if not args.dry_run:
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    if args.dry_run:
        print("dry run: nothing written")
        return
    stage.GetRootLayer().Save()
    # Check the stage we already hold. Re-opening the same layer returns the
    # cached stage whose prims this edit just expired, which raises
    # "Invalid range starting with expired 'Xform' prim" rather than telling
    # you anything about the file.
    after = [p.GetName() for p in stage.Traverse()
             if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print(f"roots now: {after}")
    if len(after) != 1:
        raise SystemExit(f"expected exactly 1 articulation root, got {after}")
    print(f"saved {args.usd}")


main()
