# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Translate MJCF <contact><exclude> pairs into PhysX filtered pairs.

The IsaacLab MJCF converter silently drops MuJoCo contact excludes, so
self-collision pairs that the MJCF intends to filter (natural same-side
proximity: arm swing past the thigh, guard tucked to the chin, deep-squat
thigh-on-torso, ...) collide anyway in Isaac. This script parses the
excludes from the source MJCF and applies UsdPhysics.FilteredPairsAPI to
the corresponding link prims in the converted physics layer.

Run AFTER convert_robot_mjcf_to_usda.py (any patch order is fine):

    python usd_convert/apply_mjcf_contact_excludes.py \
        --mjcf protomotions/data/assets/mjcf/t800.xml \
        --usd protomotions/data/assets/usd/t800/configuration/t800_flat_physics.usd

Idempotent: filtered-pair rels are replaced, not appended.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mjcf", default="protomotions/data/assets/mjcf/t800.xml")
parser.add_argument(
    "--usd",
    default="protomotions/data/assets/usd/t800/configuration/t800_flat_physics.usd",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)

from pxr import Usd, UsdPhysics  # noqa: E402


def main() -> None:
    contact = ET.parse(args.mjcf).getroot().find("contact")
    pairs = [
        (e.get("body1"), e.get("body2"))
        for e in (contact.findall("exclude") if contact is not None else [])
    ]
    if not pairs:
        print(f"no <contact><exclude> pairs in {args.mjcf}; nothing to do")
        return

    stage = Usd.Stage.Open(args.usd)
    body_prims: dict[str, Usd.Prim] = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetTypeName() == "Xform":
            body_prims.setdefault(prim.GetName(), prim)

    by_body1: dict[str, list[str]] = {}
    missing = []
    for b1, b2 in pairs:
        if b1 not in body_prims or b2 not in body_prims:
            missing.append((b1, b2))
            continue
        by_body1.setdefault(b1, []).append(b2)

    applied = 0
    for b1, others in by_body1.items():
        api = UsdPhysics.FilteredPairsAPI.Apply(body_prims[b1])
        rel = api.CreateFilteredPairsRel()
        rel.SetTargets([body_prims[b2].GetPath() for b2 in others])
        applied += len(others)
    stage.Save()

    print(f"applied {applied} filtered pairs on {len(by_body1)} bodies -> {args.usd}")
    if missing:
        print(f"WARNING: {len(missing)} pairs had unknown bodies: {missing}")
        sys.exit(1)


if __name__ == "__main__":
    main()
