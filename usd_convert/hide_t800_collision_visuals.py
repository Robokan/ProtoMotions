# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Hide T800 collision spheres that the MJCF→USD converter made visible.

Wrist bodies only had collision geoms, so Isaac Sim promoted the orange
``collision_urdf`` hand spheres into the visuals tree. Mark those (and any
other collision-purpose spheres) invisible without touching physics.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args)

from pxr import Usd, UsdGeom  # noqa: E402

BASE = Path("protomotions/data/assets/usd/t800/configuration/t800_flat_base.usd")


def main() -> None:
    stage = Usd.Stage.Open(str(BASE))
    if not stage:
        raise RuntimeError(f"failed to open {BASE}")

    hidden = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Imageable):
            continue
        path = str(prim.GetPath())
        name = prim.GetName()
        is_wrist = "WRIST_END" in path
        is_collision_branch = "/collisions" in path or path.endswith("/collisions")
        is_sphere = prim.IsA(UsdGeom.Sphere)
        # Hide wrist visuals and anything under the collisions tree.
        if not (is_wrist or is_collision_branch):
            continue
        if is_wrist and not (is_sphere or "visuals" in path or "collisions" in path):
            # Still hide Xforms under WRIST visuals so children disappear.
            pass
        img = UsdGeom.Imageable(prim)
        img.GetVisibilityAttr().Set("invisible")
        # Guide purpose keeps them out of default beauty renders if visibility is ignored.
        if is_sphere or is_collision_branch:
            img.GetPurposeAttr().Set("guide")
        hidden += 1
        print(f"  hide {path} ({prim.GetTypeName()})", flush=True)

    stage.GetRootLayer().Save()
    print(f"hid {hidden} prims; saved {BASE}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
