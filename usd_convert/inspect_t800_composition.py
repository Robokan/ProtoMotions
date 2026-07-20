# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect T800 vs Atlas USD composition / material reachability under PhysX."""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
AppLauncher(parser.parse_args(["--headless"]))

from pxr import Usd, UsdShade  # noqa: E402


def dump(label: str, usda: str) -> None:
    print(f"\n===== {label} {usda} =====", flush=True)
    st = Usd.Stage.Open(usda)
    print("defaultPrim", st.GetDefaultPrim().GetPath() if st.GetDefaultPrim() else None)
    # Count materials / bound meshes in composed stage
    n_mat = 0
    n_mesh = 0
    n_bound = 0
    sample_binds = []
    for p in st.Traverse():
        if p.IsA(UsdShade.Material):
            n_mat += 1
            if n_mat <= 3:
                print(" MAT", p.GetPath())
        if p.GetTypeName() == "Mesh":
            n_mesh += 1
            bound = UsdShade.MaterialBindingAPI(p).GetDirectBinding().GetMaterial()
            if bound:
                n_bound += 1
                if len(sample_binds) < 5:
                    sample_binds.append((str(p.GetPath()), str(bound.GetPath())))
    print(f"materials={n_mat} meshes={n_mesh} bound={n_bound}")
    for m, b in sample_binds:
        print(" BIND", m, "->", b)


dump("T800", "protomotions/data/assets/usd/t800/t800_flat.usda")
dump("ATLAS", "protomotions/data/assets/usd/atlas/atlas_flat.usda")

# Also open physics layers alone
for name in ("t800", "atlas"):
    path = f"protomotions/data/assets/usd/{name}/configuration/{name}_flat_physics.usd"
    dump(f"{name} physics", path)
