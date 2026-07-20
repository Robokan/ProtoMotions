# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect T800 USD material bindings / texture paths."""
import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args)

from pxr import Usd, UsdGeom, UsdShade  # noqa: E402

BASE = "protomotions/data/assets/usd/t800/configuration/t800_flat_base.usd"
print("base size", os.path.getsize(BASE), flush=True)
st = Usd.Stage.Open(BASE)
mats = []
meshes = []
tex_attrs = []
for prim in st.Traverse():
    if prim.IsA(UsdShade.Material):
        mats.append(prim.GetName())
    if prim.IsA(UsdGeom.Mesh):
        bound = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterial()
        meshes.append((prim.GetName(), str(bound.GetPath()) if bound else None))
    for a in prim.GetAttributes():
        n = a.GetName()
        if "texture" in n.lower() or n.endswith(":file"):
            v = a.Get()
            if v is not None:
                tex_attrs.append((prim.GetName(), n, str(v)[:160]))

print("materials", len(mats), mats, flush=True)
bound_n = sum(1 for _, b in meshes if b)
print(f"meshes bound={bound_n}/{len(meshes)}", flush=True)
print("mesh names", [m for m, _ in meshes], flush=True)
print("tex attrs", len(tex_attrs), flush=True)
for t in tex_attrs[:40]:
    print(" ", t, flush=True)
cfg = Path("protomotions/data/assets/usd/t800/configuration")
print("config", [x.name for x in cfg.iterdir()], flush=True)
matdir = cfg / "materials"
print(
    "materials dir",
    matdir.exists(),
    [p.name for p in matdir.iterdir()][:20] if matdir.exists() else None,
    flush=True,
)
os._exit(0)
