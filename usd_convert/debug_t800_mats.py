# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
AppLauncher(parser.parse_args(["--headless"]))

from pxr import Usd, UsdGeom, UsdShade  # noqa: E402

path = "protomotions/data/assets/usd/t800/configuration/t800_flat_base.usd"
st = Usd.Stage.Open(path)
print("DEFAULT", st.GetDefaultPrim().GetPath() if st.GetDefaultPrim() else None, flush=True)
print("ROOT children", [c.GetPath().pathString for c in st.GetPseudoRoot().GetChildren()], flush=True)

for prim in st.Traverse():
    if prim.IsA(UsdShade.Material) and prim.GetName() == "material_LINK_BASE":
        print("MAT", prim.GetPath(), flush=True)
        for child in prim.GetChildren():
            print("  child", child.GetName(), child.GetTypeName(), flush=True)
            if child.IsA(UsdShade.Shader):
                sh = UsdShade.Shader(child)
                for inp in sh.GetInputs():
                    val = inp.Get()
                    if val is not None:
                        print("   ", inp.GetBaseName(), "=", val, flush=True)
    if prim.IsA(UsdGeom.Mesh) and prim.GetName() == "LINK_BASE":
        b = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterial()
        print("MESH", prim.GetPath(), "bound", b.GetPath() if b else None, flush=True)
        pv = UsdGeom.PrimvarsAPI(prim)
        print(
            " primvars",
            [(x.GetPrimvarName(), str(x.GetTypeName())) for x in pv.GetPrimvars()],
            flush=True,
        )

# Atlas comparison
apath = "protomotions/data/assets/usd/atlas/configuration/atlas_flat_base.usd"
ast = Usd.Stage.Open(apath)
print("\nATLAS DEFAULT", ast.GetDefaultPrim().GetPath() if ast.GetDefaultPrim() else None, flush=True)
n = 0
for prim in ast.Traverse():
    if prim.IsA(UsdShade.Material) and n < 3:
        print("ATLAS MAT", prim.GetPath(), flush=True)
        n += 1
    if prim.IsA(UsdGeom.Mesh) and "Aluminium" in prim.GetName():
        b = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterial()
        print("ATLAS MESH", prim.GetPath(), "->", b.GetPath() if b else None, flush=True)
        break

os._exit(0)
