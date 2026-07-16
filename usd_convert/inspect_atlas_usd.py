"""Inspect the converted Atlas USD: mesh names, UV primvars, existing materials."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args)

from pxr import Usd, UsdGeom, UsdShade  # noqa: E402

st = Usd.Stage.Open("protomotions/data/assets/usd/atlas/atlas_flat.usda")
meshes = [p for p in Usd.PrimRange.Stage(st, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
mats = [p for p in Usd.PrimRange.Stage(st, Usd.TraverseInstanceProxies()) if p.IsA(UsdShade.Material)]
print("MESHES", len(meshes), "MATERIALS", len(mats), flush=True)
for p in meshes[:14]:
    pv = UsdGeom.PrimvarsAPI(p)
    uv = [v.GetName() for v in pv.GetPrimvars()
          if "st" in v.GetName().lower() or "uv" in v.GetName().lower()]
    b = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
    bpath = b.GetPrim().GetPath() if b else None
    print("M", p.GetPath(), "| uv:", uv, "| bound:", bpath, flush=True)
for p in mats[:12]:
    print("MAT", p.GetPath(), flush=True)
    for sh in p.GetChildren():
        if sh.IsA(UsdShade.Shader):
            shd = UsdShade.Shader(sh)
            ins = {i.GetBaseName(): (i.Get() if i.Get() is not None else "<conn>") for i in shd.GetInputs()}
            print("   SHADER", sh.GetName(), shd.GetIdAttr().Get(), ins, flush=True)
import os
os._exit(0)
