# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Bind the Atlas USD's materials to its visual meshes.

The MJCF->USD conversion carries the rig's materials (with diffuse textures,
copied into configuration/materials/) but binds NONE of them — every mesh
renders untextured gray. Mesh names encode their material as a suffix
(e.g. Chest__Aluminium, Hip__Aluminium_Blue), so binding is name-driven.

Also fixes the Bbody material: the rig ships no Bbody diffuse texture (the
MJCF pointed it at the ROUGHNESS map), so it becomes a dark constant — the
robot's black body panels.

Usage:
    ISAACLAB python usd_convert/patch_atlas_usd_bindings.py
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args)

from pxr import Gf, Usd, UsdGeom, UsdShade  # noqa: E402

BASE = "protomotions/data/assets/usd/atlas/configuration/atlas_flat_base.usd"

stage = Usd.Stage.Open(BASE)

# Materials by family name (material_<Family>), longest names first so
# Aluminium_Blue matches before Aluminium.
mats = {}
for prim in stage.Traverse():
    if prim.IsA(UsdShade.Material) and prim.GetName().startswith("material_"):
        mats[prim.GetName()[len("material_"):]] = UsdShade.Material(prim)
families = sorted(mats, key=len, reverse=True)
print("materials:", families, flush=True)

bound = plain = 0
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Mesh):
        continue
    name = prim.GetName()
    fam = next((f for f in families if name.endswith(f"__{f}")), None)
    if fam is None:
        plain += 1  # un-suffixed = collision/base meshes; leave unbound
        continue
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mats[fam])
    bound += 1
print(f"bound {bound} visual meshes; {plain} plain meshes left unbound", flush=True)

# Rebuild each material as a PROPER UsdPreviewSurface network. The MJCF
# converter authored MDL-style input names (diffuse_texture,
# diffuse_color_constant) on shaders whose info:id is UsdPreviewSurface —
# the renderer recognizes none of them and falls back to default gray.
# Correct schema: UsdPrimvarReader_float2(st) -> UsdUVTexture(file) ->
# surface.diffuseColor.
from pxr import Sdf  # noqa: E402

for fam, mat in mats.items():
    surface = None
    old_tex = None
    for sh in mat.GetPrim().GetChildren():
        if sh.IsA(UsdShade.Shader):
            shd = UsdShade.Shader(sh)
            if shd.GetIdAttr().Get() == "UsdPreviewSurface":
                surface = shd
                ti = shd.GetInput("diffuse_texture")
                if ti and ti.Get():
                    old_tex = ti.Get().path  # layer-relative texture path
    if surface is None:
        continue
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.4)
    if fam == "Bbody" or old_tex is None:
        # No usable diffuse texture (rig ships none for Bbody): constant color.
        color = Gf.Vec3f(0.04, 0.04, 0.045) if fam == "Bbody" else Gf.Vec3f(0.5, 0.5, 0.5)
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        print(f"{fam}: constant diffuse {tuple(color)}", flush=True)
        continue
    mpath = mat.GetPrim().GetPath()
    reader = UsdShade.Shader.Define(stage, mpath.AppendChild("stReader"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, mpath.AppendChild("diffuseTex"))
    tex.CreateIdAttr("UsdUVTexture")
    # ABSOLUTE path: the battle container's Fabric UsdToMdl pipeline fails
    # to anchor layer-relative asset paths ("can not be found: materials/X").
    import os as _os
    abs_tex = _os.path.abspath(
        _os.path.join("protomotions/data/assets/usd/atlas/configuration", str(old_tex))
    )
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(abs_tex)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    print(f"{fam}: diffuse texture network -> {old_tex}", flush=True)

stage.GetRootLayer().Save()
print("saved", BASE, flush=True)

import os  # noqa: E402
os._exit(0)
