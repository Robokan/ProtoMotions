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

# IMPORTANT: these materials expose ONLY an outputs:mdl:surface output (no
# universal outputs:surface), so every Omniverse renderer — the GUI and the
# training-time Fabric pipeline alike — resolves them through the MDL
# context, which reads the OmniPBR-STYLE inputs (diffuse_texture,
# diffuse_color_constant, metallic_constant, ...), NOT the UsdPreviewSurface
# inputs. The legacy inputs are therefore the AUTHORITATIVE ones and must be
# authored with correct values; the UsdPreviewSurface network is kept and a
# universal outputs:surface added for non-MDL consumers (usdview, other DCCs).

# Families rendered as a plain color CONSTANT so the color stays editable in
# the GUI (OmniPBR ignores the color constant whenever a texture is bound).
# Emission's shipped texture is a flat green (std ~0.01) — nothing is lost.
CONSTANT_COLOR = {
    "Emission": Gf.Vec3f(0.341, 0.906, 0.349),
}
# Bbody's diffuse is the flat Bbody_difussion.png Eric color-tunes by hand
# (same file the GMR MJCF binds); force it in case the import predates the
# GMR-side fix that pointed tex_Bbody at it.
FORCED_TEXTURE = {
    "Bbody": "materials/Bbody_difussion.png",
}

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
            elif sh.GetName() == "diffuseTex":
                # Re-run after legacy inputs were stripped: recover the
                # texture path from the network a previous run built.
                fi = UsdShade.Shader(sh).GetInput("file")
                if old_tex is None and fi and fi.Get():
                    old_tex = fi.Get().path.lstrip("./")
    if surface is None:
        continue
    if fam in FORCED_TEXTURE:
        old_tex = FORCED_TEXTURE[fam]
    sp = surface.GetPrim()
    # Universal surface output alongside the existing mdl:surface one.
    mat.CreateSurfaceOutput().ConnectToSource(
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    # Per-family PBR params from the source Atlas2025 rig; families not
    # listed use the generic painted-metal defaults. Dark families (Plastic's
    # diffuse texture averages 0.1) MUST be dielectric (metallic 0) or they
    # render as environment-gray, same failure as Bbody.
    metallic, roughness, spec = {
        "Plastic": (0.0, 0.56, 0.5),
        "Emission": (0.0, 0.5, 0.5),
        "Bbody": (0.0, 0.48, 0.13181818),
    }.get(fam, (0.4, 0.5, None))
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    # Legacy/MDL-context values (what actually renders in Omniverse).
    surface.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
    surface.CreateInput(
        "reflection_roughness_constant", Sdf.ValueTypeNames.Float
    ).Set(roughness)
    if spec is not None:
        surface.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(spec)
        surface.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(spec)
    if fam in CONSTANT_COLOR or old_tex is None:
        # Constant-color families: no texture bound, so the GUI color
        # controls actually take effect.
        color = CONSTANT_COLOR.get(fam, Gf.Vec3f(0.5, 0.5, 0.5))
        di = surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        di.GetAttr().ClearConnections()
        di.Set(color)
        sp.RemoveProperty("inputs:diffuse_texture")
        surface.CreateInput(
            "diffuse_color_constant", Sdf.ValueTypeNames.Color3f
        ).Set(color)
        # Drop any texture network a previous run of this script built.
        for child in ("diffuseTex", "stReader"):
            cpath = mat.GetPrim().GetPath().AppendChild(child)
            if stage.GetPrimAtPath(cpath):
                stage.RemovePrim(cpath)
        if fam == "Emission":
            # The accent rings glow their surface color.
            ec = surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
            ec.GetAttr().ClearConnections()
            ec.Set(color)
            sp.RemoveProperty("inputs:emissive_color_texture")
            surface.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
            surface.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(color)
            surface.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(1.0)
        print(f"{fam}: constant diffuse {tuple(color)}", flush=True)
        continue
    mpath = mat.GetPrim().GetPath()
    reader = UsdShade.Shader.Define(stage, mpath.AppendChild("stReader"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, mpath.AppendChild("diffuseTex"))
    tex.CreateIdAttr("UsdUVTexture")
    # Layer-anchored relative path. The "./" prefix matters: a bare
    # "materials/X.png" is a SEARCH path under USD's resolver rules (resolved
    # against resolver search paths, hence the earlier "can not be found"
    # errors), while "./materials/X.png" anchors to this layer's directory —
    # portable across machines and mount points.
    rel_tex = "./" + str(old_tex).lstrip("./")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel_tex)
    # MDL context reads THIS input — same layer-anchored path.
    surface.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(rel_tex)
    # No albedo adjustments: the texture IS the intended color (Bbody's flat
    # diffuse is hand-tuned at the source). Strip any authored by past runs.
    sp.RemoveProperty("inputs:albedo_add")
    sp.RemoveProperty("inputs:albedo_brightness")
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
