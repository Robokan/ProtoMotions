# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Give an overlay character's materials their textures.

fixup_creature_overlay_usd.py REPAIRS materials, and to do that it needs a
UsdPreviewSurface and a UsdUVTexture already present -- it finds them by
shader id and rewires the file paths. A creature imported through Isaac's
native path has neither: tiger.usd carries only OmniPBR MDL shaders holding
`diffuse_color_constant (0.8, 0.8, 0.8)` and no texture input at all, so the
repair pass reports "no cutout texture found" and leaves it flat grey. This
script BUILDS the missing wiring instead.

BOTH SHADING MODELS ARE WIRED, deliberately. Isaac's RTX renderer prefers
`outputs:mdl:surface` when a material has one and ignores UsdPreviewSurface;
usdview and most other tools do the opposite. Setting only one leaves the
character untextured in the viewer you happen to open it in, which is exactly
the failure this script exists to fix, so it sets the MDL's diffuse_texture
AND attaches a UsdPreviewSurface network.

TEXTURE ASSIGNMENT IS AUDITED, NOT ASSUMED. This pack's UE export has crossed
GeomSubset and Material names before (on the raptor, subset
"DromaMESH_DromaBodyM" is the 8768-face skin but binds the material called
"DromaMESH_Material__26"), so every assignment is printed with its subset's
face count. A fur texture landing on 908 card faces and a body texture on
15784 body faces is self-evidently right; a swap would be obvious.

    python data/scripts/wire_overlay_textures.py \
        --usd protomotions/data/assets/overlay/tiger.usd --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

from pxr import Sdf, Usd, UsdGeom, UsdShade

OVERLAY_TEX = "protomotions/data/assets/overlay/textures"


def find_textures(texdir: str) -> dict:
    out = {}
    for pat in ("*BaseColor*.png", "*BaseColor*.PNG", "*_D.png", "*_D.PNG"):
        for p in glob.glob(os.path.join(texdir, pat)):
            out[os.path.basename(p)] = p
    return out


def assign_textures(entries, textures: dict) -> dict:
    """material path -> (texture basename | None, reason).

    Two passes, and a texture can only be claimed ONCE.

    Pass 1 requires the filename to START WITH the full material name, e.g.
    "Tiger_M_Material_BaseColor.png" for "Tiger_M_Material". A looser
    substring test is what a first version did and it was wrong: the stem
    "Tiger_M" is a prefix of BOTH materials here, so the 15784-face body
    subset claimed the fur card map. It happened to be invisible because this
    export duplicates the same image under several names -- on an asset whose
    maps actually differ it would have silently painted fur over the body.

    Pass 2 hands the largest remaining map to the largest unassigned subset.
    Small accessory subsets (eyes) are left on their constant colour rather
    than given a body map.
    """
    out, claimed = {}, set()
    for faces, sname, mprim in entries:
        mname = mprim.GetName().lower()
        hit = next((b for b in sorted(textures)
                    if b not in claimed and b.lower().startswith(mname)), None)
        if hit:
            out[mprim.GetPath()] = (hit, "exact name prefix")
            claimed.add(hit)
    for faces, sname, mprim in entries:
        if mprim.GetPath() in out:
            continue
        if faces < 500:
            out[mprim.GetPath()] = (
                None, "too few faces for a body map; left as constant colour")
            continue
        free = sorted((b for b in textures if b not in claimed),
                      key=lambda b: -os.path.getsize(textures[b]))
        if free:
            out[mprim.GetPath()] = (free[0], "largest unclaimed map")
            claimed.add(free[0])
        else:
            out[mprim.GetPath()] = (None, "no unclaimed texture left")
    return out


def wire_mdl(shader: UsdShade.Shader, rel: str, cutout: bool) -> None:
    """Point the OmniPBR MDL at the texture (this is what Isaac renders)."""
    def s_in(name, tv, val):
        i = shader.GetInput(name) or shader.CreateInput(name, tv)
        i.Set(val)
        return i

    s_in("diffuse_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(rel))
    # A grey constant MULTIPLIES the sampled texture and darkens it; white is
    # the neutral value once a map is supplying the colour.
    s_in("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, (1.0, 1.0, 1.0))
    if cutout:
        s_in("opacity_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(rel))
        s_in("enable_opacity", Sdf.ValueTypeNames.Bool, True)
        s_in("enable_opacity_texture", Sdf.ValueTypeNames.Bool, True)
        s_in("opacity_threshold", Sdf.ValueTypeNames.Float, 0.5)
    else:
        # leave the surface fully opaque; a stray threshold can punch holes
        s_in("enable_opacity_texture", Sdf.ValueTypeNames.Bool, False)
        s_in("opacity_constant", Sdf.ValueTypeNames.Float, 1.0)
        s_in("opacity_threshold", Sdf.ValueTypeNames.Float, 0.0)


def wire_preview(stage, mat: UsdShade.Material, rel: str, uv: str,
                 cutout: bool) -> None:
    """Attach a UsdPreviewSurface network so non-Isaac viewers show it too."""
    mp = mat.GetPath()
    surf = UsdShade.Shader.Define(stage, mp.AppendChild("PreviewSurface"))
    surf.CreateIdAttr("UsdPreviewSurface")
    tex = UsdShade.Shader.Define(stage, mp.AppendChild("DiffuseTex"))
    tex.CreateIdAttr("UsdUVTexture")
    rdr = UsdShade.Shader.Define(stage, mp.AppendChild("UvReader"))
    rdr.CreateIdAttr("UsdPrimvarReader_float2")

    rdr.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(uv)
    st_out = rdr.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(rel))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
    rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f) \
        .ConnectToSource(rgb)
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if cutout:
        a = tex.CreateOutput("a", Sdf.ValueTypeNames.Float)
        surf.CreateInput("opacity", Sdf.ValueTypeNames.Float) \
            .ConnectToSource(a)
        surf.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
    else:
        surf.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    # universal render context only -- leaves outputs:mdl:surface intact
    mat.CreateSurfaceOutput().ConnectToSource(
        surf.CreateOutput("surface", Sdf.ValueTypeNames.Token))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True)
    ap.add_argument("--texdir", default=OVERLAY_TEX)
    ap.add_argument("--uv", default=None,
                    help="UV primvar name; auto-detected from the mesh")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    textures = find_textures(args.texdir)
    if not textures:
        raise SystemExit(f"no *BaseColor*/_D textures under {args.texdir}")

    stage = Usd.Stage.Open(args.usd)
    meshes = [p for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
    if not meshes:
        raise SystemExit("no Mesh in stage")
    mesh = max(meshes, key=lambda p: len(
        UsdGeom.Mesh(p).GetPointsAttr().Get() or []))

    uv = args.uv
    if uv is None:
        for v in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
            if "float2" in str(v.GetTypeName()) or "texCoord2" in str(v.GetTypeName()):
                uv = v.GetName().replace("primvars:", "")
                break
        uv = uv or "st"
    print(f"mesh {mesh.GetPath()}   uv primvar '{uv}'")

    # subsets with their bound material and face count
    entries = []
    for c in mesh.GetChildren():
        if c.GetTypeName() != "GeomSubset":
            continue
        mpath = UsdShade.MaterialBindingAPI(c).GetDirectBinding() \
            .GetMaterialPath()
        if not mpath:
            continue
        idx = c.GetAttribute("indices").Get()
        entries.append([len(idx) if idx else 0, c.GetName(),
                        stage.GetPrimAtPath(mpath)])
    if not entries:
        raise SystemExit("no GeomSubsets with material bindings")
    entries.sort(key=lambda e: -e[0])

    os.makedirs(args.texdir, exist_ok=True)
    plan = assign_textures(entries, textures)
    print(f"{'subset':<32}{'faces':>8}  material -> texture")
    changed = 0
    for rank, (faces, sname, mprim) in enumerate(entries):
        mat = UsdShade.Material(mprim)
        base, why = plan[mprim.GetPath()]
        if base is None:
            print(f"{sname:<32}{faces:8d}  {mprim.GetName()} -> (none: {why})")
            continue
        rel = f"./textures/{base}"
        # the smallest textured subset is the card geometry (fur/feathers)
        cutout = (rank == len(entries) - 1) and faces < 0.2 * entries[0][0]
        print(f"{sname:<32}{faces:8d}  {mprim.GetName()} -> {base}"
              f"{'  [alpha cutout]' if cutout else ''}  ({why})")
        if args.dry_run:
            continue
        for ch in Usd.PrimRange(mprim):
            sh = UsdShade.Shader(ch)
            if sh and sh.GetPrim().GetAttribute("info:mdl:sourceAsset"):
                wire_mdl(sh, rel, cutout)
        wire_preview(stage, mat, rel, uv, cutout)
        changed += 1

    if args.dry_run:
        print("\ndry run: nothing written")
        return
    bak = args.usd + ".pre_texwire_bak"
    if not os.path.exists(bak):
        shutil.copy(args.usd, bak)
        print(f"\nbackup: {bak}")
    stage.GetRootLayer().Save()
    print(f"wired {changed} material(s); saved {args.usd}")


main()
