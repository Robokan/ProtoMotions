# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Bind T800 USD meshes to textured materials under the default prim.

The MJCF importer left meshes unbound (white). Materials MUST live under the
default prim (``/t800_flat_cleaned/Looks/...``), matching Atlas — a top-level
``/Looks`` is dropped when IsaacLab references the asset.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args)

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

BASE = Path("protomotions/data/assets/usd/t800/configuration/t800_flat_base.usd")
TEX_SRC = Path("protomotions/data/assets/mesh/T800/texture")
MAT_DIR = BASE.parent / "materials"
DEFAULT_RGBA = Gf.Vec3f(0.792157, 0.819608, 0.933333)

MESH_TEXTURE = {
    "LINK_BASE": "LINK_BASE.png",
    "LINK_HIP_PITCH_L": "LINK_HIP_PITCH_L.png",
    "LINK_HIP_PITCH_R": "LINK_HIP_PITCH_L.png",
    "LINK_HIP_ROLL_L": "LINK_HIP_ROLL_L.png",
    "LINK_HIP_ROLL_R": "LINK_HIP_ROLL_L.png",
    "LINK_HIP_YAW_L": "LINK_HIP_YAW_L.png",
    "LINK_HIP_YAW_R": "LINK_HIP_YAW_L.png",
    "LINK_KNEE_PITCH_L": "LINK_KNEE_PITCH_L.png",
    "LINK_KNEE_PITCH_R": "LINK_KNEE_PITCH_L.png",
    "LINK_ANKLE_ROLL_L": "LINK_ANKLE_ROLL_L.png",
    "LINK_ANKLE_ROLL_R": "LINK_ANKLE_ROLL_R.png",
    "LINK_ANKLE_PITCH_L": None,
    "LINK_ANKLE_PITCH_R": None,
    "LINK_TORSO_YAW": "LINK_TORSO_YAW.png",
    "LINK_SHOULDER_PITCH_L": "LINK_SHOULDER_PITCH_L.png",
    "LINK_SHOULDER_PITCH_R": "LINK_SHOULDER_PITCH_L.png",
    "LINK_SHOULDER_ROLL_L": "LINK_SHOULDER_ROLL_L.png",
    "LINK_SHOULDER_ROLL_R": "LINK_SHOULDER_ROLL_L.png",
    "LINK_SHOULDER_YAW_L": "LINK_SHOULDER_YAW_L.png",
    "LINK_SHOULDER_YAW_R": "LINK_SHOULDER_YAW_L.png",
    "LINK_ELBOW_PITCH_L": "LINK_ELBOW_PITCH_L.png",
    "LINK_ELBOW_PITCH_R": "LINK_ELBOW_PITCH_R.png",
    "LINK_ELBOW_YAW_L": "LINK_ELBOW_YAW_L.png",
    "LINK_ELBOW_YAW_R": "LINK_ELBOW_YAW_R.png",
    "LINK_HEAD_YAW": "LINK_HEAD_YAW.png",
    "LINK_HEAD_PITCH": None,
}


def _ensure_textures() -> None:
    MAT_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for tex in MESH_TEXTURE.values():
        if tex is None:
            continue
        src, dst = TEX_SRC / tex, MAT_DIR / tex
        if not src.exists():
            print(f"WARN missing source texture: {src}", flush=True)
            continue
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dst)
            copied += 1
    print(f"materials dir: {MAT_DIR} (copied {copied})", flush=True)


def _make_material(
    stage: Usd.Stage, looks_path: Sdf.Path, name: str, tex_file: str | None
) -> UsdShade.Material:
    mat_path = looks_path.AppendChild(f"material_{name}")
    if stage.GetPrimAtPath(mat_path):
        stage.RemovePrim(mat_path)
    mat = UsdShade.Material.Define(stage, mat_path)
    surface = UsdShade.Shader.Define(stage, mat_path.AppendChild("Shader"))
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.15)
    surface.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.15)
    surface.CreateInput(
        "reflection_roughness_constant", Sdf.ValueTypeNames.Float
    ).Set(0.45)

    if tex_file is None:
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(DEFAULT_RGBA)
        surface.CreateInput(
            "diffuse_color_constant", Sdf.ValueTypeNames.Color3f
        ).Set(DEFAULT_RGBA)
    else:
        rel = f"./materials/{tex_file}"
        reader = UsdShade.Shader.Define(stage, mat_path.AppendChild("stReader"))
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, mat_path.AppendChild("diffuseTex"))
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        )
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        )
        surface.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(rel)
        surface.CreateInput(
            "diffuse_color_constant", Sdf.ValueTypeNames.Color3f
        ).Set(Gf.Vec3f(1, 1, 1))

    mat.CreateSurfaceOutput().ConnectToSource(
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    # Omniverse MDL surface output
    mat.GetPrim().CreateAttribute(
        "outputs:mdl:surface", Sdf.ValueTypeNames.Token
    ).SetConnections([surface.GetPath().AppendProperty("outputs:surface")])
    return mat


def main() -> None:
    _ensure_textures()
    stage = Usd.Stage.Open(str(BASE))
    default = stage.GetDefaultPrim()
    if not default:
        raise RuntimeError(f"no default prim in {BASE}")
    looks_path = default.GetPath().AppendChild("Looks")
    if not stage.GetPrimAtPath(looks_path):
        UsdGeom.Scope.Define(stage, looks_path)
    print(f"Looks path: {looks_path}", flush=True)

    # Remove bogus top-level /Looks from the earlier patch (outside default prim).
    if stage.GetPrimAtPath("/Looks"):
        stage.RemovePrim("/Looks")
        print("removed orphan /Looks", flush=True)

    mats = {}
    for mesh_name, tex in MESH_TEXTURE.items():
        mats[mesh_name] = _make_material(stage, looks_path, mesh_name, tex)
        print(
            f"  material_{mesh_name}: "
            f"{'texture '+tex if tex else 'constant rgba'}",
            flush=True,
        )

    bound = skipped = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mat = mats.get(prim.GetName())
        if mat is None:
            skipped += 1
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
        bound += 1

    stage.GetRootLayer().Save()
    print(f"bound {bound} meshes; skipped {skipped}", flush=True)
    print(f"saved {BASE}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
