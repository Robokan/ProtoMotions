# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Add an MDL (OmniPBR) render context to a UsdPreviewSurface asset so RTX
# renders its textures (Blender exports preview-only -> black in Kit).
# Run: python add_mdl_context.py <asset.usd>  (needs pxr)
import sys
from pxr import Usd, UsdShade, Sdf
path = sys.argv[1]
stage = Usd.Stage.Open(path)
fixed = 0
for prim in list(stage.Traverse()):
    if prim.GetTypeName() != "Material":
        continue
    mat = UsdShade.Material(prim)
    tex = {}
    for child in prim.GetChildren():
        if child.GetTypeName() != "Shader":
            continue
        sh = UsdShade.Shader(child)
        if sh.GetIdAttr().Get() != "UsdUVTexture":
            continue
        f = sh.GetInput("file")
        if not f or not f.Get():
            continue
        s = str(f.Get()).lower()
        if "basecolor" in s: tex["diffuse"] = f.Get()
        elif "normal" in s: tex["normal"] = f.Get()
        elif "occlusionroughness" in s: tex["orm"] = f.Get()
    if "diffuse" not in tex:
        continue
    mdl = UsdShade.Shader.Define(stage, prim.GetPath().AppendChild("mdl"))
    mdl.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
    mdl.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    mdl.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    mdl.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(tex["diffuse"])
    if "normal" in tex:
        mdl.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(tex["normal"])
    if "orm" in tex:
        mdl.CreateInput("ORM_texture", Sdf.ValueTypeNames.Asset).Set(tex["orm"])
        mdl.CreateInput("enable_ORM_texture", Sdf.ValueTypeNames.Bool).Set(True)
    out = mdl.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    fixed += 1
stage.GetRootLayer().Save()
print(f"{path}: MDL added to {fixed} materials")
