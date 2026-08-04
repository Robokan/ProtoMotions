# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Bake a colour tint into a creature overlay's BODY texture and point a
# variant USD at it, so fighters can be told apart on sight (champion vs
# opponent in battle exhibitions).
#
# Why bake rather than tint in the shader: Isaac translates
# UsdPreviewSurface to MDL for the RTX renderer and the `scale` multiplier
# on a UsdUVTexture is dropped in that translation -- setting it changes
# nothing on screen. Multiplying the pixels always works.
#
#   python data/scripts/tint_overlay_body.py --creature raptor \
#       --out-usd raptor.usd --suffix purple --gains 1.13 0.82 1.27
#
# The body material is found by GEOMETRY (the material painting the
# largest GeomSubset), never by name -- these exports scramble names.
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--creature", default="raptor")
parser.add_argument("--out-usd", default="raptor.usd",
                    help="USD in overlay/ to repoint (copy it first for a variant)")
parser.add_argument("--suffix", default="purple")
parser.add_argument("--gains", nargs=3, type=float, default=[1.13, 0.82, 1.27],
                    help="per-channel R G B multipliers; the MIDDLE value "
                         "drives the hue shift, the outer two the brightness")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args)

import os  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Usd, UsdShade, Sdf  # noqa: E402

OVERLAY = "protomotions/data/assets/overlay"


def body_texture_shader(stage):
    entries = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "GeomSubset":
            continue
        binding = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
        if not binding.GetMaterialPath():
            continue
        idx = prim.GetAttribute("indices").Get()
        entries.append((len(idx) if idx else 0,
                        stage.GetPrimAtPath(binding.GetMaterialPath())))
    if not entries:
        return None, None
    entries.sort()
    mat = entries[-1][1]          # biggest subset == the body skin
    shader = next(
        (UsdShade.Shader(c) for c in Usd.PrimRange(mat)
         if UsdShade.Shader(c)
         and UsdShade.Shader(c).GetShaderId() == "UsdUVTexture"), None)
    return shader, mat


def tint_shader(shader, label):
    """Bake the gains into this shader's texture. RGB only -- the alpha
    channel is left untouched so cutout masks (feather cards) survive."""
    src_rel = shader.GetInput("file").Get().path
    src = os.path.join(OVERLAY, src_rel.lstrip("./"))
    stem, ext = os.path.splitext(os.path.basename(src))
    stem = stem.split("_" + args.suffix)[0]          # don't stack suffixes
    src = os.path.join(os.path.dirname(src), stem + ext)
    if not os.path.exists(src):
        print(f"  {label}: source {os.path.basename(src)} missing, skipped")
        return
    dst = os.path.join(os.path.dirname(src), f"{stem}_{args.suffix}{ext}")
    im = Image.open(src).convert("RGBA")
    a = np.asarray(im).astype(np.float32)
    gains = np.array(list(args.gains) + [1.0], dtype=np.float32)  # alpha x1
    out = np.clip(a * gains, 0, 255).astype(np.uint8)
    Image.fromarray(out, "RGBA").save(dst)
    shader.GetInput("file").Set(
        Sdf.AssetPath(f"./textures/{os.path.basename(dst)}"))
    before = a[..., :3].reshape(-1, 3).mean(0)
    after = out[..., :3].reshape(-1, 3).mean(0)
    keep = "alpha preserved" if im.mode == "RGBA" else ""
    print(f"  {label}: {os.path.basename(src)} -> {os.path.basename(dst)}  "
          f"RGB {before.round(1)} -> {after.round(1)} {keep}")


def main():
    path = os.path.join(OVERLAY, args.out_usd)
    stage = Usd.Stage.Open(path)
    # Tint EVERY material's texture, not just the body: the raptor's feather
    # cards are a separate material and looked untinted beside a purple body.
    done = set()
    print(f"{args.out_usd}  gains {args.gains}")
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Material":
            continue
        for child in Usd.PrimRange(prim):
            sh = UsdShade.Shader(child)
            if not sh or sh.GetShaderId() != "UsdUVTexture":
                continue
            f = sh.GetInput("file")
            if not f or not f.Get():
                continue
            key = str(child.GetPath())
            if key in done:
                continue
            done.add(key)
            tint_shader(sh, prim.GetName())
    if not done:
        raise SystemExit(f"no textured materials found in {path}")
    stage.GetRootLayer().Save()


main()
