# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Post-export fixups for creature overlay USDs (run after
# build_creature_overlay_usd.py): sets meshes doubleSided (UE
# negative-scale exports render single-sided-inverted).
#
# NOTE (hard-won): do NOT rescale the skeleton bind/rest translations.
# Blender 4.5 exports BOTH the mesh points and the skeleton in
# centimeters under a 0.01 metersPerUnit correction on the prim — the
# asset is internally consistent. An earlier version of this script
# converted the skeleton to meters, which left the joints 100x smaller
# than the mesh: the BIND pose still rendered perfectly (skinning
# matrices are identity at rest regardless of units) but any driven
# pose crushed the mesh onto a centimeter-sized skeleton — the
# infamous "mangled mess". Root-caused 2026-08-03 via a UsdSkel
# variant sweep (skel-space Tail6 at ~1cm from origin under every
# rotation variant).
#
# Materials are repaired by GEOMETRY AND TEXTURE PROPERTIES, never by
# name: this pack's UE/FBX export scrambles GeomSubset vs Material names
# (the subset called "DromaMESH_DromaBodyM" is the body skin and binds to
# the material called "DromaMESH_Material__26", and vice versa), so any
# name-based assignment puts the feather cards on the skin. Instead we
# find the alpha-cutout texture and give it to whichever material paints
# the SMALLEST subset — cutout cards (feathers, fur fins) are always a
# small face count next to a full body — and give the largest opaque
# texture to the material painting the biggest subset.
import argparse
import glob
import os
import shutil
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app = AppLauncher(args)
import numpy as np
from pxr import Usd, UsdSkel, UsdGeom, UsdShade, Sdf

for name in ("raptor", "tiger"):
    path = f"protomotions/data/assets/overlay/{name}.usd"
    stage = Usd.Stage.Open(path)
    skel_prim = next(p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton))
    sk = UsdSkel.Skeleton(skel_prim)
    # Guard against the old meter-scaled skeletons: if a previous run of
    # the buggy fixup shrank the skeleton, restore centimeters so it
    # matches the mesh points again.
    for attr_name in ("restTransforms", "bindTransforms"):
        attr = skel_prim.GetAttribute(attr_name)
        mats = list(attr.Get())
        mx = max(abs(v) for m in mats for v in (m[3][0], m[3][1], m[3][2]))
        if mx < 5.0:
            from pxr import Gf
            out = []
            for m in mats:
                m2 = Gf.Matrix4d(m)
                m2.SetTranslateOnly(
                    Gf.Vec3d(m[3][0] * 100, m[3][1] * 100, m[3][2] * 100))
                out.append(m2)
            attr.Set(out)
            print(f"{name}: {attr_name} restored to cm (was meter-scaled)",
                  flush=True)
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            from pxr import UsdGeom as _UG
            _UG.Mesh(prim).GetDoubleSidedAttr().Set(True)
    stage.GetRootLayer().Save()
    b = np.array(sk.GetBindTransformsAttr().Get()[1]).reshape(4, 4)
    print(f"{name}: bind[1] trans {np.round(b[3, :3], 2)} (cm)", flush=True)


# --- material repair: cutout texture -> smallest subset ------------------
TEXDIRS = {
    "raptor": "/home/bizon/sparkpack/UnrealExportedAssets/Raptor/Game/"
              "RaptorDinosaur/Model",
    "tiger": "/home/bizon/sparkpack/UnrealExportedAssets/Tiger/Game/"
             "Animalia/Tiger_M/Textures",
}
OVERLAY_TEX = "protomotions/data/assets/overlay/textures"


def _classify(texdir):
    """-> (cutout_path|None, opaque_path|None), by inspecting alpha."""
    from PIL import Image
    cands = []
    for pat in ("*_D.png", "*_D.PNG", "*BaseColor*.png", "*BaseColor*.PNG"):
        cands += glob.glob(os.path.join(texdir, pat))
    cutout, opaque = None, None
    best_cut, best_op = -1.0, -1
    for c in sorted(set(cands)):
        try:
            im = Image.open(c)
        except Exception:
            continue
        px = im.size[0] * im.size[1]
        if im.mode in ("RGBA", "LA"):
            a = np.asarray(im)[..., -1]
            frac = float((a < 128).mean())
            if frac > 0.01 and frac > best_cut:      # real cutout mask
                cutout, best_cut = c, frac
                continue
        if px > best_op:                              # biggest opaque map
            opaque, best_op = c, px
    return cutout, opaque


def _tex_shader(mat):
    for ch in Usd.PrimRange(mat):
        sh = UsdShade.Shader(ch)
        if sh and sh.GetShaderId() == "UsdUVTexture":
            return sh
    return None


def _surface_shader(mat):
    for ch in Usd.PrimRange(mat):
        sh = UsdShade.Shader(ch)
        if sh and sh.GetShaderId() == "UsdPreviewSurface":
            return sh
    return None


def repair_materials(stage, texdir):
    cutout, opaque = _classify(texdir)
    if cutout is None:
        return "no cutout texture found; left materials alone"
    os.makedirs(OVERLAY_TEX, exist_ok=True)
    for src in (cutout, opaque):
        if src and not os.path.exists(
                os.path.join(OVERLAY_TEX, os.path.basename(src))):
            shutil.copy(src, OVERLAY_TEX)
    # subsets, their bound materials, and their face counts
    entries = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "GeomSubset":
            continue
        binding = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
        mpath = binding.GetMaterialPath()
        if not mpath:
            continue
        idx = prim.GetAttribute("indices").Get()
        entries.append((len(idx) if idx else 0, prim.GetName(),
                        stage.GetPrimAtPath(mpath)))
    if len(entries) < 2:
        return "single-material mesh; nothing to disambiguate"
    entries.sort()
    notes = []
    for rank, (faces, sname, mprim) in enumerate(entries):
        want = cutout if rank == 0 else opaque
        if want is None:
            continue
        tex = _tex_shader(mprim)
        surf = _surface_shader(mprim)
        if tex is None or surf is None:
            continue
        tex.GetInput("file").Set(
            Sdf.AssetPath(f"./textures/{os.path.basename(want)}"))
        op = surf.GetInput("opacity") or surf.CreateInput(
            "opacity", Sdf.ValueTypeNames.Float)
        if rank == 0:      # alpha-clip the cutout cards
            aout = tex.GetOutput("a") or tex.CreateOutput(
                "a", Sdf.ValueTypeNames.Float)
            op.ConnectToSource(aout)
            th = surf.GetInput("opacityThreshold") or surf.CreateInput(
                "opacityThreshold", Sdf.ValueTypeNames.Float)
            th.Set(0.5)
        else:              # opaque body: drop any stale alpha connection
            if op.GetConnectedSource():
                op.DisconnectSource()
            op.Set(1.0)
        notes.append(f"{sname}({faces}f)->{os.path.basename(want)}")
    return "; ".join(notes)


for name in ("raptor", "tiger"):
    path = f"protomotions/data/assets/overlay/{name}.usd"
    if not os.path.exists(path):
        continue
    stage = Usd.Stage.Open(path)
    if TEXDIRS.get(name) and os.path.isdir(TEXDIRS[name]):
        msg = repair_materials(stage, TEXDIRS[name])
        stage.GetRootLayer().Save()
        print(f"{name}: materials {msg}", flush=True)
