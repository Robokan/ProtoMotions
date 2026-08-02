# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Post-export fixups for creature overlay USDs (run after
# build_creature_overlay_usd.py): Blender 4.5 exports mesh POINTS in
# meters but SKELETON bind/rest transforms in centimeters — skinning a
# meter mesh with a cm skeleton crushes it invisible. Also sets meshes
# doubleSided (UE negative-scale exports render single-sided-inverted).
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app = AppLauncher(args)
import numpy as np
from pxr import Usd, UsdSkel, Gf, Vt
for name in ("raptor", "tiger"):
    path = f"protomotions/data/assets/overlay/{name}.usd"
    stage = Usd.Stage.Open(path)
    skel_prim = next(p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton))
    sk = UsdSkel.Skeleton(skel_prim)
    # Idempotence: only scale when translations are clearly centimeters
    # (a bind skeleton spanning >5 units cannot be meters for these
    # creatures) — re-running the fixup must not double-shrink.
    probe = np.array(skel_prim.GetAttribute("bindTransforms").Get()[1]).reshape(4, 4)
    if np.abs(probe[3, :3]).max() < 5.0:
        print(f"{name}: skeleton already meters, skipping scale", flush=True)
    else:
        for attr_name in ("restTransforms", "bindTransforms"):
            attr = skel_prim.GetAttribute(attr_name)
            mats = attr.Get()
            out = []
            for m in mats:
                M = np.array(m).reshape(4, 4)
                M[3, :3] *= 0.01  # translations cm -> m
                out.append(Gf.Matrix4d(*M.flatten()))
            attr.Set(Vt.Matrix4dArray(out))
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            from pxr import UsdGeom as _UG
            _UG.Mesh(prim).GetDoubleSidedAttr().Set(True)
    # geomBindTransform on meshes, if authored, also carries units
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            a = prim.GetAttribute("primvars:skel:geomBindTransform")
            if a and a.HasAuthoredValue():
                M = np.array(a.Get()).reshape(4, 4)
                M[3, :3] *= 0.01
                a.Set(Gf.Matrix4d(*M.flatten()))
    stage.GetRootLayer().Save()
    b = np.array(sk.GetBindTransformsAttr().Get()[1]).reshape(4, 4)
    print(f"{name}: bind[1] trans now {np.round(b[3,:3],3)}", flush=True)
