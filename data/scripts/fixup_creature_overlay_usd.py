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
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app = AppLauncher(args)
import numpy as np
from pxr import Usd, UsdSkel

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
