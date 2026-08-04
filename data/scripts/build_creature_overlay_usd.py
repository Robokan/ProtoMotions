# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Generalized creature overlay builder (raptor/tiger/... — the samurai
# builder's single-FBX sibling): import ONE skinned FBX, keep only the
# highest-detail LOD, wire base-color textures by material-name fuzzy
# match against a texture directory, export a UsdSkel character for the
# overlay system (keys 5/6 in the viewers).
#
# Run:
#   blender --background --python build_creature_overlay_usd.py -- \
#       --fbx <mesh.fbx> --texdir <dir with *_D.png/*BaseColor*.png> \
#       --out <overlay.usd>
import argparse
import glob
import os
import re
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("--fbx", required=True)
ap.add_argument("--texdir", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=args.fbx)

# Keep only LOD0 meshes (delete LOD1+); drop cameras/lights/empties.
for ob in list(bpy.data.objects):
    if ob.type == "MESH" and re.search(r"LOD[1-9]\d*$", ob.name):
        bpy.data.objects.remove(ob, do_unlink=True)
    elif ob.type in ("CAMERA", "LIGHT"):
        bpy.data.objects.remove(ob, do_unlink=True)

# Texture candidates: basecolor/diffuse first.
tex_files = []
for pat in ("*BaseColor*.png", "*_D.png", "*Diffuse*.png", "*_BC.png"):
    tex_files += glob.glob(os.path.join(args.texdir, pat))
tex_files = sorted(set(tex_files))
print("basecolor candidates:", [os.path.basename(t) for t in tex_files])


def tokens(name):
    n = re.sub(r"\.(png|PNG)$", "", os.path.basename(name))
    n = re.sub(r"(_BaseColor|_D$|Material|MESH|_BC)", "", n, flags=re.I)
    return set(t for t in re.split(r"[_\W]+", n.lower()) if len(t) > 2)


# GOTCHA (raptor, 2026-08-03): UE/FBX exports can scramble subset vs
# material NAMES -- the GeomSubset called "DromaMESH_DromaBodyM" (8768
# faces) is the body skin while the one called "DromaMESH_Material__26"
# (1466 faces) is the feather cards, and each must bind to the material
# named after the OTHER. Do not trust the names: after a rebuild, verify
# that the CUTOUT texture (alpha < 1 somewhere, e.g. Feathers_D.png)
# lands on the small subset and the opaque body texture on the large one,
# and swap the bindings if not. Also note this pack ships a dedicated
# feather texture that name-based fuzzy matching will NOT find, because
# neither material name contains "feather".
wired = 0
for mat in bpy.data.materials:
    mtoks = tokens(mat.name)
    best, score = None, 0
    for t in tex_files:
        sc = len(mtoks & tokens(t))
        if sc > score:
            best, score = t, sc
    if best is None and len(tex_files) == 1:
        best = tex_files[0]
    if best is None:
        print(f"  no texture for material {mat.name}")
        continue
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        continue
    im = bpy.data.images.load(best, check_existing=True)
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = im
    nt.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    # Opacity: FBX imports can carry a transparency factor that exports as
    # opacity~0 (invisible mesh). Force opaque UNLESS the texture has a real
    # cutout mask (e.g. feather cards) — then wire alpha with CLIP.
    import numpy as _np
    px = _np.asarray(im.pixels).reshape(-1, im.channels) if im.channels == 4 else None
    has_cutout = px is not None and px[:, 3].mean() < 0.97
    for link in list(nt.links):
        if link.to_node == bsdf and link.to_socket.name in ("Alpha", "Normal"):
            # FBX-imported normal-map chains export flattened/incorrectly
            # (texture straight into normal, no tangent-space transform)
            # and break Isaac's UsdPreviewSurface compile -> invisible mesh.
            nt.links.remove(link)
    if has_cutout:
        nt.links.new(node.outputs["Alpha"], bsdf.inputs["Alpha"])
        mat.blend_method = "CLIP"
        print(f"  {mat.name} <- {os.path.basename(best)} (alpha CLIP)")
    else:
        bsdf.inputs["Alpha"].default_value = 1.0
        mat.blend_method = "OPAQUE"
        print(f"  {mat.name} <- {os.path.basename(best)} (opaque)")
    wired += 1
print(f"wired {wired}/{len(bpy.data.materials)} materials")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
bpy.ops.wm.usd_export(
    filepath=args.out,
    export_materials=True,
    export_textures=True,
    export_animation=False,
    export_armatures=True,  # Blender 4.x spelling for UsdSkel export
    selected_objects_only=False,
)
print(f"saved {args.out}")
