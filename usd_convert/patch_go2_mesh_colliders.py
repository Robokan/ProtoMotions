#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Make the go2's colliders track its meshes, for sim-to-real contact fidelity.

The shipped go2 USD is in a half-converted state: thighs, calves and the base
carry BOTH a mesh collider and primitive colliders, all collisionEnabled, so
PhysX simulates the union (the primitive envelope wins wherever it is fatter
than the mesh). Hips and the base's own shell have visual meshes with no
collider at all, and the mesh decomposition is capped at 3 hulls -- barely
better than the box it was meant to replace.

This patches, per link:
  thighs / calves : keep the mesh collider, DISABLE the duplicate primitives
  hips / base     : ENABLE a convex-decomposition collider on the visual mesh,
                    DISABLE the primitives
  feet            : LEFT ALONE ON PURPOSE. The real Go2 foot is a rubber
                    hemisphere and a Sphere primitive is analytically exact for
                    it -- cheaper and better-conditioned than a faceted hull.
                    Meshifying it would be strictly worse.
  heads           : left alone (cosmetic, no contact role)

No CoACD needed: PhysX decomposes at load time from physics:approximation,
which is what the collision options in the USD are for.

    python usd_convert/patch_go2_mesh_colliders.py            # report
    python usd_convert/patch_go2_mesh_colliders.py --apply
"""

import argparse
import shutil
from pathlib import Path

from pxr import Sdf, Usd, UsdPhysics

USD = Path("protomotions/data/assets/usd/go2/Props/instanceable_meshes.usd")

# Links whose visual mesh should be the collider.
MESH_LINKS = ("thigh", "calf", "hip")
BASE_LINK = "base"
# Never touched -- see module docstring.
KEEP_PRIMITIVES = ("foot",)

# Fixed geometry with no mesh of its own whose shape IS already covered by
# another link's mesh. The go2's Head_lower/Head_upper are not robot bodies at
# all (absent from kinematic_info.body_names) -- the converter split the MJCF's
# base_link sensor-housing cylinder (x=+0.285) and sphere (x=+0.293) into their
# own Xforms. The base visual mesh spans x -0.128..+0.332, so it already
# contains that housing and decomposes it properly. Leaving these enabled would
# re-introduce exactly the primitive/mesh duplication this script removes.
DISABLE_ONLY = ("Head_lower", "Head_upper")

PHYSX_DECOMP_APIS = [
    "PhysicsCollisionAPI",
    "PhysxCollisionAPI",
    "PhysicsMeshCollisionAPI",
    "PhysxConvexDecompositionCollisionAPI",
]


def link_of(prim) -> str:
    parts = str(prim.GetPath()).split("/")
    return parts[2] if len(parts) > 2 else ""


def wants_mesh_collider(link: str) -> bool:
    if link in KEEP_PRIMITIVES:
        return False
    if link == BASE_LINK:
        return True
    return any(link.endswith(suffix) for suffix in MESH_LINKS)


def add_api_schemas(prim, names) -> None:
    """Append Physx API schema names via metadata.

    PhysxSchema's python bindings are absent outside a running Kit app, so the
    APIs are declared by editing the apiSchemas listOp directly. PhysX reads
    the composed metadata at load, so this is equivalent to Apply().
    """
    op = prim.GetMetadata("apiSchemas")
    existing = list(op.GetAddedOrExplicitItems()) if op else []
    merged = list(existing)
    for n in names:
        if n not in merged:
            merged.append(n)
    if merged != existing:
        prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(merged))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hulls", type=int, default=12,
                    help="physxConvexDecompositionCollision:maxConvexHulls")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not USD.exists():
        raise SystemExit(f"missing {USD}")

    stage = Usd.Stage.Open(str(USD))
    enable_mesh, disable_prim, bump_hulls = [], [], []

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        link = link_of(prim)
        if not link or link in KEEP_PRIMITIVES:
            continue

        is_mesh = prim.GetTypeName() == "Mesh"
        in_visuals = "/visuals/" in path
        in_collisions = "/collisions/" in path

        if link in DISABLE_ONLY:
            if in_collisions:
                disable_prim.append(prim)
        elif is_mesh and in_visuals and wants_mesh_collider(link):
            enable_mesh.append(prim)
        elif is_mesh and in_collisions and wants_mesh_collider(link):
            # e.g. base/collisions/Cylinder (convexHull) -- superseded by the
            # full-shell decomposition about to be enabled on the visual mesh.
            disable_prim.append(prim)
        elif in_collisions and wants_mesh_collider(link):
            disable_prim.append(prim)

    print(f"{USD}")
    print(f"  enable mesh collider (convexDecomposition, {args.max_hulls} hulls): {len(enable_mesh)}")
    for p in enable_mesh[:6]:
        print(f"      {p.GetPath()}")
    if len(enable_mesh) > 6:
        print(f"      ... +{len(enable_mesh)-6} more")
    print(f"  disable duplicate/primitive colliders: {len(disable_prim)}")
    for p in disable_prim[:6]:
        print(f"      {p.GetTypeName():9s} {p.GetPath()}")
    if len(disable_prim) > 6:
        print(f"      ... +{len(disable_prim)-6} more")
    print(f"  feet left on Sphere primitives (intentional -- analytically exact "
          f"for a rubber hemisphere)")

    if not args.apply:
        print("\n  (report only -- re-run with --apply)")
        return

    # Back up ONLY on the first run. Re-running --apply used to copy the
    # already-patched file over the backup, destroying the pristine original
    # (recovered 2026-09-02 from ProtoMotions-old; this repo is not under git,
    # so there was no other way back).
    backup = Path(str(USD) + ".bak_primitive_colliders")
    if backup.exists():
        print(f"  backup already exists, keeping it: {backup.name}")
    else:
        shutil.copy2(USD, backup)
        print(f"  wrote backup: {backup.name}")

    for prim in enable_mesh:
        add_api_schemas(prim, PHYSX_DECOMP_APIS)
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True, True)
        mesh_api = UsdPhysics.MeshCollisionAPI(prim)
        mesh_api.CreateApproximationAttr("convexDecomposition", True)
        attr = prim.CreateAttribute(
            "physxConvexDecompositionCollision:maxConvexHulls",
            Sdf.ValueTypeNames.Int,
        )
        attr.Set(args.max_hulls)

    for prim in disable_prim:
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False, True)

    # Existing mesh colliders (thigh/calf) also need the hull cap raised.
    for prim in stage.Traverse():
        a = prim.GetAttribute("physics:approximation")
        if a and a.IsValid() and a.Get() == "convexDecomposition":
            h = prim.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls")
            if h and h.IsValid() and h.Get() != args.max_hulls:
                h.Set(args.max_hulls)
                bump_hulls.append(prim)

    stage.GetRootLayer().Save()

    # Verify by REOPENING from disk -- an in-memory stage proves nothing.
    chk = Usd.Stage.Open(str(USD))
    mesh_on = prim_on = 0
    per_link = {}
    for prim in chk.Traverse():
        op = prim.GetMetadata("apiSchemas")
        names = list(op.GetAddedOrExplicitItems()) if op else []
        if not any("Collision" in s for s in names):
            continue
        ce = prim.GetAttribute("physics:collisionEnabled")
        if not (ce and ce.IsValid() and ce.Get()):
            continue
        link = link_of(prim)
        a = prim.GetAttribute("physics:approximation")
        kind = ("MESH:" + str(a.Get())) if prim.GetTypeName() == "Mesh" else "prim:" + prim.GetTypeName()
        per_link.setdefault(link, []).append(kind)
        if prim.GetTypeName() == "Mesh":
            mesh_on += 1
        else:
            prim_on += 1

    print(f"\n  VERIFIED ON RELOAD -- active colliders per link:")
    for k in sorted(per_link):
        print(f"      {k:14s} {per_link[k]}")
    print(f"  mesh colliders enabled: {mesh_on}   primitives still on: {prim_on}")
    print(f"  hull cap raised on {len(bump_hulls)} pre-existing mesh colliders")


if __name__ == "__main__":
    main()
