#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Convex-decompose the go2 visual meshes into collision hulls.

The stock go2 collides on primitives -- the thigh is a box around a tapered
part -- which overstates limb-to-limb contact roughly 2x. To find where a
corpus pose ACTUALLY self-intersects, the robot needs colliders that follow
the real geometry.

A single convex hull per link is no better than the box (it fills the concave
gap between thigh and calf). CoACD splits each link into several hulls that
together track the shape closely, and every physics engine handles convex
hulls natively.

    python data/scripts/decompose_go2_collision_meshes.py
    python data/scripts/decompose_go2_collision_meshes.py --threshold 0.03
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

MESH_DIR = Path("protomotions/data/assets/mesh/go2")
OUT_DIR = Path("protomotions/data/assets/mesh/go2_collide")

# Link -> visual meshes that make it up. Merged before decomposition so the
# hulls span the whole link rather than each cosmetic sub-part.
LINKS = {
    "base": ["base_0", "base_1", "base_2", "base_3", "base_4"],
    "hip": ["hip_0", "hip_1"],
    "thigh": ["thigh_0", "thigh_1"],
    "thigh_mirror": ["thigh_mirror_0", "thigh_mirror_1"],
    "calf": ["calf_0", "calf_1"],
    "calf_mirror": ["calf_mirror_0", "calf_mirror_1"],
    "foot": ["foot"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.04,
                    help="CoACD concavity threshold; lower = more hulls, tighter fit")
    ap.add_argument("--max-hulls", type=int, default=24)
    args = ap.parse_args()

    import coacd
    coacd.set_log_level("error")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'link':16s} {'tris':>7s} {'hulls':>6s} {'mesh vol':>10s} {'hull vol':>10s} {'ratio':>7s}")
    summary = {}
    for link, parts in LINKS.items():
        meshes = []
        for p in parts:
            f = MESH_DIR / f"{p}.stl"
            if not f.exists():
                f = MESH_DIR / f"{p}.obj"
            meshes.append(trimesh.load(f, force="mesh"))
        merged = trimesh.util.concatenate(meshes)

        m = coacd.Mesh(merged.vertices, merged.faces)
        parts_out = coacd.run_coacd(
            m, threshold=args.threshold, max_convex_hull=args.max_hulls,
            preprocess_mode="auto",
        )

        hulls = []
        for i, (v, f) in enumerate(parts_out):
            h = trimesh.Trimesh(vertices=np.asarray(v), faces=np.asarray(f))
            h.export(OUT_DIR / f"{link}_hull{i}.stl")
            hulls.append(h)

        # Hull volume should slightly EXCEED the mesh (hulls are supersets of
        # their region). A large ratio means the decomposition is filling
        # concavities and will over-report collisions.
        mv = float(merged.volume)
        hv = float(sum(h.volume for h in hulls))
        summary[link] = (len(hulls), mv, hv)
        print(f"  {link:14s} {len(merged.faces):7d} {len(hulls):6d} "
              f"{mv*1e6:9.1f}cc {hv*1e6:9.1f}cc {hv/max(mv,1e-9):6.2f}x")

    print(f"\nwrote {sum(v[0] for v in summary.values())} hull files to {OUT_DIR}")
    worst = max(summary.items(), key=lambda kv: kv[1][2] / max(kv[1][1], 1e-9))
    print(f"loosest fit: {worst[0]} at {worst[1][2]/max(worst[1][1],1e-9):.2f}x mesh volume")
    print("(1.0x = perfect; >1.3x means concavities are being filled -- lower --threshold)")


if __name__ == "__main__":
    main()
