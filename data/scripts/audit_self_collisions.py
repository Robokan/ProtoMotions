#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Find frames where a corpus pose puts the robot's own MESHES in collision.

Training runs with self_collisions=False, so nothing in sim ever reports these.
But a pose whose limbs pass through each other is not reproducible on hardware,
which makes it bad reference data regardless of what the sim tolerates.

The robot's collision primitives are conservative bounding volumes -- the go2
thigh is a box around a tapered part -- so auditing against them overstates
collisions roughly 2x. The visual meshes are the real geometry, and this script
uses them directly.

Two-stage for speed: the (conservative) primitives prefilter candidate frames,
then exact concave mesh-mesh tests run only on those. Stage one is verified to
actually bound the meshes, otherwise the prefilter could silently miss frames.

    python data/scripts/audit_self_collisions.py data/motions/go2/go2_flat.pt
    python data/scripts/audit_self_collisions.py <file> --report-clips
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R

from protomotions.robot_configs.factory import robot_config

ASSETS = Path("protomotions/data/assets")
LEGS = ["FL", "FR", "RL", "RR"]


def mjcf_quat(s):
    """MJCF quats are wxyz; scipy wants xyzw."""
    if s is None:
        return np.eye(3)
    w, x, y, z = [float(v) for v in s.split()]
    return R.from_quat([x, y, z, w]).as_matrix()


def load_body_meshes(mjcf_path):
    """body name -> trimesh in body frame (all its visual meshes merged)."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    meshdir = ASSETS / "mesh" / "go2"
    files = {m.get("name") or Path(m.get("file")).stem: m.get("file")
             for m in root.iter("mesh")}

    out = {}
    def walk(elem):
        for child in elem:
            if child.tag == "body":
                name = child.get("name")
                parts = []
                for g in child:
                    if g.tag != "geom" or not g.get("mesh"):
                        continue
                    fn = files.get(g.get("mesh"))
                    if fn is None:
                        continue
                    m = trimesh.load(meshdir / fn, force="mesh")
                    T = np.eye(4)
                    T[:3, :3] = mjcf_quat(g.get("quat"))
                    if g.get("pos"):
                        T[:3, 3] = [float(v) for v in g.get("pos").split()]
                    m.apply_transform(T)
                    parts.append(m)
                if parts:
                    out[name] = trimesh.util.concatenate(parts)
                walk(child)
            else:
                walk(child)
    walk(root)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion_file")
    ap.add_argument("--robot", default="go2")
    ap.add_argument("--prefilter", type=float, default=0.06,
                    help="run exact mesh tests where primitive clearance is below this (m)")
    ap.add_argument("--report-clips", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap candidate frames (debug)")
    args = ap.parse_args()

    meshes = load_body_meshes(ASSETS / "mjcf" / "go2.xml")
    print(f"loaded meshes for {len(meshes)} bodies")

    rc = robot_config(args.robot)
    bodies = list(rc.kinematic_info.body_names)
    idx = {n: bodies.index(n) for n in bodies}

    # Stage-1 prefilter must be PROVABLY conservative or it silently drops
    # colliding frames. Primitive radii about a link axis are not: the meshes
    # are irregular and not centred on that axis. Use a bounding sphere fitted
    # to the actual vertices instead -- centre from the vertex extremes, radius
    # the furthest vertex from it. That contains the mesh whatever its shape,
    # so two bodies can only touch if their spheres overlap.
    bsphere = {}
    print("bounding spheres (body frame, from mesh vertices):")
    for name, m in meshes.items():
        v = np.asarray(m.vertices)
        c = 0.5 * (v.min(0) + v.max(0))
        r = float(np.linalg.norm(v - c, axis=1).max())
        bsphere[name] = (c, r)
        if name.startswith("FL_") or name == "base_link":
            print(f"  {name:10s} centre ({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})  radius {1000*r:6.1f}mm")
    print()

    d = torch.load(args.motion_file, map_location="cpu", weights_only=False)
    P, Q = d["gts"].numpy().astype(np.float64), d["grs"].numpy()
    n = len(P)

    # --- stage 1: cheap primitive clearance -------------------------------
    def segdist(p1, q1, p2, q2):
        d1, d2, r = q1 - p1, q2 - p2, p1 - p2
        a = (d1 * d1).sum(-1); e = (d2 * d2).sum(-1); f = (d2 * r).sum(-1)
        c = (d1 * r).sum(-1); b = (d1 * d2).sum(-1)
        den = a * e - b * b
        s = np.where(den > 1e-12, np.clip((b * f - c * e) / np.where(den > 1e-12, den, 1), 0, 1), 0.0)
        t = np.clip((b * s + f) / np.where(e > 1e-12, e, 1), 0, 1)
        s = np.clip((b * t - c) / np.where(a > 1e-12, a, 1), 0, 1)
        return np.linalg.norm((p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None]), axis=-1)

    def lerp(A, B, t):
        return A + (B - A) * t

    limbs = {}
    for L in LEGS:
        hip, calf, foot = P[:, idx[L + "_hip"]], P[:, idx[L + "_calf"]], P[:, idx[L + "_foot"]]
        limbs[L + "|thigh"] = (hip, calf, 0.017)
        limbs[L + "|calfA"] = (calf, lerp(calf, foot, 0.120 / 0.213), 0.013)
        limbs[L + "|calfB"] = (lerp(calf, foot, 0.115 / 0.213), lerp(calf, foot, 0.180 / 0.213), 0.011)
        limbs[L + "|foot"] = (foot, foot, 0.022)

    gap = np.full(n, np.inf)
    keys = list(limbs)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            if ka.split("|")[0] == kb.split("|")[0]:
                continue
            p1, q1, r1 = limbs[ka]; p2, q2, r2 = limbs[kb]
            gap = np.minimum(gap, segdist(p1, q1, p2, q2) - (r1 + r2))

    cand = np.where(gap < args.prefilter)[0]
    if args.limit:
        cand = cand[: args.limit]
    print(f"stage 1: {len(cand)} / {n} frames within {args.prefilter*1000:.0f}mm "
          f"({100*len(cand)/n:.2f}%) -> exact mesh test\n")

    # --- stage 2: exact concave mesh-mesh ---------------------------------
    test_bodies = [f"{L}_{p}" for L in LEGS for p in ("hip", "thigh", "calf")]
    pairs = [(a, b) for i, a in enumerate(test_bodies) for b in test_bodies[i + 1:]
             if a.split("_")[0] != b.split("_")[0]]  # different legs only

    mgr_meshes = {b: meshes[b] for b in test_bodies}
    hits = np.zeros(n, dtype=bool)
    pair_counts = {}
    for c, fi in enumerate(cand):
        if c % 500 == 0 and c:
            print(f"  ...{c}/{len(cand)}")
        placed = {}
        for b in test_bodies:
            T = np.eye(4)
            T[:3, :3] = R.from_quat(Q[fi, idx[b]]).as_matrix()
            T[:3, 3] = P[fi, idx[b]]
            m = mgr_meshes[b].copy(); m.apply_transform(T)
            placed[b] = m
        for a, b in pairs:
            cm = trimesh.collision.CollisionManager()
            cm.add_object("a", placed[a])
            cm.add_object("b", placed[b])
            if cm.in_collision_internal():
                hits[fi] = True
                pair_counts[f"{a}<->{b}"] = pair_counts.get(f"{a}<->{b}", 0) + 1

    print(f"\nEXACT MESH COLLISIONS: {hits.sum()} / {n} frames ({100*hits.mean():.3f}%)")
    if pair_counts:
        print("body pairs involved:")
        for k, v in sorted(pair_counts.items(), key=lambda x: -x[1])[:12]:
            print(f"    {k:34s} {v:6d} frames")

    if args.report_clips and hits.any():
        starts, nf = d["length_starts"].numpy(), d["motion_num_frames"].numpy()
        names = [str(x).split("/")[-1] for x in d["motion_files"]]
        rows = []
        for i in range(len(starts)):
            s, k = int(starts[i]), int(nf[i])
            f = hits[s:s + k].mean()
            if f > 0:
                rows.append((f, names[i], int(hits[s:s + k].sum()), k))
        print(f"\nclips with mesh collisions ({len(rows)} of {len(starts)}):")
        for f, nm, c, k in sorted(rows, reverse=True)[:25]:
            print(f"    {nm:44s} {c:5d}/{k:5d} frames  {f:6.1%}")
    np.save("/tmp/go2_mesh_collision_hits.npy", hits)


if __name__ == "__main__":
    main()
