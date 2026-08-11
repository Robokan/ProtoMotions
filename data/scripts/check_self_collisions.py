# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Self-collision audit of a retargeted motion corpus.

Eric saw thighs striking each other in some walk clips. Retargeting matches
body positions and orientations; nothing in the IK objective knows the robot's
own collision geometry, so a source skeleton with narrower hips than the robot
produces a scissoring gait that puts the thighs inside one another. The AMP
discriminator then treats those frames as reference, i.e. it teaches the policy
a pose the physics engine will fight.

Replays every frame through MuJoCo and reports contacts between bodies that
should not touch, aggregated by body pair and by clip.

WHAT COUNTS AS A REAL HIT. MuJoCo already filters two categories for us:
geoms in the same body, and geoms in bodies connected by a joint
(filterparent, on by default) -- so thigh-vs-pelvis does not register just
because their capsules overlap at the hip. Sibling limbs like left vs right
thigh are NOT filtered, which is exactly the case of interest. Floor contacts
are dropped here (those are the point of a walk clip), as are contacts within
--depth of touching, since limbs legitimately brush in a narrow gait; only
penetration past that depth is reported.

CAVEAT: this reads the MJCF, while IsaacLab trains from the USD. They are
generated from the same source, but if the USD was rebuilt from a different
MJCF revision the geometry can drift -- see the --output-dir trap in
convert_robot_mjcf_to_usda. Treat depths as accurate for the MJCF and
indicative for training.

    python data/scripts/check_self_collisions.py --robot atlas \\
        --in-dir data/motions/atlas_v11 --depth 0.005
"""
from __future__ import annotations

import argparse
import collections
import glob
import os

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="atlas")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--depth", type=float, default=0.005,
                    help="report penetration deeper than this (m); limbs "
                         "legitimately brush, so 0 would be all noise")
    ap.add_argument("--limit", type=int, default=None, help="first N clips")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--worst-frames", type=int, default=0,
                    help="also list this many worst individual frames")
    ap.add_argument("--pattern", default=None,
                    help="only clips whose filename contains this substring")
    ap.add_argument("--include-structural", action="store_true",
                    help="also report pairs that already overlap at the rest "
                         "pose (model-geometry defects, not motion defects)")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, ".")
    import mujoco
    from protomotions.robot_configs.factory import robot_config

    rc = robot_config(args.robot)
    body_names = list(rc.kinematic_info.body_names)
    m = mujoco.MjModel.from_xml_path(
        f"protomotions/data/assets/mjcf/{args.robot}.xml")
    d = mujoco.MjData(m)
    mj_bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                 for i in range(1, m.nbody)]
    if mj_bodies != body_names:
        raise SystemExit("MJCF body order != robot config body order")

    # geoms belonging to the world (floor, terrain) -- their contacts are the
    # whole point of a locomotion clip, not a defect
    world_geoms = {gi for gi in range(m.ngeom) if m.geom_bodyid[gi] == 0}
    print(f"{m.ngeom} geoms, {len(world_geoms)} on the world body "
          f"(floor contacts ignored)")
    print(f"contact excludes in the MJCF: {m.nexclude}  "
          f"(MuJoCo also auto-filters same-body and parent-child pairs)")

    # Pairs that interpenetrate in the DEFAULT pose are a property of the
    # collision model, not of any clip -- e.g. atlas overlaps Foot_* with
    # Leg4_* by 6.6 cm at rest because the shin capsule runs into the foot box
    # and they are a GRANDPARENT pair (Foot <- Ankle <- Leg4), which
    # filterparent does not exclude. Left in, they flag 100% of frames and bury
    # the real findings, so they are measured once and skipped.
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    structural = {}
    for ci in range(d.ncon):
        c = d.contact[ci]
        if c.geom1 in world_geoms or c.geom2 in world_geoms:
            continue
        k = tuple(sorted((mj_bodies[m.geom_bodyid[c.geom1] - 1],
                          mj_bodies[m.geom_bodyid[c.geom2] - 1])))
        structural[k] = min(structural.get(k, 0.0), float(c.dist))
    if structural:
        print(f"\nSTRUCTURAL overlaps at the rest pose "
              f"({'reported below' if args.include_structural else 'SKIPPED'}) "
              f"-- these are collision-model defects, fix in the MJCF:")
        for k, v in sorted(structural.items(), key=lambda kv: kv[1]):
            print(f"  {k[0]+' <-> '+k[1]:<44} {-v*100:6.2f} cm at rest")
    skip = set() if args.include_structural else set(structural)

    paths = sorted(glob.glob(f"{args.in_dir}/*.motion"))
    if args.pattern:
        paths = [p for p in paths if args.pattern in os.path.basename(p)]
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no .motion files in {args.in_dir}")

    pair_count = collections.Counter()
    pair_depth = collections.defaultdict(float)
    pair_worst = {}
    clip_frames = collections.Counter()
    clip_depth = collections.defaultdict(float)
    worst = []
    n_frames = n_hit_frames = 0

    for path in paths:
        name = os.path.basename(path)
        mo = torch.load(path, weights_only=False, map_location="cpu")
        pos = mo["rigid_body_pos"].numpy().astype(np.float64)
        rot = mo["rigid_body_rot"].numpy().astype(np.float64)
        dof = mo["dof_pos"].numpy().astype(np.float64)
        T = pos.shape[0]
        for t in range(T):
            d.qpos[:3] = pos[t, 0]
            d.qpos[3:7] = rot[t, 0][[3, 0, 1, 2]]   # xyzw -> wxyz
            d.qpos[7:] = dof[t]
            mujoco.mj_forward(m, d)
            n_frames += 1
            hit = False
            for ci in range(d.ncon):
                c = d.contact[ci]
                if c.geom1 in world_geoms or c.geom2 in world_geoms:
                    continue
                if c.dist >= -args.depth:
                    continue
                b1 = mj_bodies[m.geom_bodyid[c.geom1] - 1]
                b2 = mj_bodies[m.geom_bodyid[c.geom2] - 1]
                key = tuple(sorted((b1, b2)))
                if key in skip:
                    continue
                pair_count[key] += 1
                if -c.dist > pair_depth[key]:
                    pair_depth[key] = -c.dist
                    pair_worst[key] = (name, t)
                if -c.dist > clip_depth[name]:
                    clip_depth[name] = -c.dist
                hit = True
                worst.append((-c.dist, name, t, key))
            if hit:
                n_hit_frames += 1
                clip_frames[name] += 1
        print(f"  {name[:56]:<56} {clip_frames[name]:4d}/{T:4d} frames  "
              f"worst {clip_depth[name]*100:5.2f} cm", flush=True)

    print(f"\n{len(paths)} clips, {n_frames} frames")
    print(f"frames with self-penetration deeper than "
          f"{args.depth*100:.1f} cm: {n_hit_frames} "
          f"({100*n_hit_frames/max(n_frames,1):.2f}%)")

    if not pair_count:
        print("\nno self-collisions past the threshold")
        return

    print(f"\nBODY PAIRS (top {args.top} by frame count):")
    print(f"  {'pair':<44} {'frames':>7}  {'worst':>8}   worst clip")
    for key, n in pair_count.most_common(args.top):
        cn, ct = pair_worst[key]
        print(f"  {key[0]+' <-> '+key[1]:<44} {n:7d}  "
              f"{pair_depth[key]*100:6.2f} cm   {cn[:38]} @{ct}")

    print(f"\nCLIPS (top {args.top} by affected frames):")
    for name, n in clip_frames.most_common(args.top):
        print(f"  {name[:60]:<60} {n:5d} frames  worst {clip_depth[name]*100:5.2f} cm")

    if args.worst_frames:
        print(f"\nWORST {args.worst_frames} INDIVIDUAL FRAMES:")
        for dep, name, t, key in sorted(worst, reverse=True)[:args.worst_frames]:
            print(f"  {dep*100:6.2f} cm  {key[0]} <-> {key[1]:<22} {name[:44]} frame {t}")


if __name__ == "__main__":
    main()
