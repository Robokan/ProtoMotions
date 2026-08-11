# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay a .motion clip in the MuJoCo viewer with contacts drawn.

Companion to check_self_collisions.py: that one aggregates, this one lets you
watch a specific frame range and judge how bad an interpenetration actually
looks before deciding whether to fix the data.

PURE KINEMATIC REPLAY -- qpos is set from the clip and mj_forward is called, so
collision detection runs but no physics is integrated. The robot therefore
passes through itself exactly as the corpus specifies, which is the point: this
shows what the AMP discriminator is being handed, not what the simulator would
do about it.

Contact points and forces are enabled, the contact markers are scaled up to be
visible at body scale, and colliding frames are printed to the console with
depths so the visual and the numbers can be checked against each other.

Structural overlaps (see check_self_collisions.py -- atlas has Foot_* inside its
own Leg4_* by 6.6 cm at rest, a grandparent pair filterparent misses) are
skipped by default, otherwise every frame draws contacts at both ankles.

    python data/scripts/view_motion_collisions.py --robot atlas \\
        --clip data/motions/atlas_v11/walk_ff_loop_180_R....motion \\
        --start 180 --end 245 --speed 0.25
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="atlas")
    ap.add_argument("--clip", required=True,
                    help="path to a .motion file, or a substring to search for")
    ap.add_argument("--in-dir", default="data/motions/atlas_v11")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--speed", type=float, default=0.25,
                    help="playback rate; 0.25 = quarter speed")
    ap.add_argument("--depth", type=float, default=0.005)
    ap.add_argument("--include-structural", action="store_true")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, ".")
    import mujoco
    import mujoco.viewer

    path = args.clip
    if not os.path.exists(path):
        hits = sorted(glob.glob(f"{args.in_dir}/*{args.clip}*.motion"))
        if not hits:
            raise SystemExit(f"no clip matching {args.clip!r} in {args.in_dir}")
        path = hits[0]
        if len(hits) > 1:
            print(f"{len(hits)} clips matched; using {os.path.basename(path)}")

    m = mujoco.MjModel.from_xml_path(
        f"protomotions/data/assets/mjcf/{args.robot}.xml")
    d = mujoco.MjData(m)
    mj_bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                 for i in range(1, m.nbody)]
    world_geoms = {gi for gi in range(m.ngeom) if m.geom_bodyid[gi] == 0}

    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    structural = set()
    for ci in range(d.ncon):
        c = d.contact[ci]
        if c.geom1 in world_geoms or c.geom2 in world_geoms:
            continue
        structural.add(tuple(sorted((mj_bodies[m.geom_bodyid[c.geom1] - 1],
                                     mj_bodies[m.geom_bodyid[c.geom2] - 1]))))
    skip = set() if args.include_structural else structural
    if skip:
        print(f"skipping structural pairs: {sorted(skip)}")

    mo = torch.load(path, weights_only=False, map_location="cpu")
    pos = mo["rigid_body_pos"].numpy().astype(np.float64)
    rot = mo["rigid_body_rot"].numpy().astype(np.float64)
    dof = mo["dof_pos"].numpy().astype(np.float64)
    fps = float(mo.get("fps", 30))
    T = pos.shape[0]
    a, b = max(0, args.start), min(T, args.end if args.end is not None else T)
    print(f"{os.path.basename(path)}: frames {a}-{b} of {T} at {fps:g} fps, "
          f"{args.speed}x speed")

    # make contact markers big enough to see against a 1.5 m robot
    m.vis.scale.contactwidth = 0.08
    m.vis.scale.contactheight = 0.04
    m.vis.scale.forcewidth = 0.02

    def show(t):
        d.qpos[:3] = pos[t, 0]
        d.qpos[3:7] = rot[t, 0][[3, 0, 1, 2]]       # xyzw -> wxyz
        d.qpos[7:] = dof[t]
        mujoco.mj_forward(m, d)
        hits = []
        for ci in range(d.ncon):
            c = d.contact[ci]
            if c.geom1 in world_geoms or c.geom2 in world_geoms:
                continue
            if c.dist >= -args.depth:
                continue
            k = tuple(sorted((mj_bodies[m.geom_bodyid[c.geom1] - 1],
                              mj_bodies[m.geom_bodyid[c.geom2] - 1])))
            if k in skip:
                continue
            hits.append((-c.dist, k))
        return hits

    with mujoco.viewer.launch_passive(m, d) as v:
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        # transparent bodies so interpenetration is visible from outside
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        print("viewer open -- Ctrl-C here to stop; looping the range")
        while v.is_running():
            for t in range(a, b):
                if not v.is_running():
                    break
                hits = show(t)
                v.sync()
                if hits:
                    worst = max(hits)
                    print(f"  frame {t:4d}  {len(hits)} contact(s)  worst "
                          f"{worst[0]*100:5.2f} cm  {worst[1][0]} <-> {worst[1][1]}",
                          flush=True)
                time.sleep(1.0 / (fps * max(args.speed, 1e-3)))


if __name__ == "__main__":
    main()
