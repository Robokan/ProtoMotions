# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Re-index .motion clips after bodies are removed from a robot.

A .motion stores per-frame tensors sized to the robot that produced it --
rigid_body_pos is [T, num_bodies, 3] and dof_pos is [T, num_dofs]. Delete
a body from the MJCF and every existing clip becomes silently
incompatible: the shapes no longer line up, and if they happen to line up
anyway the columns mean different joints.

Retargeting the whole corpus again would cost hours. But when the new
skeleton is a strict SUBSET of the old one in the SAME order -- which is
the case when you delete leaf marker bones -- the conversion is exact and
free: drop the removed columns, keep everything else untouched. This
script verifies the subset property first and refuses to run otherwise,
because a silent mis-mapping here would corrupt the corpus in a way that
only shows up as a policy that cannot learn.

    python data/scripts/remap_motion_bodies.py \
        --old-mjcf /tmp/raptor_pre_head.xml \
        --new-mjcf protomotions/data/assets/mjcf/raptor.xml \
        --in-dir data/motions/raptor_v5 --backup-dir data/motions/raptor_v5_pre_remap
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import torch

# [T, num_bodies, ...] tensors
_BODY_KEYS = ("rigid_body_pos", "rigid_body_rot", "rigid_body_vel",
              "rigid_body_ang_vel", "rigid_body_contacts")
# [T, num_dofs] tensors
_DOF_KEYS = ("dof_pos", "dof_vel")


def skeleton(path: str):
    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
              for i in range(1, m.nbody)]
    joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(m.njnt)]
    joints = [j for j in joints if j and not j.startswith("root")]
    return bodies, joints


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-mjcf", required=True)
    ap.add_argument("--new-mjcf", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--backup-dir", default=None)
    args = ap.parse_args()

    ob, oj = skeleton(args.old_mjcf)
    nb, nj = skeleton(args.new_mjcf)

    # Refuse anything that is not a same-order subset. Reordering or adding
    # bodies cannot be fixed by dropping columns and needs a real retarget.
    if nb != [x for x in ob if x in nb]:
        raise SystemExit("new bodies are not a same-order subset of the old "
                         "skeleton -- a column drop would mis-map them")
    if nj != [x for x in oj if x in nj]:
        raise SystemExit("new joints are not a same-order subset of the old "
                         "skeleton -- a column drop would mis-map them")

    body_keep = torch.tensor([i for i, x in enumerate(ob) if x in nb])
    dof_keep = torch.tensor([i for i, x in enumerate(oj) if x in nj])
    print(f"bodies {len(ob)} -> {len(nb)}  (dropping "
          f"{[x for x in ob if x not in nb]})")
    print(f"dofs   {len(oj)} -> {len(nj)}")

    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)

    paths = sorted(glob.glob(f"{args.in_dir}/*.motion"))
    done = skipped = 0
    for p in paths:
        d = torch.load(p, weights_only=False, map_location="cpu")
        if d["rigid_body_pos"].shape[1] == len(nb):
            skipped += 1
            continue
        if d["rigid_body_pos"].shape[1] != len(ob):
            raise SystemExit(
                f"{os.path.basename(p)} has {d['rigid_body_pos'].shape[1]} "
                f"bodies, expected {len(ob)} or {len(nb)}")
        if args.backup_dir:
            shutil.copy(p, f"{args.backup_dir}/{os.path.basename(p)}")
        for k in _BODY_KEYS:
            if k in d:
                d[k] = d[k].index_select(1, body_keep).contiguous()
        for k in _DOF_KEYS:
            if k in d:
                d[k] = d[k].index_select(1, dof_keep).contiguous()
        torch.save(d, p)
        done += 1

    print(f"remapped {done} clips, {skipped} already current"
          + (f", backups in {args.backup_dir}" if args.backup_dir else ""))


main()
