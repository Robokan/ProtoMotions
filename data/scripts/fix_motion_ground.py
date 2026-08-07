# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Lift clips whose feet sink through the floor.

A retarget is done against the collider radii of the day. Reshape the body
afterwards -- the tiger's legs went to 0.707x when its mass was refitted to
the mesh -- and every clip now plants its feet BELOW z=0, because the
retarget put the ankle joint where the old, fatter foot's surface touched
down. Measured on tiger_v5: 1.7 to 2.0 cm of penetration.

Contacts push the robot out again, so this is survivable, but it means every
reset starts inside the ground and the first physics step is a depenetration
impulse. That is a bad initial state for AMP -- the discriminator sees the
recovery, not the gait.

The correction is a PER-CLIP CONSTANT lift, chosen so the clip's single
deepest moment just clears the floor:

    dz = clearance - min over all frames and bodies of z

A constant offset is the only safe choice. A per-frame lift would flatten
the vertical motion of the gait: the body genuinely rises and falls during a
stride, and re-zeroing each frame would remove exactly the signal AMP is
trying to learn. It also keeps the clip internally consistent -- velocities
are unchanged by a constant translation, so nothing has to be recomputed.

Clips already clear of the floor are copied through untouched.

    python data/scripts/fix_motion_ground.py --robot tiger \
        --in-dir data/motions/tiger_v5 --clearance 0.005
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="tiger")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="default: edit in place, backing up to <in-dir>_pre_ground")
    ap.add_argument("--clearance", type=float, default=0.005,
                    help="height the deepest body should end up at")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", type=int, default=12)
    ap.add_argument("--stride", type=int, default=3,
                    help="sample every Nth frame when measuring surfaces")
    args = ap.parse_args()

    out_dir = args.out_dir or args.in_dir
    backup = f"{args.in_dir}_pre_ground" if args.out_dir is None else None
    if backup and not args.dry_run:
        os.makedirs(backup, exist_ok=True)
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)

    # Measure the lowest COLLIDER SURFACE, not the lowest body origin.
    # rigid_body_pos holds joint centres; a toe's origin can sit above the
    # floor while its capsule -- and the skin around it -- reach well below.
    # Measured on tiger walk clips: origins at +0.50 cm, surfaces at -3.50 cm,
    # the offender being RigLFLegDigit32, a toe tip. Fixing to the origin left
    # the feet visibly buried.
    import mujoco
    mj = mujoco.MjModel.from_xml_path(
        f"protomotions/data/assets/mjcf/{args.robot}.xml")
    md = mujoco.MjData(mj)

    def lowest_surface(dof_row, root_pos, root_rot_xyzw):
        md.qpos[:3] = root_pos
        md.qpos[3:7] = root_rot_xyzw[[3, 0, 1, 2]]     # xyzw -> wxyz
        md.qpos[7:] = dof_row
        mujoco.mj_forward(mj, md)
        lo = 1e9
        for g in range(mj.ngeom):
            r = mj.geom_size[g, 0]
            z = md.geom_xpos[g][2]
            if mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                ax = md.geom_xmat[g].reshape(3, 3)[:, 2]
                s = z - (abs(ax[2]) * mj.geom_size[g, 1] + r)
            else:
                s = z - r
            lo = min(lo, s)
        return lo

    rows = []
    for path in sorted(glob.glob(f"{args.in_dir}/*.motion")):
        name = os.path.basename(path)
        d = torch.load(path, weights_only=False, map_location="cpu")
        pos = d["rigid_body_pos"]
        dof = d["dof_pos"].numpy()
        rp = pos.numpy()
        rr = d["rigid_body_rot"].numpy()
        low = min(lowest_surface(dof[t], rp[t, 0], rr[t, 0])
                  for t in range(0, len(dof), args.stride))
        dz = args.clearance - low
        if dz <= 1e-6:                      # already clear
            rows.append((name, low, 0.0))
            if out_dir != args.in_dir and not args.dry_run:
                shutil.copy(path, f"{out_dir}/{name}")
            continue
        if not args.dry_run:
            if backup:
                shutil.copy(path, f"{backup}/{name}")
            pos = pos.clone()
            pos[:, :, 2] += dz
            d["rigid_body_pos"] = pos
            # A constant translation leaves every velocity untouched, so
            # rigid_body_vel / ang_vel / dof_* stay valid as they are.
            torch.save(d, f"{out_dir}/{name}")
        rows.append((name, low, dz))

    lifted = [r for r in rows if r[2] > 0]
    lifted.sort(key=lambda r: -r[2])
    print(f"scanned {len(rows)} clips; lifted {len(lifted)}")
    if lifted:
        print(f"{'clip':<44}{'was':>9}{'lift':>9}")
        for name, low, dz in lifted[:args.report]:
            print(f"  {name:<42}{low*100:+8.1f}c{dz*100:+8.1f}c")
        if len(lifted) > args.report:
            print(f"  ... and {len(lifted)-args.report} more")
    if args.dry_run:
        print("dry run: nothing written")
    else:
        print(f"written to {out_dir}" + (f"; backups in {backup}" if backup else ""))


main()
