# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Foot ground/pitch correction applied to SOURCE .motion clips.

Same correction as fix_foot_ground_pitch.py (read its docstring for the
diagnosis and why the lift must precede the pitch), but operating on the
per-clip .motion files instead of a packaged corpus — because
``motion_lib`` COPIES velocity fields rather than deriving them
(MOTION_KEYS maps gvs <- rigid_body_vel, dvs <- dof_vel, ...), so fixing a
packaged corpus leaves velocities describing the OLD ankle angles. The AMP
discriminator reads velocities, so that inconsistency would be trained on.

VELOCITIES ARE RECOMPUTED SURGICALLY, not wholesale. The correction is a
pure rotation of the ankle about its own origin plus a constant per-clip
lift, so exactly two things change and nothing else may be touched:

  * dof_vel of the ankle pitch DOFs        -> central differences
  * rigid_body_ang_vel of Ankle_*/Foot_*   -> from the corrected quats

Body ORIGINS do not move (Ankle and Foot share an origin with the joint
anchor, so pitching rotates them in place), and a constant lift has zero
derivative, so rigid_body_vel is provably unchanged. Recomputing every
velocity by finite differences would instead replace the retarget's own
(smoother) velocities everywhere with numerical ones — a silent
regression on 30 untouched DOFs to fix 2.

    python data/scripts/fix_foot_clips.py --robot atlas \\
        --in-dir data/motions/atlas_v11 --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np
import torch

from fix_foot_ground_pitch import _ang_vel_from_quats, _hull_vertices, _rot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="atlas")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="default: in place, backing up to <in-dir>_pre_footfix")
    ap.add_argument("--clearance", type=float, default=0.002)
    ap.add_argument("--max-pitch-deg", type=float, default=40.0)
    ap.add_argument("--limit", type=int, default=None, help="first N clips only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, ".")
    import mujoco
    from protomotions.robot_configs.factory import robot_config

    rc = robot_config(args.robot)
    body_names = list(rc.kinematic_info.body_names)
    dof_names = list(rc.control.control_info.keys())
    m = mujoco.MjModel.from_xml_path(
        f"protomotions/data/assets/mjcf/{args.robot}.xml")
    d = mujoco.MjData(m)
    mj_bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                 for i in range(1, m.nbody)]
    if mj_bodies != body_names:
        raise SystemExit("MJCF body order != robot config body order")

    feet, touched_bodies = {}, set()
    for gi in range(m.ngeom):
        b = mj_bodies[m.geom_bodyid[gi] - 1]
        if m.geom_contype[gi] == 0 or "foot" not in b.lower():
            continue
        mid = m.geom_dataid[gi]
        if mid < 0:
            continue
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        V = m.mesh_vert[va:va + vn].copy()
        pitch = next((j for j in dof_names
                      if j.startswith(b) and "pitch" in j.lower()), None)
        if pitch is None:
            continue
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, pitch)
        feet[b] = dict(geom=gi, V=V, hull=_hull_vertices(V),
                       dof=dof_names.index(pitch), jid=jid)
        # bodies whose ORIENTATION the pitch joint moves: the joint's own
        # body and every descendant of it
        jb = m.jnt_bodyid[jid]
        for i in range(1, m.nbody):
            p = i
            while p > 0:
                if p == jb:
                    touched_bodies.add(i - 1)
                    break
                p = m.body_parentid[p]
    if not feet:
        raise SystemExit("no foot collision geoms with a pitch dof")
    print(f"feet: {list(feet)}")
    print(f"bodies whose rotation changes: "
          f"{[body_names[i] for i in sorted(touched_bodies)]}")

    deltas = np.radians(np.arange(-args.max_pitch_deg,
                                  args.max_pitch_deg + 0.25, 0.25))
    order = np.argsort(np.abs(deltas))
    paths = sorted(glob.glob(f"{args.in_dir}/*.motion"))
    if args.limit:
        paths = paths[:args.limit]
    out_dir = args.out_dir or args.in_dir
    backup = f"{args.in_dir}_pre_footfix" if args.out_dir is None else None
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
        if backup:
            os.makedirs(backup, exist_ok=True)

    tot = dict(clips=0, frames=0, corrected=0, unfixable=0,
               before=0.0, after=0.0, maxdelta=0.0)
    tot["before"] = tot["after"] = 1e9
    for path in paths:
        name = os.path.basename(path)
        mo = torch.load(path, weights_only=False, map_location="cpu")
        pos = mo["rigid_body_pos"].numpy().astype(np.float64).copy()
        rot = mo["rigid_body_rot"].numpy().astype(np.float64).copy()
        dof = mo["dof_pos"].numpy().astype(np.float64).copy()
        dvel = mo["dof_vel"].numpy().astype(np.float64).copy()
        avel = mo["rigid_body_ang_vel"].numpy().astype(np.float64).copy()
        dt = 1.0 / float(mo.get("fps", 30))
        T = pos.shape[0]

        def lows(t):
            d.qpos[:3] = pos[t, 0]
            d.qpos[3:7] = rot[t, 0][[3, 0, 1, 2]]
            d.qpos[7:] = dof[t]
            mujoco.mj_forward(m, d)
            out = {}
            for b, f in feet.items():
                gi = f["geom"]
                R = d.geom_xmat[gi].reshape(3, 3)
                p = d.geom_xpos[gi].copy()
                cur = float((f["hull"] @ R.T + p)[:, 2].min())
                ax = d.xaxis[f["jid"]].copy()
                an = d.xanchor[f["jid"]].copy()
                Rk = _rot(ax, deltas)
                pk = an + np.einsum("kij,j->ki", Rk, p - an)
                Rw = np.einsum("kij,jl->kil", Rk, R)
                wk = np.einsum("vj,kij->kvi", f["hull"], Rw) + pk[:, None, :]
                mins = wk[:, :, 2].min(axis=1)
                out[b] = (cur, mins)
            return out

        # pass 1: the height error rotation cannot fix -> one lift per clip
        best = np.array([min(v[1].max() for v in lows(t).values())
                         for t in range(T)])
        cur0 = np.array([min(v[0] for v in lows(t).values()) for t in range(T)])
        dz = max(0.0, args.clearance - float(best.min()))
        pos[:, :, 2] += dz

        # pass 2: per-frame ankle pitch
        n_corr = 0
        for t in range(T):
            fl = lows(t)
            hit = False
            for b, f in feet.items():
                cur, mins = fl[b]
                if cur >= args.clearance:
                    continue
                ok = [k for k in order if mins[k] >= args.clearance]
                if not ok:
                    tot["unfixable"] += 1
                    continue
                delta = float(deltas[ok[0]])
                lo, hi = m.jnt_range[f["jid"]]
                dof[t, f["dof"]] = float(np.clip(dof[t, f["dof"]] + delta, lo, hi))
                tot["maxdelta"] = max(tot["maxdelta"], abs(delta))
                hit = True
            if hit:
                n_corr += 1
                d.qpos[:3] = pos[t, 0]
                d.qpos[3:7] = rot[t, 0][[3, 0, 1, 2]]
                d.qpos[7:] = dof[t]
                mujoco.mj_forward(m, d)
                rot[t] = d.xquat[1:][:, [1, 2, 3, 0]]      # wxyz -> xyzw

        # verify on the FULL mesh
        after = np.zeros(T)
        for t in range(T):
            d.qpos[:3] = pos[t, 0]
            d.qpos[3:7] = rot[t, 0][[3, 0, 1, 2]]
            d.qpos[7:] = dof[t]
            mujoco.mj_forward(m, d)
            after[t] = min(
                float((f["V"] @ d.geom_xmat[f["geom"]].reshape(3, 3).T
                       + d.geom_xpos[f["geom"]])[:, 2].min())
                for f in feet.values())

        # surgical velocity update (see module docstring)
        for f in feet.values():
            j = f["dof"]
            dvel[:, j] = np.gradient(dof[:, j], dt)
        for bi in sorted(touched_bodies):
            # [T, B, 3]: index the BODY axis, not the frame axis
            avel[:, bi] = _ang_vel_from_quats(rot[:, bi], dt)

        tot["clips"] += 1
        tot["frames"] += T
        tot["corrected"] += n_corr
        tot["before"] = min(tot["before"], float(cur0.min()))
        tot["after"] = min(tot["after"], float(after.min()))
        if not args.dry_run:
            if backup:
                shutil.copy(path, f"{backup}/{name}")
            mo["rigid_body_pos"] = torch.from_numpy(pos).float()
            mo["rigid_body_rot"] = torch.from_numpy(rot).float()
            mo["dof_pos"] = torch.from_numpy(dof).float()
            mo["dof_vel"] = torch.from_numpy(dvel).float()
            mo["rigid_body_ang_vel"] = torch.from_numpy(avel).float()
            torch.save(mo, f"{out_dir}/{name}")
        print(f"  {name[:54]:<54} lift {dz*100:5.2f} cm  "
              f"pitch-fixed {n_corr:4d}/{T:4d}  "
              f"low {cur0.min()*100:+6.2f} -> {after.min()*100:+6.2f} cm",
              flush=True)

    print(f"\n{tot['clips']} clips, {tot['frames']} frames")
    print(f"  frames pitch-corrected {tot['corrected']} "
          f"({100*tot['corrected']/max(tot['frames'],1):.1f}%)")
    print(f"  largest correction     {np.degrees(tot['maxdelta']):.1f} deg")
    print(f"  unfixable feet         {tot['unfixable']}")
    print(f"  deepest point          {tot['before']*100:+.2f} -> "
          f"{tot['after']*100:+.2f} cm")
    if args.dry_run:
        print("dry run: nothing written")
    else:
        print(f"written to {out_dir}" + (f"; backups in {backup}" if backup else ""))


main()
