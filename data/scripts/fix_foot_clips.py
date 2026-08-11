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

VELOCITIES ARE UPDATED ADDITIVELY, not recomputed. Run AFTER
lowpass_motion_clips.py, whose whole point is that the tremor lives in the
velocity channels (dof_vel carries 21.0% of its energy above 8 Hz against
7.4% for dof_pos -- which is why playback looks clean while the AMP
discriminator sees a buzz). Overwriting dof_vel with np.gradient would throw
that filtering away on exactly the DOFs this script touches and substitute
finite-difference noise; an earlier version did, driving ankle dof_vel from
20.8 to 27.5 rad/s. Differentiation is linear, so adding d(correction)/dt
instead keeps positions and velocities consistent AND keeps the filter:

  * dof_vel of the ankle pitch DOF  += d(correction)/dt
  * ang_vel of that joint's subtree += d(correction)/dt * world joint axis

rigid_body_vel is untouched, provably: the ankle rotates about its own
origin so body origins do not move, and a constant per-clip lift has zero
derivative.

THE PITCH TRAJECTORY IS A SHORTEST PATH, NOT A PER-FRAME CHOICE. Picking
the smallest clearing angle at each frame independently is what caused the
leg vibration Eric reported. The feasible set is frequently two DISJOINT
bands (pitch the toe up, or rotate far the other way so the heel clears),
so neighbouring frames could sit in different bands: measured 110 deg of
correction inside a single frame (3300 deg/s) and ankle chatter 1.4x the
retarget's own, 3.6x on the worst clip. No amount of smoothing repairs it,
because under a hard per-frame clearance constraint no continuous path
exists.

So the priority is inverted: the frame-to-frame rate limit is the HARD
constraint (--max-rate-deg, default 10 deg/frame = 300 deg/s, within real
ankle capability) and penetration becomes a quadratic COST. A uniform delta
grid makes each step's predecessor set a fixed window, so the DP is a
sliding-window minimum per frame -- O(T*K) via minimum_filter1d. Isolated
frames that would need an impossible swing keep a few mm of penetration
instead, which is invisible and, unlike a 98 deg snap, is not something the
discriminator will reward reproducing.

Rate limit vs residual penetration, measured on the four worst clips:

    6 deg/frame (180 deg/s) -> deepest -1.84 cm
   10 deg/frame (300 deg/s) -> deepest -0.68 cm   <- default
   14 deg/frame (420 deg/s) -> deepest -0.46 cm

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
from scipy.ndimage import minimum_filter1d

from fix_foot_ground_pitch import _ang_vel_from_quats, _hull_vertices, _rot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="atlas")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="default: in place, backing up to <in-dir>_pre_footfix")
    ap.add_argument("--clearance", type=float, default=0.002)
    ap.add_argument("--max-pitch-deg", type=float, default=65.0)
    ap.add_argument("--max-rate-deg", type=float, default=10.0,
                    help="hard cap on ankle-pitch correction change per frame; "
                         "10 deg at 30 fps = 300 deg/s")
    ap.add_argument("--stay-weight", type=float, default=0.05,
                    help="preference for leaving the retarget angle alone, "
                         "relative to 1 cm^2 of penetration")
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
        sub = []
        for i in range(1, m.nbody):
            p = i
            while p > 0:
                if p == jb:
                    touched_bodies.add(i - 1)
                    sub.append(i - 1)
                    break
                p = m.body_parentid[p]
        feet[b]["subtree"] = sub
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
               before=0.0, after=0.0, maxdelta=0.0, maxjump=0.0)
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

        def sweep():
            """cur[t] and the achievable sole height for every grid delta."""
            cur = {b: np.zeros(T) for b in feet}
            mins = {b: np.zeros((T, len(deltas))) for b in feet}
            for t in range(T):
                for b, (c, mn) in lows(t).items():
                    cur[b][t] = c
                    mins[b][t] = mn
            return cur, mins

        # pass 1: the height error rotation cannot fix -> one lift per clip
        cur, mins = sweep()
        cur0 = np.min(np.stack([cur[b] for b in feet]), axis=0)
        best = np.min(np.stack([mins[b].max(axis=1) for b in feet]), axis=0)
        dz = max(0.0, args.clearance - float(best.min()))
        pos[:, :, 2] += dz

        # pass 2: ankle pitch, chosen as a SMOOTH TRAJECTORY through the
        # feasible set rather than per-frame independently.
        #
        # The old version took the smallest clearing angle at each frame in
        # isolation, which forced 0 on any already-clear frame while its
        # neighbour took 30-65 deg -- a discontinuity that showed up as ankle
        # chatter (1.4x on average, 3.6x worst clip) and as a 110 deg
        # correction inside a single frame. The grid search already knows the
        # FULL SET of clearing angles per frame, so the minimum is only one of
        # many admissible choices; picking the smoothest admissible sequence
        # costs nothing in clearance.
        #
        # Feasibility is an INTERVAL, not a lower bound: rotating further than
        # needed lifts the toe but drops the heel, so both ends bind. Each
        # frame contributes [lo, hi] (the contiguous clearing run nearest zero,
        # intersected with the joint range) and a projected-smoothing loop
        # alternates Gaussian smoothing with clamping back into those
        # intervals. Frames with the foot well clear have wide intervals and so
        # absorb most of the smoothing; frames genuinely pinned by geometry
        # keep their required angle.
        # SHORTEST PATH over the delta grid, with the frame rate limit as a
        # HARD constraint and ground penetration as a COST.
        #
        # Treating clearance as a hard per-frame constraint is what produced
        # the vibration. The feasible set is frequently two DISJOINT bands
        # (pitch the toe up, or rotate far the other way so the heel clears),
        # and demanding clearance every frame lets consecutive frames sit in
        # different bands -- measured as a 98 deg swing inside one frame
        # (2900 deg/s, physically impossible for an ankle). Smoothing cannot
        # repair that: no continuous path exists inside the constraint.
        #
        # So the priority is inverted. Only trajectories changing by at most
        # --max-rate-deg per frame are admissible, and among those we minimise
        # penetration (quadratic, in cm) plus a small preference for leaving
        # the retarget alone. Isolated frames where clearing would require an
        # impossible swing now keep a few mm of penetration instead, which is
        # invisible and, unlike a 98 deg snap, is not something the AMP
        # discriminator will reward reproducing.
        #
        # With a uniform grid the rate limit makes each step's predecessor set
        # a fixed-width window, so the DP reduces to a sliding-window minimum
        # per frame -- O(T*K) via minimum_filter1d rather than O(T*K^2).
        cur, mins = sweep()
        applied = {}
        step = float(np.degrees(deltas[1] - deltas[0]))
        W = max(1, int(round(args.max_rate_deg / step)))     # states per frame
        for b, f in feet.items():
            jlo, jhi = m.jnt_range[f["jid"]]
            pen_cm = np.maximum(0.0, args.clearance - mins[b]) * 100.0
            E = pen_cm ** 2 + args.stay_weight * (np.degrees(deltas)[None, :] / 10.0) ** 2
            # a delta outside the joint range is not a choice at all
            bad = ((dof[:, f["dof"]][:, None] + deltas[None, :] < jlo)
                   | (dof[:, f["dof"]][:, None] + deltas[None, :] > jhi))
            E = np.where(bad, 1e12, E)
            tot["unfixable"] += int((mins[b] < args.clearance).all(axis=1).sum())

            D = E[0].copy()
            for t in range(1, T):
                D = E[t] + minimum_filter1d(D, 2 * W + 1, mode="nearest")
            k = int(np.argmin(D))
            # backtrack: recompute the forward table cheaply in reverse by
            # re-deriving each predecessor inside its window
            path = np.empty(T, dtype=int)
            path[-1] = k
            Ds = [E[0].copy()]
            for t in range(1, T):
                Ds.append(E[t] + minimum_filter1d(Ds[-1], 2 * W + 1, mode="nearest"))
            for t in range(T - 1, 0, -1):
                s, e = max(0, path[t] - W), min(len(deltas), path[t] + W + 1)
                path[t - 1] = s + int(np.argmin(Ds[t - 1][s:e]))
            applied[b] = deltas[path]
            x = applied[b]
            dof[:, f["dof"]] += x
            tot["maxdelta"] = max(tot["maxdelta"], float(np.abs(x).max()))
            tot["maxjump"] = max(tot["maxjump"],
                                 float(np.abs(np.diff(x)).max()) if T > 1 else 0.0)

        n_corr = int((np.abs(np.stack([applied[b] for b in feet])) > 1e-9)
                     .any(axis=0).sum())
        axw = {b: np.zeros((T, 3)) for b in feet}
        for t in range(T):
            d.qpos[:3] = pos[t, 0]
            d.qpos[3:7] = rot[t, 0][[3, 0, 1, 2]]
            d.qpos[7:] = dof[t]
            mujoco.mj_forward(m, d)
            rot[t] = d.xquat[1:][:, [1, 2, 3, 0]]      # wxyz -> xyzw
            for b, f in feet.items():
                axw[b][t] = d.xaxis[f["jid"]]

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

        # Velocity update: ADD the correction's own derivative rather than
        # recomputing the channel. The input clips are already low-passed and
        # their velocities are consistent with their positions, so overwriting
        # with np.gradient would throw that filtering away on exactly the DOFs
        # we touch and substitute finite-difference noise -- the mistake that
        # drove ankle dof_vel from 20.8 to 27.5 rad/s. d/dt is linear, so
        # adding d(correction)/dt keeps consistency and preserves the filter.
        for b, f in feet.items():
            xdot = np.gradient(applied[b], dt)          # rad/s about the axis
            dvel[:, f["dof"]] += xdot
            for bi in f["subtree"]:
                avel[:, bi] += xdot[:, None] * axw[b]

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
    print(f"  largest 1-frame jump   {np.degrees(tot['maxjump']):.1f} deg")
    print(f"  unfixable feet         {tot['unfixable']}")
    print(f"  deepest point          {tot['before']*100:+.2f} -> "
          f"{tot['after']*100:+.2f} cm")
    if args.dry_run:
        print("dry run: nothing written")
    else:
        print(f"written to {out_dir}" + (f"; backups in {backup}" if backup else ""))


main()
