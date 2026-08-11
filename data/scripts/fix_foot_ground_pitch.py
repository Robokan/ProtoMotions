# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Stop a rigid robot foot from driving its toe through the floor.

THE BUG THIS FIXES (measured on atlas_pretrain_corpus_v10, 2026-08-10):
93.4% of frames put a foot collider below the ground plane, 46.4% deeper
than 5 cm, worst -13.5 cm — and for every penetration past 3 cm the
deepest point is at the TOE (toe_frac median 0.99, i.e. the very tip).

The cause is a morphology mismatch that the retarget could not see. GMR's
Atlas config (soma_bvh_to_atlas.json) maps

    Foot_L: ['LeftFoot', pos_w=100, rot_w=10, pos_offset=[0,0,0], rot_q]

so the robot foot inherits the SOURCE ANKLE'S ORIENTATION. The mocap
solved a foot that effectively ends at the ball of the foot: during
push-off that segment pitches steeply while the toes stay flat on the
ground. Atlas has no toe joint, so the same rotation swings its whole
26.5 cm foot about the ankle and the tip goes straight down. The geometry
predicts the damage exactly: a rigid foot pitched 30 deg drops its toe
13.2 cm, and the measured worst frame is -13.5 cm at +45 deg of pitch.
Corpus foot pitch reaches +45.8 deg (L) / +51.8 deg (R) at p99.

There is a second, much smaller error in the same config: pos_offset is
[0,0,0], so the robot's ankle is placed at the HUMAN's ankle height
(~7 cm) while Atlas needs 8.82 cm from ankle to sole. That is a uniform
~1.8 cm sink, visible as a flat foot resting slightly under the floor.

TWO CORRECTIONS, IN THIS ORDER — the order matters, and it is not the
order you would first guess:

1. PER-CLIP CONSTANT LIFT, sized as ONLY WHAT ROTATION CANNOT FIX. For
   every frame the script first asks "what is the highest this sole could
   possibly sit if I were free to re-pitch the ankle?" The worst such value
   over a clip is the part of the sink that is a HEIGHT error (the 1.8 cm
   ankle deficit) rather than a rotation error, and that is the lift.

2. PER-FRAME ANKLE PITCH for the residual toe dive, rotating by the
   SMALLEST angle that lifts the deepest point back to `clearance`. Clean
   frames are left bit-identical; airborne feet are untouched.

Pitch-first was tried and provably cannot work: with the ankle pinned
1.8 cm too low, the best achievable rotation still leaves the sole at
-1.8 cm, so 28629 of 49862 feet had NO solution and the worst frame did
not improve at all (-13.47 cm before and after). No rotation about a
too-low pivot can lift a foot out of the floor.

Sizing the lift by max penetration would be the opposite error: it would
raise a whole clip ~13 cm to chase one toe spike and leave the robot
hovering through every flat-footed frame. Hence "what rotation cannot
fix" — it isolates the height component exactly.

WHAT IS AND IS NOT PRESERVED. Ankle pitch is the only DOF touched, and
only on offending frames; the ankle POSITION does not move (Ankle and Foot
share an origin, so this is a pure rotation about the ankle). Everything
above the ankle — root trajectory, spine, arms, knee, hip — is untouched,
so gait timing and style survive. The cost is that the ankle angle now
deviates from the mocap on push-off frames, which is exactly the deviation
required by a foot with no toe joint. Contact labels are left alone; they
were derived from the source and a pitch change does not move the contact
point in time.

    python data/scripts/fix_foot_ground_pitch.py --robot atlas \\
        --corpus data/atlas_pretrain_corpus_v10.pt --dry-run
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch


def _hull_vertices(V: np.ndarray, cap: int = 400) -> np.ndarray:
    """Candidate vertices for a lowest-point query.

    The world-lowest vertex of a rigid body is always on its convex hull,
    so the hull is a lossless reduction from ~4800 verts to a few hundred —
    which is what makes a per-frame angle search affordable.
    """
    try:
        from scipy.spatial import ConvexHull

        return V[np.unique(ConvexHull(V).vertices)]
    except Exception:
        # Fallback: keep the lower half plus fore/aft extremes. Not exact,
        # so verification below re-checks against the FULL mesh.
        keep = V[:, 1] < np.median(V[:, 1])
        idx = np.flatnonzero(keep)
        if len(idx) > cap:
            idx = idx[np.linspace(0, len(idx) - 1, cap).astype(int)]
        return V[idx]


def _rot(axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrices for a stack of angles -> [K,3,3]."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    c = np.cos(ang)[:, None, None]
    s = np.sin(ang)[:, None, None]
    return np.eye(3)[None] * c + s * K[None] + (1 - c) * np.outer(a, a)[None]


def _ang_vel_from_quats(q: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity from a quaternion track [T,4] (xyzw) -> [T,3].

    Central differences on the relative rotation: w = 2*vec(q_{t+1} q_t^-1)/dt,
    sign-corrected so the shortest arc is used (q and -q are the same
    rotation, and a sign flip in the track would otherwise read as a
    ~2/dt spike).
    """
    T = q.shape[0]
    qq = q.copy()
    flip = np.sum(qq[1:] * qq[:-1], axis=-1) < 0
    for t in range(1, T):                       # make the track continuous
        if flip[t - 1]:
            qq[t:] *= -1.0
            flip = np.sum(qq[1:] * qq[:-1], axis=-1) < 0
    x, y, z, w = qq[:, 0], qq[:, 1], qq[:, 2], qq[:, 3]
    conj = np.stack([-x, -y, -z, w], axis=-1)
    out = np.zeros((T, 3))
    for t in range(T):
        a = min(t + 1, T - 1)
        b = max(t - 1, 0)
        span = (a - b) * dt
        if span <= 0:
            continue
        q1, q0c = qq[a], conj[b]
        # Hamilton product q1 * conj(q0), xyzw layout
        vx = q1[3] * q0c[0] + q1[0] * q0c[3] + q1[1] * q0c[2] - q1[2] * q0c[1]
        vy = q1[3] * q0c[1] - q1[0] * q0c[2] + q1[1] * q0c[3] + q1[2] * q0c[0]
        vz = q1[3] * q0c[2] + q1[0] * q0c[1] - q1[1] * q0c[0] + q1[2] * q0c[3]
        vw = q1[3] * q0c[3] - q1[0] * q0c[0] - q1[1] * q0c[1] - q1[2] * q0c[2]
        s = -1.0 if vw < 0 else 1.0             # shortest arc
        out[t] = 2.0 * np.array([vx, vy, vz]) * s / span
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="atlas")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default=None, help="default: in place, with a .bak")
    ap.add_argument("--clearance", type=float, default=0.002,
                    help="height the deepest sole point should end up at (m)")
    ap.add_argument("--max-pitch-deg", type=float, default=40.0,
                    help="cap on the correction; beyond this the frame is "
                         "not a toe-rotation problem and the residual lift "
                         "handles it")
    ap.add_argument("--no-lift", action="store_true",
                    help="skip the per-clip uniform lift")
    ap.add_argument("--clips", type=int, default=None,
                    help="only process the first N clips — for iterating on "
                         "the method without paying for all 25k frames")
    ap.add_argument("--clip-match", default=None,
                    help="only clips whose filename contains this substring "
                         "(iteration aid; combine with --dry-run)")
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

    # Feet = bodies carrying a collision geom whose name looks like a foot.
    feet = {}
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
            print(f"  {b}: no pitch dof found, skipping")
            continue
        feet[b] = dict(geom=gi, V=V, hull=_hull_vertices(V),
                       dof=dof_names.index(pitch), dof_name=pitch,
                       jid=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, pitch))
    if not feet:
        raise SystemExit("no foot collision geoms with a pitch dof")
    for b, f in feet.items():
        print(f"  {b}: geom {f['geom']}, {len(f['V'])} verts "
              f"({len(f['hull'])} hull), pitch dof '{f['dof_name']}'")

    corpus = torch.load(args.corpus, weights_only=False, map_location="cpu")
    gts = corpus["gts"].numpy().copy()
    grs = corpus["grs"].numpy().copy()
    dps = corpus["dps"].numpy().copy()
    starts = corpus["length_starts"].numpy()
    nframes = corpus["motion_num_frames"].numpy()
    N = gts.shape[0]

    # --clips / --clip-match: restrict the working set. Frames outside it
    # are left untouched, so a partial run must never be written back.
    all_files = [str(x).split("/")[-1] for x in corpus["motion_files"]]
    sel = list(range(len(nframes)))
    if args.clip_match:
        sel = [c for c in sel if args.clip_match in all_files[c]]
    if args.clips is not None:
        sel = sel[:args.clips]
    partial = len(sel) != len(nframes)
    if partial:
        if not args.dry_run:
            raise SystemExit(
                "--clips/--clip-match are for iteration only; refusing to "
                "write a corpus whose remaining clips are unprocessed.")
        tot = sum(int(nframes[c]) for c in sel)
        print(f"\nLIMITED RUN: {len(sel)} clips, {tot} frames")
        for c in sel[:5]:
            print(f"    {all_files[c]}")

    # frame indices in the working set
    work = np.concatenate([
        np.arange(int(starts[c]), int(starts[c]) + int(nframes[c]))
        for c in sel]) if sel else np.array([], dtype=int)

    deltas = np.radians(np.arange(-args.max_pitch_deg, args.max_pitch_deg + 0.25,
                                  0.25))
    order = np.argsort(np.abs(deltas))     # try the smallest corrections first

    def foot_lows(t):
        """(current lowest, best-achievable-by-rotation lowest) per foot."""
        d.qpos[:3] = gts[t, 0]
        d.qpos[3:7] = grs[t, 0][[3, 0, 1, 2]]
        d.qpos[7:] = dps[t]
        mujoco.mj_forward(m, d)
        out = {}
        for b, f in feet.items():
            gi = f["geom"]
            R = d.geom_xmat[gi].reshape(3, 3)
            p = d.geom_xpos[gi].copy()
            cur = float((f["hull"] @ R.T + p)[:, 2].min())
            axis = d.xaxis[f["jid"]].copy()
            anchor = d.xanchor[f["jid"]].copy()
            Rk = _rot(axis, deltas)
            pk = anchor + np.einsum("kij,j->ki", Rk, p - anchor)
            Rw = np.einsum("kij,jl->kil", Rk, R)
            wk = np.einsum("vj,kij->kvi", f["hull"], Rw) + pk[:, None, :]
            mins = wk[:, :, 2].min(axis=1)
            out[b] = (cur, mins, float(mins.max()))
        return out

    # ---- pass 1: how much of the sink is HEIGHT (rotation cannot fix)? ----
    best_by_frame = np.full(N, np.nan)
    cur_by_frame = np.full(N, np.nan)
    body_low_before = np.full(N, np.nan)
    for t in work:
        fl = foot_lows(t)
        cur_by_frame[t] = min(v[0] for v in fl.values())
        best_by_frame[t] = min(v[2] for v in fl.values())
        # WHOLE-BODY clearance: on a fall/ground clip the torso is the
        # contact, and a foot-derived lift would make the body hover.
        body_low_before[t] = float(d.geom_xpos[:, 2].min())

    lifts = np.zeros(len(nframes))
    for c in sel:
        s, e = int(starts[c]), int(starts[c]) + int(nframes[c])
        dz = args.clearance - best_by_frame[s:e].min()
        lifts[c] = max(0.0, dz)
        if lifts[c] > 0 and not args.no_lift:
            gts[s:e, :, 2] += lifts[c]
    print(f"\nstep 1 — per-clip lift (the height error rotation cannot fix):")
    print(f"  clips lifted   {int((lifts > 1e-6).sum())} / {len(sel)}")
    print(f"  lift  median   {np.median(lifts[lifts > 1e-6])*100:.2f} cm"
          if (lifts > 1e-6).any() else "  (none needed)")
    print(f"  lift  max      {lifts.max()*100:.2f} cm")

    # ---- pass 2: per-frame ankle pitch for the residual toe dive ---------
    stats = dict(corrected=0, max_delta=0.0, unfixable=0)
    lows_after = np.full(N, np.nan)
    body_low_after = np.full(N, np.nan)
    for t in work:
        fl = foot_lows(t)
        touched = False
        for b, f in feet.items():
            cur, mins, _ = fl[b]
            if cur >= args.clearance:
                continue
            ok = [k for k in order if mins[k] >= args.clearance]
            if not ok:
                stats["unfixable"] += 1
                continue
            delta = float(deltas[ok[0]])
            lo, hi = m.jnt_range[f["jid"]]
            dps[t, f["dof"]] = float(np.clip(dps[t, f["dof"]] + delta, lo, hi))
            stats["max_delta"] = max(stats["max_delta"], abs(delta))
            touched = True
        if touched:
            stats["corrected"] += 1
        # rewrite body poses from the (possibly corrected) dofs and verify
        # against the FULL mesh, not the hull approximation
        d.qpos[:3] = gts[t, 0]
        d.qpos[3:7] = grs[t, 0][[3, 0, 1, 2]]
        d.qpos[7:] = dps[t]
        mujoco.mj_forward(m, d)
        if touched:
            gts[t] = d.xpos[1:]
            grs[t] = d.xquat[1:][:, [1, 2, 3, 0]]           # wxyz -> xyzw
        fl2 = 1e9
        for b, f in feet.items():
            gi = f["geom"]
            w = f["V"] @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi]
            fl2 = min(fl2, float(w[:, 2].min()))
        lows_after[t] = fl2
        body_low_after[t] = float(d.geom_xpos[:, 2].min())

    print(f"\nstep 2 — ankle pitch:")
    print(f"  frames corrected   {stats['corrected']} / {len(work)} "
          f"({100*stats['corrected']/max(len(work),1):.1f}%)")
    print(f"  largest correction {np.degrees(stats['max_delta']):.1f} deg")
    print(f"  still unfixable    {stats['unfixable']} feet")
    print(f"\nRESULT (full mesh, {len(work)} frames):")
    print(f"  deepest point  {np.nanmin(cur_by_frame)*100:+.2f} cm -> "
          f"{np.nanmin(lows_after)*100:+.2f} cm")
    cb, la = cur_by_frame[work], lows_after[work]
    for thr in (0.01, 0.03, 0.05):
        print(f"  penetration >{thr*100:.0f} cm: "
              f"{(cb < -thr).mean()*100:5.1f}% -> {(la < -thr).mean()*100:5.1f}%")
    # HOVER CHECK: did the lift push any body part off the floor that was
    # resting on it? Matters on falls/ground work, where the torso — not the
    # foot — is the contact and a foot-derived lift is meaningless.
    bb, ba = body_low_before[work], body_low_after[work]
    print(f"\n  whole-body lowest point: before {bb.min()*100:+.2f} cm  "
          f"after {ba.min()*100:+.2f} cm")
    hover = ba.min()
    if hover > 0.02:
        print(f"  *** HOVER WARNING: nothing touches the floor "
              f"(min {hover*100:+.2f} cm) — the lift is wrong for these clips")

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    out = Path(args.out or args.corpus)
    if out == Path(args.corpus):
        bak = out.with_suffix(out.suffix + ".pre_footfix")
        if not bak.exists():
            shutil.copy(out, bak)
            print(f"\nbacked up original -> {bak}")
    corpus["gts"] = torch.from_numpy(gts)
    corpus["grs"] = torch.from_numpy(grs)
    corpus["dps"] = torch.from_numpy(dps)
    torch.save(corpus, out)
    print(f"written {out}")
    print("NOTE: velocities (gvs/gavs/dvs) are unchanged. The correction is a "
          "small per-frame ankle rotation, so ankle angular velocity is now "
          "slightly stale; rebuild the corpus from clips if that matters.")


# Guarded because fix_foot_clips.py imports the helpers above; a bare
# main() call would run this script's argparse on import.
if __name__ == "__main__":
    main()
