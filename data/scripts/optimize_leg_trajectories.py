# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Whole-trajectory re-optimization of leg collisions in .motion clips.

Every per-frame fix tried on the leg self-collisions lost to the plain IK
collision limit, and always for the same structural reason: acting only at
the contact frame, its one available move is a visible pose change right at
the pass (measured: constant lateral offset 4.91 -> 12.73 cm worst, target
min-separation floors 4.91 -> 9.87 cm). What actually clears a close pass
is bending the swing leg's whole arc outward by ~1 mm per frame starting
well before the contact -- an adjustment no per-frame solver can express.

So this does what the field does (H2O/PHC-style pipelines run the same idea
at the source): a small offline optimization over a WINDOW of frames around
each collision, jointly minimizing

    w_dev    * (q - q_original)^2          stay the same motion
    w_smooth * (second difference)^2       spread any change over time
    w_pen    * hinge(clearance - dist)^2   don't interpenetrate
    w_foot   * hinge(orig_foot_z - foot_z)^2   don't dig the feet in

over the LEG dofs only (hips/knees: Leg_[1358]_[LR]_Joint). Ankles are
deliberately excluded: fix_foot_clips owns foot-ground geometry and this
pass must not undo it -- the foot-height hinge guards against the leg
motion lowering a foot, and verification re-measures ground clearance.

Contact distances come from MuJoCo itself with an enlarged margin on the
leg colliders (contacts are otherwise only reported once penetrating,
which gives the optimizer no gradient until it is already too late).
Penetration/foot gradients are finite-differenced per frame over the 8 leg
dofs (9 forwards per frame); deviation/smoothness gradients are analytic.

The first/last `pin` frames of each window are frozen so the optimized
segment rejoins the untouched trajectory with no seam. dof_vel and the leg
subtree's body velocities are recomputed inside the window only -- the same
surgical policy as fix_foot_clips, and for the same reason: wholesale
finite-difference velocities would replace the retarget's smoother
derivatives everywhere to fix a handful of dofs.

    python data/scripts/optimize_leg_trajectories.py --robot atlas \\
        --in-dir data/motions/atlas_v12 --out-dir /tmp/opt \\
        --clips walk_ff_loop_180_R_walk_arc_cw_loop_R_walk_ff_loop_360_R_003__A478_M
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

LEG_BODIES = {"Leg1_L", "Leg2_L", "Leg3_L", "Leg4_L", "Foot_L",
              "Leg1_R", "Leg2_R", "Leg3_R", "Leg4_R", "Foot_R", "Hip"}
ARM_BODIES = {f"Arm{i}_{s}" for i in range(1, 10) for s in "LR"} | {"Hand1_L", "Hand1_R"}
FOOT_BODIES = ("Foot_L", "Foot_R")

# Mode semantics (Eric, 2026-08-13): legs need a real CLEARANCE -- a shin
# grazing a shin mid-swing becomes a trip in sim. Arms only need
# NON-PENETRATION: arms legitimately sit close to (and rest against) the
# body -- a guard on the chest is intentional style -- but pressing INTO a
# body part is a retarget artifact. So arm clearance ~0: deliberate contact
# survives as contact, interpenetration gets re-pathed out.
MODES = {
    "legs": dict(bodies=LEG_BODIES, pair="both", dof_prefix="Leg_",
                 foot_guard=True, clearance_cm=0.5),
    "arms": dict(bodies=ARM_BODIES, pair="any", dof_prefix="Arm_",
                 foot_guard=False, clearance_cm=0.2),
}


def _helpers():
    spec = importlib.util.spec_from_file_location(
        "ffgp", Path(__file__).with_name("fix_foot_ground_pitch.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._ang_vel_from_quats


class LegWorld:
    """MuJoCo wrapper: leg-pair distances and foot heights for one frame."""

    def __init__(self, robot: str, margin: float, mode: dict = None):
        self.mode = mode or MODES["legs"]
        import sys
        sys.path.insert(0, ".")
        import mujoco
        from protomotions.robot_configs.factory import robot_config

        self.mujoco = mujoco
        self.m = mujoco.MjModel.from_xml_path(
            f"protomotions/data/assets/mjcf/{robot}.xml")
        self.d = mujoco.MjData(self.m)
        rc = robot_config(robot)
        self.dof_names = list(rc.kinematic_info.dof_names)
        self.body_names = [
            mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(1, self.m.nbody)]
        if self.body_names != list(rc.kinematic_info.body_names):
            raise SystemExit("MJCF body order != robot config body order")
        self.leg_geoms = set()
        for g in range(self.m.ngeom):
            b = self.m.geom_bodyid[g]
            if b > 0 and self.m.geom_contype[g] != 0 \
                    and self.body_names[b - 1] in self.mode["bodies"]:
                self.leg_geoms.add(g)
                # report contacts BEFORE penetration so the optimizer sees
                # a gradient while there is still time to steer away
                self.m.geom_margin[g] = margin
        self.world_geoms = {g for g in range(self.m.ngeom)
                            if self.m.geom_bodyid[g] == 0}
        # optimized dofs (legs: hips/knees, NOT ankles; arms: all arm dofs)
        self.opt_idx = [i for i, n in enumerate(self.dof_names)
                        if n.startswith(self.mode["dof_prefix"])]
        # joint-range bounds for the optimizer (dof i <-> joint 1+i after
        # the free root); without these L-BFGS can walk a dof past its limit
        self.opt_bounds = [(float(self.m.jnt_range[1 + i][0]),
                            float(self.m.jnt_range[1 + i][1]))
                           for i in self.opt_idx]
        self.all_geoms_by_body = {}
        self.foot_geoms = {}
        for b in FOOT_BODIES:
            gi = [g for g in range(self.m.ngeom)
                  if self.m.geom_bodyid[g] == self.body_names.index(b) + 1
                  and self.m.geom_contype[g] != 0][0]
            mid = self.m.geom_dataid[gi]
            va, vn = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
            self.foot_geoms[b] = (gi, self.m.mesh_vert[va:va + vn].copy())

    def set_frame(self, root_pos, root_rot_xyzw, dof):
        self.d.qpos[:3] = root_pos
        self.d.qpos[3:7] = root_rot_xyzw[[3, 0, 1, 2]]
        self.d.qpos[7:] = dof
        self.mujoco.mj_forward(self.m, self.d)

    def leg_pair_distances(self):
        """Signed distances (m) of watched contacts (<= margin).

        pair="both": both geoms in the mode's body set (leg-vs-leg).
        pair="any":  at least one geom in the set (arm-vs-anything) --
        Eric's spec is that an arm pressing into ANY body part is wrong.
        """
        out = []
        both = self.mode["pair"] == "both"
        for ci in range(self.d.ncon):
            c = self.d.contact[ci]
            if c.geom1 in self.world_geoms or c.geom2 in self.world_geoms:
                continue
            a, b = c.geom1 in self.leg_geoms, c.geom2 in self.leg_geoms
            if (a and b) if both else (a or b):
                out.append(float(c.dist))
        return out

    def foot_lows(self):
        lows = {}
        for b, (gi, V) in self.foot_geoms.items():
            R = self.d.geom_xmat[gi].reshape(3, 3)
            lows[b] = float((V @ R.T + self.d.geom_xpos[gi])[:, 2].min())
        return lows


def window_cost(world, x, q0_win, pos_win, rot_win, foot0, free,
                clearance, w_dev, w_smooth, w_pen, w_foot):
    """Cost + analytic dev/smooth grad + FD pen/foot grad, per window."""
    T = q0_win.shape[0]
    idx = world.opt_idx
    q = q0_win.copy()
    q[np.ix_(free, idx)] = x.reshape(len(free), len(idx))

    dev = q[:, idx] - q0_win[:, idx]
    cost = w_dev * float((dev ** 2).sum())
    grad_q = 2.0 * w_dev * dev                        # [T, J] wrt q[:,idx]

    acc = q[2:, idx] - 2 * q[1:-1, idx] + q[:-2, idx]
    cost += w_smooth * float((acc ** 2).sum())
    ga = 2.0 * w_smooth * acc
    grad_q[2:] += ga
    grad_q[1:-1] += -2.0 * ga
    grad_q[:-2] += ga

    eps = 1e-3
    for k, t in enumerate(free):
        world.set_frame(pos_win[t], rot_win[t], q[t])
        base = 0.0
        for dist in world.leg_pair_distances():
            if dist < clearance:
                base += w_pen * (clearance - dist) ** 2
        if world.mode["foot_guard"]:
            for b, low in world.foot_lows().items():
                if low < foot0[t][b]:
                    base += w_foot * (foot0[t][b] - low) ** 2
        cost += base
        if base > 0.0:
            for jj, j in enumerate(idx):
                qp = q[t].copy(); qp[j] += eps
                world.set_frame(pos_win[t], rot_win[t], qp)
                pert = 0.0
                for dist in world.leg_pair_distances():
                    if dist < clearance:
                        pert += w_pen * (clearance - dist) ** 2
                if world.mode["foot_guard"]:
                    for b, low in world.foot_lows().items():
                        if low < foot0[t][b]:
                            pert += w_foot * (foot0[t][b] - low) ** 2
                grad_q[t, jj] += (pert - base) / eps
    return cost, grad_q[free].ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="atlas")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--clips", nargs="*", default=None,
                    help="stems to process (default: every clip with a "
                         "leg contact deeper than --trigger-cm)")
    ap.add_argument("--mode", choices=list(MODES), default="legs")
    ap.add_argument("--trigger-cm", type=float, default=1.0)
    ap.add_argument("--clearance-cm", type=float, default=None,
                    help="default per mode: legs 0.5 (real gap), arms 0.2 "
                         "(contact ok, penetration not)")
    ap.add_argument("--margin-cm", type=float, default=2.0)
    ap.add_argument("--pad", type=int, default=15,
                    help="frames of context around each bad stretch")
    ap.add_argument("--pin", type=int, default=2,
                    help="frozen frames at each window edge (seam continuity)")
    ap.add_argument("--w-dev", type=float, default=30.0)
    ap.add_argument("--w-smooth", type=float, default=300.0)
    ap.add_argument("--w-pen", type=float, default=4000.0)
    ap.add_argument("--w-foot", type=float, default=4000.0)
    ap.add_argument("--maxiter", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ang_vel_from_quats = _helpers()
    mode = MODES[args.mode]
    world = LegWorld(args.robot, args.margin_cm / 100.0, mode)
    if args.clearance_cm is None:
        args.clearance_cm = mode["clearance_cm"]
    clear = args.clearance_cm / 100.0
    trig = args.trigger_cm / 100.0

    out = Path(args.out_dir)
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    paths = sorted(Path(args.in_dir).glob("*.motion"))
    if args.clips:
        paths = [p for p in paths if any(c in p.name for c in args.clips)]

    for p in paths:
        mo = torch.load(p, weights_only=False, map_location="cpu")
        pos = mo["rigid_body_pos"].numpy().astype(np.float64)
        rot = mo["rigid_body_rot"].numpy().astype(np.float64)
        dof = mo["dof_pos"].numpy().astype(np.float64).copy()
        dvel = mo["dof_vel"].numpy().astype(np.float64).copy()
        bvel = mo["rigid_body_vel"].numpy().astype(np.float64).copy()
        avel = mo["rigid_body_ang_vel"].numpy().astype(np.float64).copy()
        fps = float(mo.get("fps", 30)); dt = 1.0 / fps
        T = dof.shape[0]

        # per-frame worst penetration and original foot heights
        worst = np.zeros(T); foot0 = []
        for t in range(T):
            world.set_frame(pos[t, 0], rot[t, 0], dof[t])
            dists = [d for d in world.leg_pair_distances() if d < 0]
            worst[t] = -min(dists) if dists else 0.0
            foot0.append(world.foot_lows())
        bad = worst > trig
        if not bad.any():
            if not args.dry_run:
                shutil.copy(p, out / p.name)
            continue

        # bad stretches -> padded windows, merged when overlapping
        runs = []
        t = 0
        while t < T:
            if bad[t]:
                s = t
                while t < T and bad[t]:
                    t += 1
                a, b = max(0, s - args.pad), min(T, t + args.pad)
                if runs and a <= runs[-1][1]:
                    runs[-1] = (runs[-1][0], b)
                else:
                    runs.append((a, b))
            else:
                t += 1

        report = []
        for a, b in runs:
            q0_win = dof[a:b].copy()
            # pin frames only where the window borders UNTOUCHED trajectory;
            # at a clip boundary there is nothing to rejoin, and pinning
            # there locks in any penetration the clip opens or ends with
            pin_lo = 0 if a == 0 else args.pin
            pin_hi = 0 if b == T else args.pin
            free = list(range(pin_lo, (b - a) - pin_hi))
            x0 = q0_win[np.ix_(free, world.opt_idx)].ravel()
            f0 = [foot0[a + t] for t in range(b - a)]
            res = minimize(
                lambda x: window_cost(world, x, q0_win, pos[a:b, 0],
                                      rot[a:b, 0], f0, free, clear,
                                      args.w_dev, args.w_smooth,
                                      args.w_pen, args.w_foot),
                x0, jac=True, method="L-BFGS-B",
                bounds=world.opt_bounds * len(free),
                options={"maxiter": args.maxiter})
            qn = q0_win.copy()
            qn[np.ix_(free, world.opt_idx)] = res.x.reshape(
                len(free), len(world.opt_idx))
            dof[a:b] = qn
            # measure the window after
            w_after = 0.0; dq = 0.0
            for t in range(a, b):
                world.set_frame(pos[t, 0], rot[t, 0], dof[t])
                dists = [d for d in world.leg_pair_distances() if d < 0]
                if dists:
                    w_after = max(w_after, -min(dists))
                dq = max(dq, float(np.abs(np.degrees(
                    dof[t] - mo["dof_pos"].numpy()[t])).max()))
            report.append((a, b, worst[a:b].max(), w_after, dq, res.nit))

        # FK writeback + surgical velocities inside the touched windows
        for a, b in runs:
            for t in range(a, b):
                world.set_frame(pos[t, 0], rot[t, 0], dof[t])
                pos[t] = world.d.xpos[1:].copy()
                rot[t] = world.d.xquat[1:][:, [1, 2, 3, 0]]
            lo, hi = max(0, a - 1), min(T, b + 1)
            for j in world.opt_idx:
                dvel[lo:hi, j] = np.gradient(dof[lo:hi, j], dt)
            leg_ids = [world.body_names.index(n) for n in world.mode["bodies"]
                       if n in world.body_names]
            for bi in leg_ids:
                bvel[lo:hi, bi] = np.gradient(pos[lo:hi, bi], dt, axis=0)
                avel[lo:hi, bi] = ang_vel_from_quats(rot[lo:hi, bi], dt) \
                    if hi - lo > 2 else avel[lo:hi, bi]

        for (a, b, w0, w1, dq, nit) in report:
            print(f"  {p.name[:44]:<44} win {a:3d}-{b:3d}  "
                  f"pen {w0*100:5.2f} -> {w1*100:5.2f} cm  "
                  f"max dq {dq:4.1f} deg  ({nit} iters)", flush=True)
        if not args.dry_run:
            mo["rigid_body_pos"] = torch.from_numpy(pos).float()
            mo["rigid_body_rot"] = torch.from_numpy(rot).float()
            mo["dof_pos"] = torch.from_numpy(dof).float()
            mo["dof_vel"] = torch.from_numpy(dvel).float()
            mo["rigid_body_vel"] = torch.from_numpy(bvel).float()
            mo["rigid_body_ang_vel"] = torch.from_numpy(avel).float()
            torch.save(mo, out / p.name)


if __name__ == "__main__":
    main()
