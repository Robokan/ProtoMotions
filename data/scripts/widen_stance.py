# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Widen a clip's stance by a CONSTANT hip abduction -- Eric's fix.

The history that justifies this (2026-08-15): the robot's legs are thicker
than the mocap human's, so retargeted walks pass with millimetres of
clearance and the policy clips its own knees. Three clever fixes were tried:

  * lateral IK-target offsets  -> the per-frame IK re-solve turned a simple
    push into hip artifacts (worst penetration got DEEPER);
  * min-separation floors      -> same, via a different door;
  * windowed trajectory optim  -> cleared the geometry but the time-varying
    per-window corrections read as an unnatural leg wobble in the viewer.

Eric's observation cut through: the legs NEVER actually cross in this data
(measured 0/1243 frames), so a UNIFORM widen is always a pure clearance
gain -- and a constant joint offset adds exactly zero velocity, zero jerk,
zero wobble. The motion's character is untouched; the stance is wider.

Mechanics: add +delta to the left hip-abduction dof and -delta to the right
(atlas: Leg_3_[LR]_Joint; +5 deg moves the foot 6.7 cm laterally with ~zero
forward/vertical coupling), counter-rotate the ankle rolls so the soles stay
flat, FK-rebuild body poses, and leave every velocity field alone (constant
offsets have no derivative; the residual world-frame error from body-frame
offsets during turns is omega x offset ~ 2 cm/s, noise next to a 90 cm/s
walk). Delta is found by binary search: the smallest value whose worst-frame
leg-pair gap meets --min-gap-cm.

    python data/scripts/widen_stance.py --robot atlas \\
        --in data/motions/atlas_v11/<clip>.motion --out <out>.motion \\
        --min-gap-cm 1.5
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

LEG_BODIES = {"Leg1_L", "Leg2_L", "Leg3_L", "Leg4_L", "Foot_L",
              "Leg1_R", "Leg2_R", "Leg3_R", "Leg4_R", "Foot_R", "Hip"}


def build_world(robot):
    import mujoco
    m = mujoco.MjModel.from_xml_path(
        f"protomotions/data/assets/mjcf/{robot}.xml")
    d = mujoco.MjData(m)
    body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                  for i in range(1, m.nbody)]
    leg_geoms = {g for g in range(m.ngeom)
                 if m.geom_bodyid[g] > 0 and m.geom_contype[g] != 0
                 and body_names[m.geom_bodyid[g] - 1] in LEG_BODIES}
    world_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
    return mujoco, m, d, body_names, leg_geoms, world_geoms


def min_gap_and_ground(mujoco, m, d, leg_geoms, world_geoms, root_pos,
                       root_rot, dof, margin):
    """Worst leg-pair signed distance and lowest body point over the clip."""
    T = dof.shape[0]
    worst_gap = np.inf
    lowest = np.inf
    m.opt.__class__  # noqa -- keep mujoco import alive for clarity
    for t in range(T):
        d.qpos[:3] = root_pos[t]
        d.qpos[3:7] = root_rot[t][[3, 0, 1, 2]]
        d.qpos[7:] = dof[t]
        mujoco.mj_forward(m, d)
        lowest = min(lowest, float(d.geom_xpos[:, 2].min()))
        for ci in range(d.ncon):
            c = d.contact[ci]
            if c.geom1 in world_geoms or c.geom2 in world_geoms:
                continue
            if c.geom1 in leg_geoms and c.geom2 in leg_geoms:
                worst_gap = min(worst_gap, float(c.dist))
    return worst_gap, lowest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="atlas")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-gap-cm", type=float, default=1.5)
    ap.add_argument("--max-delta-deg", type=float, default=4.0)
    ap.add_argument("--abduction-dofs", default="Leg_3_L_Joint,Leg_3_R_Joint",
                    help="left,right hip abduction dof names")
    ap.add_argument("--ankle-roll-dofs", default="Foot_L_Roll,Foot_R_Roll",
                    help="left,right ankle roll dofs for sole compensation; "
                         "'' to skip")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, ".")
    from protomotions.robot_configs.factory import robot_config

    rc = robot_config(args.robot)
    dn = list(rc.kinematic_info.dof_names)
    abd_l, abd_r = [dn.index(x) for x in args.abduction_dofs.split(",")]
    roll = None
    if args.ankle_roll_dofs:
        roll = [dn.index(x) for x in args.ankle_roll_dofs.split(",")]

    mujoco, m, d, body_names, leg_geoms, world_geoms = build_world(args.robot)
    # widen the collision margin so near-miss distances are reported
    m.geom_margin[:] = 0.05

    mo = torch.load(args.inp, weights_only=False, map_location="cpu")
    dof0 = mo["dof_pos"].numpy().astype(np.float64)
    rp = mo["rigid_body_pos"].numpy().astype(np.float64)[:, 0]
    rr = mo["rigid_body_rot"].numpy().astype(np.float64)[:, 0]

    def apply(delta_rad):
        dof = dof0.copy()
        dof[:, abd_l] += delta_rad
        dof[:, abd_r] -= delta_rad
        if roll is not None:
            # counter-rotate the soles by the same constant
            dof[:, roll[0]] -= delta_rad
            dof[:, roll[1]] += delta_rad
        return dof

    base_gap, base_low = min_gap_and_ground(
        mujoco, m, d, leg_geoms, world_geoms, rp, rr, dof0, 0.05)
    print(f"  baseline: worst leg gap {base_gap*100:+.2f} cm, "
          f"lowest point {base_low*100:+.2f} cm")

    target = args.min_gap_cm / 100.0
    lo, hi = 0.0, np.radians(args.max_delta_deg)
    # check the ceiling first
    gap_hi, low_hi = min_gap_and_ground(
        mujoco, m, d, leg_geoms, world_geoms, rp, rr, apply(hi), 0.05)
    if gap_hi < target:
        print(f"  NOTE: even {args.max_delta_deg} deg reaches only "
              f"{gap_hi*100:+.2f} cm; using the ceiling")
        best = hi
    else:
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            gap, _ = min_gap_and_ground(
                mujoco, m, d, leg_geoms, world_geoms, rp, rr, apply(mid), 0.05)
            if gap >= target:
                hi = mid
            else:
                lo = mid
        best = hi

    dof = apply(best)
    gap, low = min_gap_and_ground(
        mujoco, m, d, leg_geoms, world_geoms, rp, rr, dof, 0.05)

    # FK-rebuild poses; velocities untouched (constant offset, zero derivative)
    T = dof.shape[0]
    rbp = mo["rigid_body_pos"].numpy().astype(np.float64).copy()
    rbr = mo["rigid_body_rot"].numpy().astype(np.float64).copy()
    for t in range(T):
        d.qpos[:3] = rp[t]
        d.qpos[3:7] = rr[t][[3, 0, 1, 2]]
        d.qpos[7:] = dof[t]
        mujoco.mj_forward(m, d)
        rbp[t] = d.xpos[1:]
        rbr[t] = d.xquat[1:][:, [1, 2, 3, 0]]
    mo["dof_pos"] = torch.from_numpy(dof).float()
    mo["rigid_body_pos"] = torch.from_numpy(rbp).float()
    mo["rigid_body_rot"] = torch.from_numpy(rbr).float()
    torch.save(mo, args.out)
    print(f"  widened by {np.degrees(best):.2f} deg/side: worst gap "
          f"{base_gap*100:+.2f} -> {gap*100:+.2f} cm | lowest point "
          f"{base_low*100:+.2f} -> {low*100:+.2f} cm | wrote {args.out}")


if __name__ == "__main__":
    main()
