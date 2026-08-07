# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Can this robot's actuators actually produce the motion in these clips?

A clip retargeted onto a skeleton is only kinematically valid: it says where
the joints go, never whether the motors could get them there. Scale the body
and that gap widens fast, because torque grows as s^4 while the clip's
accelerations are unchanged -- the tiger went to 271 kg and its efforts moved
by only 1.25x.

For every frame this runs INVERSE DYNAMICS and compares the required joint
torque against each actuator's effort limit.

    tau = M(q) qddot + C(q, qdot) - contact and constraint forces

CONTACTS AND CONSTRAINTS ARE DISABLED, and that is a deliberate, documented
compromise. Leaving them on makes the numbers meaningless: measuring the
raptor with contacts live produced a 4425 N.m demand on a 40 kg hip, because
self-collision impulses and joint-limit constraint forces land in
qfrc_inverse and get read as actuator effort. With them off, the ground is
not there to support the body, so STANCE-PHASE demand is OVERSTATED -- the
joints are asked to hold up the whole animal unaided. Read the result as an
upper bound: a clip that passes is certainly feasible, a clip that fails may
still be fine. It is a triage tool, not a verdict.

Velocity and acceleration come from finite differences of dof_pos at the
clip's own dt, so a clip retimed via fps is handled correctly.

    python data/scripts/scan_actuator_feasibility.py --robot tiger \
        --motion-dir data/motions/tiger_v5
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="tiger")
    ap.add_argument("--motion-dir", required=True)
    ap.add_argument("--report", type=int, default=14)
    ap.add_argument("--stride", type=int, default=2,
                    help="sample every Nth frame; inverse dynamics is the cost")
    args = ap.parse_args()

    sys.path.insert(0, ".")
    import mujoco
    from protomotions.robot_configs.factory import robot_config

    rc = robot_config(args.robot)
    mj = mujoco.MjModel.from_xml_path(
        f"protomotions/data/assets/mjcf/{args.robot}.xml")
    # see the docstring: with these ON, constraint forces masquerade as torque
    mj.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT
                            | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    data = mujoco.MjData(mj)

    jnames = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
              for j in range(mj.njnt)
              if mj.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,
                                    mujoco.mjtJoint.mjJNT_SLIDE)]
    limits = np.array([
        float(getattr(rc.control.control_info.get(n), "effort_limit", 0.0) or 0.0)
        or np.inf for n in jnames])
    print(f"{args.robot}: {len(jnames)} actuated dofs, "
          f"effort limits {np.nanmin(limits[np.isfinite(limits)]):.0f}"
          f"-{np.nanmax(limits[np.isfinite(limits)]):.0f} N.m, "
          f"mass {mj.body_mass.sum():.1f} kg")

    rows = []
    worst_dof = {}
    for path in sorted(glob.glob(f"{args.motion_dir}/*.motion")):
        d = torch.load(path, weights_only=False, map_location="cpu")
        dof = d["dof_pos"].numpy().astype(np.float64)
        rp = d["rigid_body_pos"].numpy().astype(np.float64)
        rr = d["rigid_body_rot"].numpy().astype(np.float64)
        dt = 1.0 / float(d.get("fps", 30))
        if len(dof) < 5:
            continue
        vel = np.gradient(dof, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        over_frac = np.zeros(len(jnames))
        peak = np.zeros(len(jnames))
        n_used = 0
        for t in range(2, len(dof) - 2, args.stride):
            data.qpos[:3] = rp[t, 0]
            data.qpos[3:7] = rr[t, 0][[3, 0, 1, 2]]
            data.qpos[7:] = dof[t]
            data.qvel[:] = 0.0
            data.qvel[6:] = vel[t]
            data.qacc[:] = 0.0
            data.qacc[6:] = acc[t]
            mujoco.mj_inverse(mj, data)
            tau = np.abs(data.qfrc_inverse[6:])
            peak = np.maximum(peak, tau)
            over_frac += (tau > limits)
            n_used += 1
        if not n_used:
            continue
        over_frac /= n_used
        ratio = peak / limits
        rows.append((os.path.basename(path)[:-7], float(ratio.max()),
                     float(over_frac.max()), jnames[int(ratio.argmax())]))
        for i, n in enumerate(jnames):
            worst_dof[n] = max(worst_dof.get(n, 0.0), ratio[i])

    rows.sort(key=lambda r: -r[1])
    print(f"\n{'clip':<36}{'peak/limit':>11}{'worst dof':>26}{'frames over':>13}")
    for name, r, frac, dofn in rows[:args.report]:
        flag = "  <-- INFEASIBLE" if r > 1.0 else ""
        print(f"  {name:<34}{r:10.2f}x{dofn:>26}{100*frac:11.0f}%{flag}")
    bad = [r for r in rows if r[1] > 1.0]
    print(f"\n{len(bad)}/{len(rows)} clips exceed an effort limit somewhere "
          f"(upper bound -- contacts disabled, so stance is overstated)")
    print("\ndofs most often saturated:")
    for n, r in sorted(worst_dof.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {n:<28}{r:6.2f}x limit")


main()
