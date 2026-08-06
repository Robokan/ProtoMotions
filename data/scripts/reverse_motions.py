# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Time-reverse clips to turn falls into get-ups.

The raptor corpus has six ways to fall over and exactly one way to stand
back up (Sleep_End, a slow wake-from-rest). A knocked-down fighter
therefore has almost nothing to imitate. Playing a death backwards gives
a plausible rise from the same final pose, which costs nothing to
produce and multiplies the recovery material by six.

WHERE THIS IS VALID. Reversing time negates velocity but NOT gravity, so
a reversed clip is only physical where the motion is quasi-static -- in
contact with the ground, driven by pushing against it. That is exactly
what a get-up is. It is NOT valid across a ballistic phase: the moment
in a death where the body topples freely, played backwards, becomes a
body leaping upward off the floor unaided. So:

  - velocities are negated and reversed (v_new[i] = -v_old[N-1-i]),
    which is exact, rather than refitted by finite difference;
  - the clip is optionally trimmed to its GROUNDED span, dropping the
    airborne topple at the start of the death (= the end of the get-up);
  - the ballistic error of the result is reported, so a clip that is
    still physically silly is visible rather than assumed fine.

    python data/scripts/reverse_motions.py \
        --in-dir data/motions/raptor_v5 --out-dir data/motions/raptor_reversed \
        --clips DeathL DeathR DeathSpecial
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

# per-frame tensors that are simply reversed
_POSE_KEYS = ("dof_pos", "rigid_body_pos", "rigid_body_rot", "rigid_body_contacts")
# per-frame tensors that are reversed AND sign-flipped (they are d/dt)
_VEL_KEYS = ("dof_vel", "rigid_body_vel", "rigid_body_ang_vel")


def grounded_span(pos: np.ndarray, clearance: float):
    """First and last frame where some part of the body is near the floor.

    Used to drop a death's airborne topple, which reversed would read as
    the raptor launching itself off the ground.
    """
    low = pos[:, :, 2].min(axis=1)
    on = np.flatnonzero(low <= clearance)
    if len(on) == 0:
        return None
    return int(on[0]), int(on[-1])


def ballistic_error(pos: np.ndarray, mass: np.ndarray, dt: float,
                    clearance: float) -> float:
    """Mean |COM vertical accel + g| over airborne frames; 0 = free fall."""
    air = pos[:, :, 2].min(axis=1) > clearance
    if air.sum() < 4:
        return 0.0
    com = (pos * mass[None, :, None]).sum(1) / mass.sum()
    acc = np.gradient(np.gradient(com[:, 2], dt), dt)
    return float(np.abs(acc[air] + 9.81).mean())


def reverse_clip(d: dict, trim_to_grounded: bool, clearance: float):
    """Return a time-reversed copy of a loaded motion dict."""
    n = d["rigid_body_pos"].shape[0]
    lo, hi = 0, n - 1
    trimmed = 0
    if trim_to_grounded:
        span = grounded_span(d["rigid_body_pos"].numpy(), clearance)
        if span is not None:
            lo, hi = span
            trimmed = (n - 1 - hi) + lo
    out = dict(d)
    idx = torch.arange(hi, lo - 1, -1)          # reversed, inclusive
    for k in _POSE_KEYS:
        if k in d:
            out[k] = d[k][idx].contiguous()
    for k in _VEL_KEYS:
        if k in d:
            out[k] = (-d[k][idx]).contiguous()
    return out, trimmed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="raptor")
    ap.add_argument("--in-dir", default="data/motions/raptor_v5")
    ap.add_argument("--out-dir", default="data/motions/raptor_reversed")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip stems to reverse; mirrors (_M) are picked up "
                         "automatically")
    ap.add_argument("--prefix", default="Getup_",
                    help="name prefix for the reversed clip. The _M mirror "
                         "suffix stays at the END so mirror-aware tooling "
                         "keeps working (Getup_DeathL_M).")
    ap.add_argument("--trim-to-grounded", action="store_true",
                    help="drop frames where the body is airborne, so the "
                         "reversed clip does not start with an un-physical "
                         "leap off the floor")
    ap.add_argument("--clearance", type=float, default=0.10)
    args = ap.parse_args()

    import mujoco
    import sys
    sys.path.insert(0, ".")
    from protomotions.robot_configs.factory import robot_config

    rc = robot_config(args.robot)
    bn = list(rc.kinematic_info.body_names)
    mj = mujoco.MjModel.from_xml_path(
        f"protomotions/data/assets/mjcf/{args.robot}.xml")
    sn = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i)
          for i in range(mj.nbody)]
    mass = np.array([mj.body_mass[sn.index(b)] if b in sn else 0.0 for b in bn])

    os.makedirs(args.out_dir, exist_ok=True)
    stems = args.clips
    if not stems:
        stems = sorted(os.path.basename(p)[:-7]
                       for p in glob.glob(f"{args.in_dir}/*.motion"))
    # pull in the mirrors of whatever was asked for
    want = []
    for s in stems:
        for suffix in ("", "_M"):
            if os.path.exists(f"{args.in_dir}/{s}{suffix}.motion"):
                want.append(f"{s}{suffix}")
    want = sorted(set(want))

    print(f"{'clip':<38}{'frames':>8}{'trimmed':>9}"
          f"{'start h':>9}{'end h':>8}{'ballistic':>11}")
    for stem in want:
        d = torch.load(f"{args.in_dir}/{stem}.motion",
                       weights_only=False, map_location="cpu")
        dt = 1.0 / float(d.get("fps", 30))
        out, trimmed = reverse_clip(d, args.trim_to_grounded, args.clearance)
        pos = out["rigid_body_pos"].numpy()
        # a get-up should START prone and END standing
        s_h, e_h = float(pos[0, 0, 2]), float(pos[-1, 0, 2])
        err = ballistic_error(pos, mass, dt, args.clearance)
        # _M must stay the final token for mirror-aware tooling
        base = stem[:-2] if stem.endswith("_M") else stem
        new = f"{args.prefix}{base}" + ("_M" if stem.endswith("_M") else "")
        torch.save(out, f"{args.out_dir}/{new}.motion")
        print(f"{new:<38}{pos.shape[0]:8d}{trimmed:9d}"
              f"{s_h*100:8.0f}c{e_h*100:7.0f}c{err:10.1f}")

    print(f"\nwritten to {args.out_dir}")


main()
