# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Cut self-collision segments out of .motion clips, keeping the clean parts.

Some retargeted clips are mostly fine but contain a stretch the robot cannot
physically hold -- limbs interpenetrating because the robot's are thicker than
the source human's. Atlas is the only robot with asset.self_collisions=True,
so those frames are not cosmetic: the sim resolves them as real contacts, the
robot kicks its own shin and trips, and the AMP discriminator meanwhile
rewards reproducing the impossible pose.

Dropping a whole clip for a bad half-second throws away good motion; this
splits it instead, the same shape as trim_motion_stillness (mask -> bridge ->
runs -> sub-clips), reusing its run helpers so the two behave identically.

WHICH PAIRS COUNT is the important knob, and the default is deliberately
narrow. Hand and arm contacts do not destabilise the robot -- Eric's call, and
constraining them in the retarget measurably made things worse (arm branch
flips on punch/kick clips went 44 -> 63 deg/frame). Legs, feet and pelvis are
what trip it. --pairs all is available for auditing but will carve up clips
over contacts that do not matter.

Velocities need no recomputation: slicing selects frames, it does not move
bodies, so every derivative stays consistent with its own frames. Only the
join points vanish, and those are the frames being removed.

    python data/scripts/trim_motion_collisions.py --robot atlas \\
        --in-dir data/motions/atlas_v12 --out-dir data/motions/atlas_v13 \\
        --depth-cm 3 --dry-run
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import torch

# frame-indexed keys, matching mine_seed_combat_events.trim_motion
FRAME_KEYS = [
    "dof_pos",
    "dof_vel",
    "rigid_body_pos",
    "rigid_body_rot",
    "rigid_body_vel",
    "rigid_body_ang_vel",
    "rigid_body_contacts",
    "local_rigid_body_rot",
]

_LEG_PREFIXES = ("leg", "foot", "ankle")


def _load_run_helpers():
    """Reuse trim_motion_stillness's run detection (it is __main__-guarded)."""
    path = Path(__file__).with_name("trim_motion_stillness.py")
    spec = importlib.util.spec_from_file_location("_tms", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.close_gaps, mod.long_still_runs


def _pair_filter(mode: str, body_names):
    """Return a predicate on (bodyA, bodyB) selecting collisions that matter."""
    if mode == "all":
        return lambda a, b: True
    legs = {
        n for n in body_names
        if n and (n.lower().startswith(_LEG_PREFIXES) or n == "Hip")
    }
    return lambda a, b: a in legs and b in legs


def collision_mask(model, data, mo, depth_m, keep_pair, mujoco):
    """Per-frame True where a selected pair penetrates deeper than depth_m."""
    pos = mo["rigid_body_pos"].numpy().astype(np.float64)
    rot = mo["rigid_body_rot"].numpy().astype(np.float64)
    dof = mo["dof_pos"].numpy().astype(np.float64)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(1, model.nbody)
    ]
    world_geoms = {
        g for g in range(model.ngeom) if model.geom_bodyid[g] == 0
    }
    T = pos.shape[0]
    mask = np.zeros(T, dtype=bool)
    worst = np.zeros(T)
    for t in range(T):
        data.qpos[:3] = pos[t, 0]
        data.qpos[3:7] = rot[t, 0][[3, 0, 1, 2]]  # xyzw -> wxyz
        data.qpos[7:] = dof[t]
        mujoco.mj_forward(model, data)
        for ci in range(data.ncon):
            c = data.contact[ci]
            if c.geom1 in world_geoms or c.geom2 in world_geoms:
                continue  # ground contact is not self-collision
            if c.dist >= -depth_m:
                continue
            a = names[model.geom_bodyid[c.geom1] - 1]
            b = names[model.geom_bodyid[c.geom2] - 1]
            if keep_pair(a, b):
                mask[t] = True
                worst[t] = max(worst[t], -c.dist)
    return mask, worst


def pop_mask(mo, dof_names, thr_deg, prefixes):
    """Per-frame True around an IK branch flip -- a joint that jumps.

    The retarget solves each frame independently, so a redundant limb can
    settle on a different IK branch from one frame to the next and the joint
    teleports. 0.5 rad/frame (~29 deg) is the threshold this rig's arm flips
    were originally caught with; real motion sits an order of magnitude below
    it (median arm motion is well under 1 deg/frame on locomotion, ~4-8 on a
    kip-up, against 50-60 for a flip).

    A jump spans frames t and t+1, so BOTH are marked: keeping either leaves
    the teleport at a segment boundary, which is the artifact being removed.
    """
    if not prefixes:
        return np.zeros(mo["dof_pos"].shape[0], dtype=bool), 0.0
    idx = [i for i, n in enumerate(dof_names) if n.startswith(prefixes)]
    if not idx:
        return np.zeros(mo["dof_pos"].shape[0], dtype=bool), 0.0
    dof = np.degrees(mo["dof_pos"].numpy().astype(np.float64)[:, idx])
    jump = np.abs(np.diff(dof, axis=0)).max(axis=1)  # [T-1]
    bad = jump > thr_deg
    mask = np.zeros(dof.shape[0], dtype=bool)
    mask[:-1] |= bad
    mask[1:] |= bad
    return mask, float(jump.max() if len(jump) else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="atlas")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pairs", choices=["legs", "all"], default="legs",
                    help="legs (default): only contacts that can trip the robot")
    ap.add_argument("--depth-cm", type=float, default=3.0,
                    help="penetration deeper than this marks a frame bad")
    ap.add_argument("--min-seconds", type=float, default=1.0,
                    help="discard kept segments shorter than this")
    ap.add_argument("--bridge-frames", type=int, default=5,
                    help="clean stretches shorter than this inside a bad run "
                         "are treated as bad, so one clip is not shredded into "
                         "fragments by a few good frames")
    ap.add_argument("--blip-frames", type=int, default=2,
                    help="bad runs this short are ignored rather than cut out")
    ap.add_argument("--pop-deg", type=float, default=29.0,
                    help="a joint jumping more than this in one frame is an "
                         "IK branch flip (29 deg ~ the 0.5 rad/frame the arm "
                         "flips on this rig were first caught with); 0 disables")
    ap.add_argument("--pop-dofs", default="Arm",
                    help="comma-separated dof-name prefixes to police for pops "
                         "('Arm' default, 'all' for every dof, '' to disable)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, ".")
    import mujoco

    close_gaps, long_runs = _load_run_helpers()
    model = mujoco.MjModel.from_xml_path(
        f"protomotions/data/assets/mjcf/{args.robot}.xml")
    data = mujoco.MjData(model)
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(1, model.nbody)
    ]
    keep_pair = _pair_filter(args.pairs, body_names)
    depth_m = args.depth_cm / 100.0

    paths = sorted(Path(args.in_dir).glob("*.motion"))
    if args.limit:
        paths = paths[: args.limit]
    out = Path(args.out_dir)
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    from protomotions.robot_configs.factory import robot_config
    dof_names = list(robot_config(args.robot).kinematic_info.dof_names)
    if args.pop_dofs.strip().lower() == "all":
        pop_prefixes = tuple(sorted({n[:3] for n in dof_names}))
    else:
        pop_prefixes = tuple(x for x in args.pop_dofs.split(",") if x)
    if args.pop_deg <= 0:
        pop_prefixes = ()

    tot = dict(clips=0, clean=0, split=0, dropped=0, frames_in=0,
               frames_out=0, subclips=0, by_collision=0, by_pop=0)
    for p in paths:
        mo = torch.load(p, weights_only=False, map_location="cpu")
        fps = float(mo.get("fps", 30))
        min_frames = max(1, int(args.min_seconds * fps))
        coll, worst = collision_mask(model, data, mo, depth_m, keep_pair, mujoco)
        pops, worst_pop = pop_mask(mo, dof_names, args.pop_deg, pop_prefixes)
        mask = coll | pops
        tot["by_collision"] += int(coll.sum())
        tot["by_pop"] += int((pops & ~coll).sum())
        T = len(mask)
        tot["clips"] += 1
        tot["frames_in"] += T

        # ignore isolated blips, then treat short clean gaps inside a bad
        # stretch as bad so the clip splits at two seams instead of six
        bad = np.zeros(T, dtype=bool)
        for s, e in long_runs(mask, max(1, args.blip_frames), 0):
            bad[s:e] = True
        bad = close_gaps(bad, args.bridge_frames)

        if not bad.any():
            tot["clean"] += 1
            tot["frames_out"] += T
            tot["subclips"] += 1
            if not args.dry_run:
                shutil.copy(p, out / p.name)
            continue

        # keep the complement: contiguous clean runs long enough to be useful
        keep = long_runs(~bad, min_frames, 0)
        if not keep:
            tot["dropped"] += 1
            print(f"  {p.name[:50]:<50} DROPPED  (no clean run >= "
                  f"{args.min_seconds:g}s; worst {worst.max()*100:.1f} cm, "
                  f"pop {worst_pop:.0f} deg)")
            continue

        tot["split"] += 1
        for i, (a, b) in enumerate(keep):
            sub = dict(mo)
            for k in FRAME_KEYS:
                if k in sub and hasattr(sub[k], "__getitem__"):
                    sub[k] = sub[k][a:b].clone()
            tot["frames_out"] += b - a
            tot["subclips"] += 1
            if not args.dry_run:
                torch.save(sub, out / f"{p.stem}__p{i}.motion")
        cut = T - sum(b - a for a, b in keep)
        why = []
        if coll.any():
            why.append(f"{worst.max()*100:.1f}cm")
        if pops.any():
            why.append(f"pop {worst_pop:.0f}deg")
        print(f"  {p.name[:50]:<50} {len(keep)} part(s), cut {cut:4d}/{T:4d} "
              f"frames ({cut/fps:4.1f}s)  [{', '.join(why)}]")

    print(f"\n{tot['clips']} clips: {tot['clean']} clean, {tot['split']} split, "
          f"{tot['dropped']} dropped entirely")
    print(f"  bad frames: {tot['by_collision']} from collision, "
          f"{tot['by_pop']} from pops alone")
    print(f"  {tot['subclips']} output clips, "
          f"{tot['frames_out']}/{tot['frames_in']} frames kept "
          f"({100*tot['frames_out']/max(tot['frames_in'],1):.1f}%)")
    if args.dry_run:
        print("dry run: nothing written")
    else:
        print(f"written to {out}")


if __name__ == "__main__":
    main()
