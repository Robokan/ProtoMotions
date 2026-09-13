#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Cut a policy capture into a packaged motion library, keeping only MOTION.

Input is a raw dump from capture_policy_motion.py: [T, E, ...] tensors of one
rollout per env. Output is a packaged lib in the same schema as
data/motions/go2/*.pt (gts/grs/gvs/gavs/dps/dvs + per-clip index arrays).

Three things have to happen between the two, and each is load-bearing:

RE-ORIGIN. Captured positions are world coordinates, so every env sits at its
own slot on the env grid -- 100..215 m from the origin for a 128-env layout.
Source corpora keep root xy within a couple of metres of zero, so each segment
is shifted to start at xy=(0,0). Z is left absolute: it is measured from the
floor and carries the standing height.

DROP THE RSI HAND-OFF. Episodes begin in a reference state that carries the
mocap's velocity, so the first stretch is partly the reference's motion rather
than the policy's. --skip-sec of every episode is discarded.

KEEP ONLY MOVING FRAMES. A style-only AMP policy spends part of its time
standing, and a corpus of standing teaches an LLC to stand. Filter terms:

  planar speed >= --min-speed      actually translating
  |roll|,|pitch| < --max-tilt-deg  upright
  --min-z < root z < --max-z       standing, not lying or leaping
  clearance > --min-clearance      trunk well above the feet
  rear drop < --max-rear-drop      NOT redundant with pitch: a quadruped sits
                                   by folding its rear legs, so the trunk stays
                                   near level while the hindquarters sink.
                                   Front-hip minus rear-hip height catches it.

The last three mirror extract_go2_real_backwards.py, which learned them the
hard way -- rearing and sitting clips both passed a pitch-only filter.

    python data/scripts/segment_policy_capture.py --raw /tmp/go2_reversed_raw.pt
    python data/scripts/segment_policy_capture.py --raw ... --out ... --apply
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# raw capture field -> packaged motion-lib key
_MAP = {
    "rigid_body_pos": "gts",
    "rigid_body_rot": "grs",
    "rigid_body_vel": "gvs",
    "rigid_body_ang_vel": "gavs",
    "dof_pos": "dps",
    "dof_vel": "dvs",
}


def runs_of(mask: np.ndarray, min_frames: int):
    """Contiguous True runs of at least min_frames, as [start, end)."""
    out, s = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= min_frames:
                out.append((s, i))
            s = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data/motions/go2/go2_reversed_mocap.pt")
    ap.add_argument("--robot", default="go2")
    ap.add_argument("--skip-sec", type=float, default=1.0,
                    help="discard this much of each episode after a reset -- "
                         "the RSI reference state's velocity bleeds into it")
    ap.add_argument("--min-speed", type=float, default=0.15,
                    help="[m/s] planar root speed; below this is standing")
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--max-tilt-deg", type=float, default=25.0)
    ap.add_argument("--min-z", type=float, default=0.25)
    ap.add_argument("--max-z", type=float, default=0.45)
    ap.add_argument("--min-clearance", type=float, default=0.26)
    ap.add_argument("--max-rear-drop", type=float, default=0.06)
    ap.add_argument("--smooth-sec", type=float, default=0.1,
                    help="smoothing window for the speed test, so single-frame "
                         "dips don't fragment one walk into many clips")
    ap.add_argument("--pad", type=int, default=3)
    ap.add_argument("--max-clips", type=int, default=0,
                    help="0 = keep all; otherwise keep the longest N")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, ".")
    from protomotions.robot_configs.factory import robot_config

    body_names = list(robot_config(args.robot).kinematic_info.body_names)
    FH = [body_names.index("FL_hip"), body_names.index("FR_hip")]
    RH = [body_names.index("RL_hip"), body_names.index("RR_hip")]
    FEET = [body_names.index(f + "_foot") for f in ("FL", "FR", "RL", "RR")]

    raw = torch.load(args.raw, weights_only=False, map_location="cpu")
    dt = float(raw["control_dt"])
    fps = 1.0 / dt
    pos = raw["rigid_body_pos"].numpy()      # [T, E, B, 3]
    rot = raw["rigid_body_rot"].numpy()      # [T, E, B, 4] xyzw
    vel = raw["rigid_body_vel"].numpy()      # [T, E, B, 3]
    dones = raw["dones"].numpy()             # [T, E]
    T, E = pos.shape[0], pos.shape[1]

    skip = int(round(args.skip_sec * fps))
    min_frames = int(round(args.min_seconds * fps))
    sm = max(1, int(round(args.smooth_sec * fps)))

    # episode age per env, so the RSI hand-off is skipped after EVERY reset
    age = np.zeros((T, E), dtype=int)
    for t in range(1, T):
        age[t] = np.where(dones[t - 1] > 0, 0, age[t - 1] + 1)

    segments = []      # (env, start, end)
    speed_all = []
    for e in range(E):
        P = pos[:, e]                                   # [T, B, 3]
        root_v = vel[:, e, 0, :2]
        speed = np.linalg.norm(root_v, axis=1)
        if sm > 1:
            speed = np.convolve(speed, np.ones(sm) / sm, mode="same")
        rpy = R.from_quat(rot[:, e, 0]).as_euler("zyx")
        yaw, pitch, roll = rpy[:, 0], rpy[:, 1], rpy[:, 2]
        heading = np.stack([np.cos(yaw), np.sin(yaw)], 1)
        along = (root_v * heading).sum(1)

        rear_drop = P[:, FH, 2].mean(axis=1) - P[:, RH, 2].mean(axis=1)
        clearance = P[:, 0, 2] - P[:, FEET, 2].mean(axis=1)

        ok = (
            (speed >= args.min_speed)
            & (np.abs(np.degrees(roll)) < args.max_tilt_deg)
            & (np.abs(np.degrees(pitch)) < args.max_tilt_deg)
            & (P[:, 0, 2] > args.min_z)
            & (P[:, 0, 2] < args.max_z)
            & (rear_drop < args.max_rear_drop)
            & (clearance > args.min_clearance)
            & (age >= skip)[:, e]
        )
        speed_all.append(speed[age[:, e] >= skip])
        # Episode index per frame, so padding cannot cross a reset. A run ends
        # exactly WHERE the reset happens (the mask goes False there), so an
        # unclamped +pad welds the reset teleport -- a ~190 m single-frame jump
        # on a 128-env grid -- onto the tail of every clip.
        ep = np.cumsum(age[:, e] == 0)
        for a, b in runs_of(ok, min_frames):
            same = np.flatnonzero(ep == ep[a])
            lo, hi = int(same[0]), int(same[-1]) + 1
            a = max(a - args.pad, lo)
            b = min(b + args.pad, hi)
            segments.append((e, a, b, (b - a) / fps,
                             float(np.median(along[a:b]))))

    sp = np.concatenate(speed_all)
    print(f"raw: {T} steps x {E} envs @ {fps:.0f}Hz = {T*E*dt:.0f}s "
          f"({int(dones.sum())} resets)")
    print(f"planar speed after skipping {args.skip_sec}s/episode: "
          f"median {np.median(sp):.3f}  mean {sp.mean():.3f}  "
          f"p90 {np.percentile(sp,90):.3f}  max {sp.max():.3f} m/s")
    for thr in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        print(f"    frames with speed >= {thr:.2f}: {100*(sp>=thr).mean():5.1f}%")

    segments.sort(key=lambda x: -x[3])
    if args.max_clips and len(segments) > args.max_clips:
        print(f"\n  keeping longest {args.max_clips} of {len(segments)} segments")
        segments = segments[: args.max_clips]

    total = sum(x[3] for x in segments)
    print(f"\n{len(segments)} segments pass the filter, {total:.1f}s total")
    if not segments:
        print("  nothing passed -- loosen --min-speed or --min-seconds")
        return
    durs = np.array([x[3] for x in segments])
    alongs = np.array([x[4] for x in segments])
    print(f"  duration  min {durs.min():.2f}s  median {np.median(durs):.2f}s  "
          f"max {durs.max():.2f}s")
    print(f"  along-heading speed  median {np.median(alongs):+.3f} m/s  "
          f"({100*(alongs<0).mean():.0f}% of clips move rearward)")
    print(f"  longest 8:")
    for e, a, b, dur, spd in segments[:8]:
        print(f"    env{e:03d} [{a:4d}:{b:4d}]  {dur:5.2f}s  {spd:+.3f} m/s")

    if not args.apply:
        print("\n  (report only -- re-run with --apply)")
        return

    out = {k: [] for k in _MAP.values()}
    files, nf, dts, lens = [], [], [], []
    for i, (e, a, b) in enumerate((s[0], s[1], s[2]) for s in segments):
        for src, dst in _MAP.items():
            chunk = raw[src][a:b, e].clone()
            if dst == "gts":
                # re-origin: world -> segment-local xy, z untouched
                chunk[..., :2] -= chunk[0, 0, :2].clone()
            out[dst].append(chunk)
        files.append(f"policy_reversed_env{e:03d}_{a:04d}.motion")
        nf.append(b - a)
        dts.append(dt)
        lens.append((b - a - 1) * dt)

    packed = {k: torch.cat(v, dim=0).contiguous() for k, v in out.items()}
    nf_t = torch.tensor(nf, dtype=torch.long)
    starts = nf_t.roll(1)
    starts[0] = 0
    packed["length_starts"] = starts.cumsum(0)
    packed["motion_num_frames"] = nf_t
    packed["motion_dt"] = torch.tensor(dts, dtype=torch.float32)
    packed["motion_lengths"] = torch.tensor(lens, dtype=torch.float32)
    packed["motion_weights"] = torch.ones(len(segments), dtype=torch.float32)
    packed["motion_files"] = tuple(files)
    torch.save(packed, args.out)

    chk = torch.load(args.out, weights_only=False, map_location="cpu")
    assert int(chk["gts"].shape[0]) == int(chk["motion_num_frames"].sum())
    assert int(chk["length_starts"][-1]) + int(chk["motion_num_frames"][-1]) \
        == int(chk["gts"].shape[0])
    # No clip may contain a teleport. A reset moves the robot metres-to-
    # hundreds-of-metres in one frame; real motion at 50 Hz moves centimetres.
    # Checked per clip because a jump only ever appears at an episode boundary.
    g = chk["gts"].numpy()
    st = chk["length_starts"].numpy()
    ct = chk["motion_num_frames"].numpy()
    worst, worst_i = 0.0, -1
    for i, (s0, k) in enumerate(zip(st, ct)):
        P = g[s0 : s0 + k, 0, :2]
        if k > 1:
            j = float(np.abs(np.diff(P, axis=0)).max())
            if j > worst:
                worst, worst_i = j, i
    step_limit = 0.5  # m/frame == 25 m/s at 50 Hz; nothing real comes close
    assert worst < step_limit, (
        f"clip {worst_i} jumps {worst:.2f} m in one frame -- a reset teleport "
        f"leaked into a segment"
    )
    print(f"\n  wrote {args.out}  (verified on reload)")
    print(f"    largest single-frame root step {worst:.4f} m (limit {step_limit})")
    print(f"    {len(chk['motion_files'])} clips, {g.shape[0]} frames, "
          f"{float(chk['motion_lengths'].sum()):.1f}s")
    print(f"    root xy {g[:,0,0].min():.2f}..{g[:,0,0].max():.2f}, "
          f"{g[:,0,1].min():.2f}..{g[:,0,1].max():.2f}   "
          f"root z {g[:,0,2].min():.3f}..{g[:,0,2].max():.3f}")


if __name__ == "__main__":
    main()
