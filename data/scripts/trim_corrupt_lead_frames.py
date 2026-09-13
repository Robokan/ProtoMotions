#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Trim leading frames whose DOF values fall outside the robot's joint limits.

go2_flat.pt carries a corrupt lead-in on 33_clip_2_mirror: its first 5 frames
put RR_thigh_joint ~149 deg past its upper stop. Spawning there would have the
solver fight the joint limit. Only the lead-in is bad, so the clip is trimmed
rather than dropped.

    python data/scripts/trim_corrupt_lead_frames.py data/motions/go2/go2_flat.pt --robot go2
    python data/scripts/trim_corrupt_lead_frames.py <file> --robot go2 --apply
"""

import argparse
import shutil

import numpy as np
import torch

from protomotions.robot_configs.factory import robot_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion_file")
    ap.add_argument("--robot", default="go2")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    ki = robot_config(args.robot).kinematic_info
    lo = np.array([float(x) for x in ki.dof_limits_lower])
    hi = np.array([float(x) for x in ki.dof_limits_upper])

    d = torch.load(args.motion_file, map_location="cpu", weights_only=False)
    n_frames = int(d["gts"].shape[0])
    dp = d["dps"].numpy()
    viol = np.maximum(np.maximum(dp - hi, 0), np.maximum(lo - dp, 0)).max(axis=1)

    starts = d["length_starts"].numpy()
    counts = d["motion_num_frames"].numpy()
    names = [str(x).split("/")[-1] for x in d["motion_files"]]

    drop = []
    for c in range(len(starts)):
        s, n = int(starts[c]), int(counts[c])
        bad_local = np.where(viol[s : s + n] > 0)[0]
        if len(bad_local) == 0:
            continue
        # Only a contiguous run at the very start is safe to trim; anything
        # else is mid-clip corruption and needs a human look.
        if bad_local[0] != 0 or not np.array_equal(bad_local, np.arange(len(bad_local))):
            print(f"  !! {names[c]}: {len(bad_local)} bad frames NOT a leading run "
                  f"(local {bad_local[:6]}) -- skipping, needs review")
            continue
        k = len(bad_local)
        if k >= n - 1:
            print(f"  !! {names[c]}: entire clip out of limits -- skipping")
            continue
        print(f"  {names[c]}: trimming {k} leading frames "
              f"(worst {np.degrees(viol[s:s+k].max()):.1f} deg over limit), {n} -> {n-k}")
        drop.append((c, s, k))

    if not drop:
        print("nothing to trim")
        return
    if not args.apply:
        print("\n  (report only -- re-run with --apply)")
        return

    shutil.copy2(args.motion_file, args.motion_file + ".bak_untrimmed")
    drop_idx = np.concatenate([np.arange(s, s + k) for _, s, k in drop])
    keep = torch.from_numpy(np.setdiff1d(np.arange(n_frames), drop_idx))

    for key, v in list(d.items()):
        if torch.is_tensor(v) and v.shape and v.shape[0] == n_frames:
            d[key] = v[keep]
    for c, _, k in drop:
        d["motion_num_frames"][c] -= k
        d["motion_lengths"][c] = (d["motion_num_frames"][c] - 1).float() * d["motion_dt"][c]
    shifted = d["motion_num_frames"].roll(1)
    shifted[0] = 0
    d["length_starts"] = shifted.cumsum(0)
    torch.save(d, args.motion_file)

    # Verify at the point of consumption.
    chk = torch.load(args.motion_file, map_location="cpu", weights_only=False)
    assert int(chk["gts"].shape[0]) == int(chk["motion_num_frames"].sum()), "packing inconsistent"
    dp2 = chk["dps"].numpy()
    v2 = np.maximum(np.maximum(dp2 - hi, 0), np.maximum(lo - dp2, 0)).max(axis=1)
    print(f"\n  {len(chk['motion_files'])} clips, {int(chk['gts'].shape[0])} frames, "
          f"{float(chk['motion_lengths'].sum())/60:.1f} min")
    print(f"  out-of-limit frames remaining: {int((v2 > 0).sum())} (verified on reload)")


if __name__ == "__main__":
    main()
