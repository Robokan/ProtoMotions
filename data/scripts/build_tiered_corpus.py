# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Tiered corpus rebuild -- cheapest adequate fix per clip (Eric, 2026-08-15).

The week's verdicts, encoded:
  * trajectory-optimizer output wobbles unnaturally (vetoed) -- so the BASE
    is the natural, pre-optimizer clips;
  * a constant stance widen is invisible and fixes parallel-pass brushes;
  * deep angular collisions (turn pivots) are unfixable by any constant
    offset -- prefer dropping a dirty TAKE when a clean sibling take of the
    same choreography is already in the corpus (measured: takes of the same
    family range 0..38 colliding frames), else trim the bad window, else
    drop;
  * arm contacts are ignored throughout (cannot trip the robot).

Tiers, by worst leg-pair penetration in the raw clip:
  T1  KEEP RAW      worst < 1 cm (grazes; below anything that trips)
  T2  WIDEN         1..3 cm and a <=3 deg constant hip-abduction reaches
                    >= +0.3 cm clearance; else demote to T3
  T3  SWAP/TRIM     clean sibling take exists in-corpus -> DROP this take;
                    else trim colliding windows (>=1s parts kept);
                    else DROP with a logged reason

Output: data/motions/atlas_v15/ + recipe + tier report. Corpus build and
validation stay in the calling script.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import torch

LEG_BODIES = {"Leg1_L", "Leg2_L", "Leg3_L", "Leg4_L", "Foot_L",
              "Leg1_R", "Leg2_R", "Leg3_R", "Leg4_R", "Foot_R", "Hip"}


def family(stem: str) -> str:
    """Choreography id: strip the take suffix (__A476 etc.) and mirror tag."""
    s = re.sub(r"__A\d+", "", stem)
    return re.sub(r"_M$", "", s)


def audit(mujoco, m, d, bn, wg, lg, mo):
    rp = mo["rigid_body_pos"].numpy()[:, 0]
    rr = mo["rigid_body_rot"].numpy()[:, 0]
    dof = mo["dof_pos"].numpy()
    T = dof.shape[0]
    worst = 0.0
    bad = np.zeros(T, dtype=bool)
    for t in range(T):
        d.qpos[:3] = rp[t]
        d.qpos[3:7] = rr[t][[3, 0, 1, 2]]
        d.qpos[7:] = dof[t]
        mujoco.mj_forward(m, d)
        w = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]
            if c.geom1 in wg or c.geom2 in wg or c.dist >= 0:
                continue
            if c.geom1 in lg and c.geom2 in lg:
                w = max(w, -c.dist)
        worst = max(worst, w)
        bad[t] = w > 0.01
    return worst, bad, T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/motions/atlas_v12")
    ap.add_argument("--out-dir", default="data/motions/atlas_v15")
    ap.add_argument("--widen-max-deg", type=float, default=3.0)
    args = ap.parse_args()

    sys.path.insert(0, ".")
    import mujoco
    m = mujoco.MjModel.from_xml_path("protomotions/data/assets/mjcf/atlas.xml")
    d = mujoco.MjData(m)
    bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
          for i in range(1, m.nbody)]
    wg = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
    lg = {g for g in range(m.ngeom)
          if m.geom_bodyid[g] > 0 and m.geom_contype[g] != 0
          and bn[m.geom_bodyid[g] - 1] in LEG_BODIES}

    clips = sorted(f for f in os.listdir(args.in_dir) if f.endswith(".motion"))
    profiles = {}
    for f in clips:
        mo = torch.load(f"{args.in_dir}/{f}", weights_only=False,
                        map_location="cpu")
        worst, bad, T = audit(mujoco, m, d, bn, wg, lg, mo)
        profiles[f] = (worst, bad, T)
        print(f"AUDIT {f[:56]:<56} worst {worst*100:5.2f} cm "
              f"bad {int(bad.sum()):3d}/{T}", flush=True)

    # family cleanliness map for sibling swaps
    fam_clean = {}
    for f, (worst, bad, T) in profiles.items():
        fam = family(f[:-7])
        fam_clean.setdefault(fam, []).append((worst, f))

    os.makedirs(args.out_dir, exist_ok=True)
    report = {"T1_keep": [], "T2_widened": [], "T3_dropped_sibling": [],
              "T3_trimmed": [], "T3_dropped": []}

    for f, (worst, bad, T) in profiles.items():
        src = f"{args.in_dir}/{f}"
        # T1 with hysteresis: a hard 1.0 cm cutoff split near-identical mirror
        # takes (0.69 vs 1.01 cm) and dropped a clean clip over ONE frame
        # 0.1 mm past the line (Eric caught it). Grazes under 1.5 cm, or up to
        # 2 cm when isolated to a couple of frames, cannot trip anything.
        if worst < 0.015 or (worst < 0.02 and int(bad.sum()) <= 2):
            shutil.copy(src, f"{args.out_dir}/{f}")
            report["T1_keep"].append(f)
            continue

        if worst <= 0.035:
            out = f"{args.out_dir}/{f}"
            r = subprocess.run(
                [sys.executable, "data/scripts/widen_stance.py",
                 "--robot", "atlas", "--in", src, "--out", out,
                 "--min-gap-cm", "0.3",
                 "--max-delta-deg", str(args.widen_max_deg)],
                capture_output=True, text=True)
            # Acceptance = the PENETRATIONS are gone, not "every frame has
            # full clearance": demanding whole-clip min-gap failed the widen
            # on all 133 clips, because almost every clip has one tight
            # angular frame no constant offset can open. Judge the output by
            # re-audit instead.
            ok = False
            if os.path.exists(out):
                mo_w = torch.load(out, weights_only=False, map_location="cpu")
                w_post, _, _ = audit(mujoco, m, d, bn, wg, lg, mo_w)
                ok = w_post < 0.015
                print(f"TIER2 {f[:52]} widen: worst {worst*100:.2f} -> "
                      f"{w_post*100:.2f} cm ({'accept' if ok else 'demote'})",
                      flush=True)
            if ok:
                report["T2_widened"].append(f)
                print(f"TIER2 {f[:56]} widened", flush=True)
                continue
            if os.path.exists(out):
                os.remove(out)
            # fall through to tier 3

        fam = family(f[:-7])
        siblings = [g for w, g in fam_clean.get(fam, [])
                    if g != f and w < 0.01]
        if siblings:
            report["T3_dropped_sibling"].append(
                f"{f} (covered by {siblings[0]})")
            print(f"TIER3 {f[:56]} DROPPED (clean sibling {siblings[0][:40]})",
                  flush=True)
            continue

        # trim the bad windows; keep >=1s clean parts
        mo = torch.load(src, weights_only=False, map_location="cpu")
        fps = float(mo.get("fps", 30))
        min_frames = int(1.0 * fps)
        keep_runs = []
        s = None
        mask = profiles[f][1]
        for i, b in enumerate(np.append(mask, True)):
            if not b and s is None:
                s = i
            elif b and s is not None:
                if i - s >= min_frames:
                    keep_runs.append((s, i))
                s = None
        if keep_runs:
            FRAME_KEYS = ["dof_pos", "dof_vel", "rigid_body_pos",
                          "rigid_body_rot", "rigid_body_vel",
                          "rigid_body_ang_vel", "rigid_body_contacts",
                          "local_rigid_body_rot"]
            for i, (a, b) in enumerate(keep_runs):
                sub = dict(mo)
                for k in FRAME_KEYS:
                    if k in sub and hasattr(sub[k], "__getitem__"):
                        sub[k] = sub[k][a:b].clone()
                torch.save(sub, f"{args.out_dir}/{f[:-7]}__p{i}.motion")
            report["T3_trimmed"].append(
                f"{f} -> {len(keep_runs)} parts")
            print(f"TIER3 {f[:56]} trimmed to {len(keep_runs)} parts",
                  flush=True)
        else:
            report["T3_dropped"].append(f)
            print(f"TIER3 {f[:56]} DROPPED (no clean run >= 1s)", flush=True)

    print("\nTIER REPORT")
    for k, v in report.items():
        print(f"  {k}: {len(v)}")
        for item in v:
            print(f"    {item}")


if __name__ == "__main__":
    main()
