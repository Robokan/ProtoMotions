# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Battle-league progress check: are the fighters fighting yet?

Reads the run's tensorboard scalars and reports the three behavioral
milestones Eric watches for, in order:

  1. APPROACH  -- fighters face and close distance (battle_approach reward)
  2. STRIKING  -- attempts happen and land (kick_attempt, hit_taken)
  3. FIGHTS    -- bouts get decided (battle_win, Elo spread, pool growth)

Prints a compact trend table (early window vs recent window) and a verdict
line per milestone so a human or an agent can act on it.

    python scripts/check_battle_progress.py [--run atlas_ase_battle_hlc_v4]
"""

import argparse
import glob
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="atlas_ase_battle_hlc_v4")
args = parser.parse_args()

from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)

pts = {}
for d in sorted(glob.glob(f"results/{args.run}/lightning_logs/version_*")):
    a = EventAccumulator(d, size_guidance={"scalars": 0})
    a.Reload()
    for t in a.Tags()["scalars"]:
        for e in a.Scalars(t):
            pts.setdefault(t, {})[e.step] = e.value

if not pts:
    print(f"NO DATA for {args.run}")
    raise SystemExit(1)

steps = sorted(pts.get("info/episode_reward", pts[next(iter(pts))]))
top = steps[-1]
print(f"run: {args.run}   latest epoch: {top}")


def window(tag, lo, hi):
    d = pts.get(tag, {})
    xs = [v for s, v in d.items() if lo <= s <= hi]
    return statistics.mean(xs) if xs else float("nan")


early = (max(0, min(50, top // 10)), max(1, top // 5))
recent = (max(0, top - max(50, top // 5)), top)

TAGS = [
    ("approach", "env/raw_r/battle_approach_mean"),
    ("kick_attempt", "env/scaled_r/battle_kick_attempt_mean"),
    ("hit_taken", "env/scaled_r/battle_hit_taken_mean"),
    ("strike_div", "env/scaled_r/battle_strike_diversity_mean"),
    ("win", "env/scaled_r/battle_win_mean"),
    ("idle_penalty", "env/scaled_r/battle_idle_mean"),
    ("elo", "league/agent_elo"),
    ("pool_size", "league/pool_size"),
    ("own_fam_wr", "league/own_family_win_rate"),
]

print(f"\n{'metric':<14} {'early(%d-%d)' % early:>16} {'recent(%d-%d)' % recent:>16}   trend")
rows = {}
for name, tag in TAGS:
    e, r = window(tag, *early), window(tag, *recent)
    rows[name] = (e, r)
    arrow = "up" if r > e * 1.05 + 1e-9 else ("down" if r < e * 0.95 - 1e-9 else "flat")
    print(f"{name:<14} {e:>16.5f} {r:>16.5f}   {arrow}")

print("\nMILESTONES:")
ap_e, ap_r = rows["approach"]
print(
    "  1. APPROACH : %s (raw approach reward %.4f -> %.4f)"
    % ("PROGRESSING" if ap_r > ap_e else "NOT YET", ap_e, ap_r)
)
ka, ht = rows["kick_attempt"][1], rows["hit_taken"][1]
striking = (abs(ka) > 1e-6) or (abs(ht) > 1e-6)
print(
    "  2. STRIKING : %s (kick_attempt %.5f, hit_taken %.5f in recent window)"
    % ("HAPPENING" if striking else "NOT YET", ka, ht)
)
win_e, win_r = rows["win"]
pool = rows["pool_size"][1]
# The win reward sits at a NEGATIVE timeout floor (~-1.7) while every bout
# times out undecided; "fights are being decided" means it RISES off that
# floor, or the league gate passed (pool grows beyond the seed member).
decided = (win_r > win_e + 0.05) or pool > 1.5
print(
    "  3. FIGHTS   : %s (win reward %.5f -> %.5f, pool %.0f, elo %.0f)"
    % ("DECIDED" if decided else "NOT YET", win_e, win_r, pool, rows["elo"][1])
)
