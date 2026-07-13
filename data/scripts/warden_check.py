"""Weekend warden telemetry check: prints key=value lines + a VERDICT.

Run inside the container from the ProtoMotions root. Reads the league's
tfevents over a trailing window and classifies health per WEEKEND_ORDERS.md.
"""

import glob
import sys

from tensorboard.backend.event_processing import event_accumulator

RUN = sys.argv[1] if len(sys.argv) > 1 else "soma_battle_league_v2"

files = sorted(glob.glob(f"results/{RUN}/lightning_logs/*/events.out.tfevents.*"))
if not files:
    print("VERDICT=NO_DATA")
    sys.exit(0)

ea = event_accumulator.EventAccumulator(files[-1], size_guidance={"scalars": 0})
ea.Reload()


def series(tag):
    try:
        return [e.value for e in ea.Scalars(tag)]
    except KeyError:
        return []


def recent(vals, k=20):
    return sum(vals[-k:]) / max(len(vals[-k:]), 1) if vals else 0.0


facing = series("env/battle/facing_mean")
hands = series("env/battle/dealt_hands_mean")
legs = series("env/battle/dealt_legs_mean")
draws = series("env/battle/draw_mean")
kos = series("env/battle/end_ko_mean")
points = series("env/battle/end_points_mean")
ep_len = series("info/episode_length")

hits_series = [h + l for h, l in zip(hands, legs)]
hits_recent = recent(hits_series)
hits_peak = max(hits_series) if hits_series else 0.0

print(f"epochs={len(facing)}")
print(f"facing_recent={recent(facing):.3f}")
print(f"hits_recent={hits_recent:.4f} hits_peak={hits_peak:.4f}")
print(f"hands_recent={recent(hands):.4f} legs_recent={recent(legs):.4f}")
print(f"draw_recent={recent(draws):.5f} ko_recent={recent(kos):.5f} points_recent={recent(points):.5f}")
print(f"episode_len_recent={recent(ep_len):.0f}")

# Classification (thresholds per WEEKEND_ORDERS.md)
n = len(facing)
if n < 40:
    print("VERDICT=WARMING_UP")
elif recent(facing) > 0.8 and hits_recent < 0.005 and n > 100:
    print("VERDICT=WRONG_AXIS_SUSPECTED")
elif (
    hits_peak > 0.05
    and hits_recent < 0.1 * hits_peak
    and n > 150
    and recent(points) < 0.4 * recent(draws)
    and max(hits_series[-20:] or [0.0]) < 0.02
):
    # True relapse = sustained near-zero hits AND draws dominating points.
    # Burst-style fighting (zeros between heavy exchanges) is healthy and
    # must not trip this (learned Sat night: bursts of 0.2+ between lulls).
    print("VERDICT=PASSIVITY_RELAPSE")
elif abs(recent(facing) - 0.5) < 0.05 and n > 150:
    # Facing pinned at chance (~0.5) = fighters NOT orienting toward each
    # other, regardless of hit numbers (spawn-proximity flailing can fake
    # hits). This is the weekend failure the viewer caught but telemetry
    # missed: healthy engagement drives facing meaningfully above 0.5.
    print("VERDICT=NOT_ENGAGING")
elif recent(facing) < 0.4 and n > 150:
    print("VERDICT=FACING_WEAK")
else:
    print("VERDICT=HEALTHY")
