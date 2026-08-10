# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Calibration probe: contact IMPULSE vs the current kinetic-energy damage.

Runs a real exhibition fight and, for every hit event the existing FSM
fires, records BOTH quantities side by side. Changes nothing about the
damage model -- it only measures, so the numbers can be compared before
any game-rule change.

Why impulse. Damage is currently ``0.5 * m_striker * v_closing^2`` using
the STRIKING BODY's own mass. That understates a punch, whose effective
mass is the arm and the torso behind it, and it breaks outright on the
T800, whose fist collider sits on LINK_WRIST_END -- a 0.001 kg massless
frame, i.e. ~0 J for any punch. A contact impulse (``J = integral F dt``)
needs no effective-mass model at all: the articulated solver already
resolves how much of the chain is coupled, so ``J = m_eff * dv`` comes
out for free.

Attribution. In the paired-env layout a body's *net* contact force
includes the ground, so it cannot be read as "hit by the opponent".
But every contact sensor is already filtered against the ground and any
scene objects, so ``force_matrix_w`` gives exactly the force attributable
to those -- and the residual

    F_opponent = net_forces_w - sum(force_matrix_w over filters)

is the opponent's contribution, with no new sensors needed.

Windowing. Impulse keeps accruing while contact lasts, so a punch that
turns into a shove would keep scoring (this is how an earlier
force-based model drained 100% health in 3 s of guard-grinding).
Integration is therefore bounded to --window-ms after each event onset.

FINDING (2026-08-04) — RETRACTED 2026-08-08. The first run reported hit
events on ``LeftShin`` with ``sensor=NO`` and concluded that most damage
bodies lack sensors. That was a bug in this probe, not a property of the
system: ``damage_body_ids`` index the COMMON body order while the hook
printed them through ``sim._robot.data.body_names`` (SIMULATOR order) and
built its force tensor in simulator order too. The events were real hits
on sensored bodies, mislabeled. An event can only fire where
``f_mag > force_on``, which is itself proof the body had a live sensor.

CORRECTED FINDING (2026-08-08): head and torso — the rows that decide
fights — are sensored on every battle robot, which is why the leagues
train. The one real gap is the PELVIS damage row (Hips / Waist /
LINK_BASE / RigPelvis: always the robot's root body), absent from the
semantic sensor list on all six robots, plus SOMA's Spine1/Spine2. Those
rows read a constant 0 N and have never been able to score.
``probe_impulse_hook.install()`` now injects those sensors into replayed
configs, and the battle experiments sensor them for future runs. The
``force_matrix_w`` residual channel (opponent = net - ground) works on
every sensored body; sensors already record per-physics-step history
(``history_length=decimation``), so impulse integration has full
resolution.

Run (headless is fine; a fight needs no display):

    python data/scripts/probe_impulse_vs_ke.py \
        --run soma_battle_league_v5 --bouts 2 --headless
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="soma_battle_league_v5")
    ap.add_argument("--bouts", type=int, default=2)
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--window-ms", type=float, default=80.0,
                    help="impulse integration window after event onset; a "
                         "real impact lasts ~50-100 ms")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out", default="data/impulse_vs_ke_probe.txt")
    args = ap.parse_args()

    snaps = sorted(glob.glob(
        f"results/{args.run}/lightning_logs/*/league/policy_*.ckpt"))
    if not snaps:
        sys.exit(f"no league snapshots under results/{args.run}")
    a, b = snaps[-1], snaps[max(0, len(snaps) - 2)]
    resolved = f"results/{args.run}/resolved_configs_inference.pt"

    # The probe body runs INSIDE battle_tournament via PROBE_IMPULSE, which
    # tournament code does not know about -- we inject it with sitecustomize
    # style monkeypatching from an env var read by the patch module below.
    env = dict(os.environ)
    env["PROBE_IMPULSE"] = "1"
    env["PROBE_WINDOW_MS"] = str(args.window_ms)
    env["PROBE_OUT"] = args.out
    env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    cmd = [
        sys.executable, "-c",
        "import probe_impulse_hook; probe_impulse_hook.install();"
        " import runpy, sys;"
        " sys.argv = ['battle_tournament.py'] + sys.argv[1:];"
        " runpy.run_path('protomotions/battle_tournament.py',"
        " run_name='__main__')",
        "--resolved-configs", resolved,
        "--exhibition", a, b,
        "--bouts", str(args.bouts),
        "--num-envs", str(args.num_envs),
        "--no-fast-sampling",
    ]
    if args.headless:
        cmd.append("--headless")
    print("probe:", " ".join(cmd[:3]), "...", flush=True)
    subprocess.run(cmd, env=env, cwd=REPO, check=False)

    if os.path.exists(args.out):
        print("\n===== probe results =====")
        with open(args.out) as fh:
            sys.stdout.write(fh.read())
    else:
        print(f"probe: no output written to {args.out}")


if __name__ == "__main__":
    main()
