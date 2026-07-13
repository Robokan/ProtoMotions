# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Battle tournament evaluation CLI (SOMA_GPC_COMBAT_PLAN Phase 7).

Round-robin evaluation, single-pairing regression gates, and exhibition
matches over slim league adapter checkpoints. Construction mirrors
``inference_agent.py``: configs come frozen from the league run's
``resolved_configs_inference.pt`` (kept beside the checkpoint).

Round-robin ladder over a league directory::

    python protomotions/battle_tournament.py \\
        --resolved-configs results/soma_battle_league/resolved_configs_inference.pt \\
        --adapters results/soma_battle_league/league/ \\
        --matches-per-pairing 32 --num-envs 128 --headless \\
        --output tournament_report.json

Regression gate (candidate vs previous snapshot)::

    python protomotions/battle_tournament.py \\
        --resolved-configs ... --gate results/.../league/policy_7.ckpt \\
        --gate-against results/.../league/policy_6.ckpt --matches-per-pairing 64

Exhibition (two checkpoints, one arena, viewer on)::

    python protomotions/battle_tournament.py \\
        --resolved-configs ... --exhibition ckptA.ckpt ckptB.ckpt \\
        --num-envs 2 --deterministic
"""

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_parser():
    parser = argparse.ArgumentParser(description="Battle adapter tournament")
    parser.add_argument(
        "--resolved-configs",
        required=True,
        help="Path to the league run's resolved_configs_inference.pt",
    )
    parser.add_argument(
        "--adapters",
        default=None,
        help="Directory of policy_*.ckpt adapters (or comma-separated list) "
        "for the round-robin ladder",
    )
    parser.add_argument("--gate", default=None, help="Candidate adapter for the gate")
    parser.add_argument(
        "--gate-against", default=None, help="Reference adapter for the gate"
    )
    parser.add_argument(
        "--gate-threshold", type=float, default=0.55, help="Gate pass score"
    )
    parser.add_argument(
        "--exhibition",
        nargs=2,
        default=None,
        metavar=("CKPT_A", "CKPT_B"),
        help="Two adapters for an exhibition match (viewer on, 1+ arenas)",
    )
    parser.add_argument("--matches-per-pairing", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--simulator", default="isaaclab")
    parser.add_argument("--headless", action="store_true", default=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use mean/greedy actions instead of sampling",
    )
    parser.add_argument("--output", default=None, help="JSON report path")
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=0,
        help="Diagnostic: step the first pairing N steps, printing opponent-"
        "observation and engagement stats instead of running matches",
    )
    parser.add_argument(
        "--autocast",
        choices=["off", "bf16", "fp16"],
        default="off",
        help="Run prior forwards under bf16/fp16 autocast (Blackwell tensor "
        "cores; fp32 weights kept). Speeds inference; viewer/eval only.",
    )
    parser.add_argument(
        "--fast-sampling",
        action="store_true",
        help="Use nucleus sampling at inference (skip the per-token reference "
        "forward). Halves the prior decodes; behavior may shift slightly.",
    )
    parser.add_argument(
        "--action-hold",
        type=int,
        default=1,
        help="Viewer smoothness: re-decode each fighter's policy only every N "
        "control steps (physics still steps every frame). 2-3 ~2-3x's fps at "
        "the cost of slightly coarser control. 1 = decode every step.",
    )
    return parser


parser = create_parser()
args = parser.parse_args()

# isaacgym/isaaclab must be imported before torch
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402


def resolve_adapter_list(spec: str):
    path = Path(spec)
    if path.is_dir():
        adapters = sorted(path.glob("policy_*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not adapters:
            raise FileNotFoundError(f"No policy_*.ckpt adapters in {path}")
        return [str(p) for p in adapters]
    return [entry.strip() for entry in spec.split(",") if entry.strip()]


def main():
    resolved_path = Path(args.resolved_configs)
    assert resolved_path.exists(), f"Missing resolved configs: {resolved_path}"
    resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)

    robot_config = resolved["robot"]
    simulator_config = resolved["simulator"]
    terrain_config = resolved.get("terrain")
    scene_lib_config = resolved["scene_lib"]
    motion_lib_config = resolved["motion_lib"]
    env_config = resolved["env"]
    agent_config = resolved["agent"]

    headless = args.headless
    if args.exhibition is not None and headless is None:
        headless = False
    if headless is None:
        headless = True
    simulator_config.headless = headless

    if args.num_envs is not None:
        if args.num_envs % 2 != 0:
            raise ValueError("--num-envs must be even (2 envs per match)")
        simulator_config.num_envs = args.num_envs

    # Evaluation never trains: neutralize league growth machinery
    if hasattr(agent_config, "league"):
        agent_config.league.staleness_epochs = 10**9
        agent_config.league.gate_min_games = 10**9

    fabric_config = FabricConfig(
        accelerator="cpu" if args.simulator == "mujoco" else "gpu",
        devices=1,
        num_nodes=1,
        loggers=[],
        callbacks=[],
    )
    fabric = Fabric(**fabric_config.as_kwargs())
    fabric.launch()

    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher = AppLauncher(
            {"headless": headless, "device": str(fabric.device)}
        )
        simulator_extra_params["simulation_app"] = app_launcher.app

    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.utils.component_builder import build_all_components

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=getattr(env_config, "save_dir", None),
        **simulator_extra_params,
    )

    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(
        config=agent_config,
        env=env,
        fabric=fabric,
        root_dir=resolved_path.parent,
    )
    agent.setup()

    from protomotions.agents.league.tournament import BattleTournament

    tournament = BattleTournament(
        agent,
        deterministic=args.deterministic,
        action_hold=args.action_hold,
        autocast_dtype=None if args.autocast == "off" else args.autocast,
        sampling_mode="nucleus" if args.fast_sampling else None,
    )

    if args.probe_steps > 0:
        assert args.exhibition is not None, "--probe-steps requires --exhibition"
        ckpt_a, ckpt_b = args.exhibition
        tournament.probe(ckpt_a, ckpt_b, steps=args.probe_steps)
        return

    if args.exhibition is not None:
        ckpt_a, ckpt_b = args.exhibition
        log.info("Exhibition: %s vs %s", ckpt_a, ckpt_b)
        result = tournament.run_pairing(
            ckpt_a, ckpt_b, matches=args.matches_per_pairing
        )
        log.info(
            "Exhibition result: %d-%d-%d (A wins - B wins - draws)",
            result.wins_a,
            result.wins_b,
            result.draws,
        )
        return

    if args.gate is not None:
        if args.gate_against is None:
            raise ValueError("--gate requires --gate-against")
        passed = tournament.regression_gate(
            args.gate,
            args.gate_against,
            matches=args.matches_per_pairing,
            threshold=args.gate_threshold,
        )
        sys.exit(0 if passed else 1)

    if args.adapters is None:
        raise ValueError("Provide --adapters, --gate, or --exhibition")

    adapters = resolve_adapter_list(args.adapters)
    log.info("Round-robin over %d adapters", len(adapters))
    report = tournament.run_round_robin(
        adapters, matches_per_pairing=args.matches_per_pairing
    )
    if args.output:
        report.to_json(args.output)


if __name__ == "__main__":
    main()
