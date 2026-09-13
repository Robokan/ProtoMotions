# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""ASE LLC pretrain: atlas-specific fixes on top of ase/mlp.py.

HYPERPARAMETER PORTS FROM THE TEMPLATE ARE REVERTED (2026-09-03, Eric:
"we really should be using the protomotion defaults not the port"). This
file originally copied three trainer values out of the IsaacLabASE
Template's rl_games_ase_cfg.yaml to break the v6 discriminator plateau
(style reward flat at ~0.19). Audited 2026-09-03, all three were unsound:

- entropy_coef 0.005 -> 0.01: taken from the ONE Template config that
  sets it (sword_and_shield); all three HUMANOID ASE configs use 0.0, and
  the Template's other nonzero values are on hrl/HLC configs where the
  policy emits latents, not joint targets. Inert here regardless
  (learnable_std=False gates it). Now 0.0 -- see the assignment below.
- gamma 0.99 -> 0.95: the NUMBER was ported, the UNITS were not. The
  Template humanoid runs sim.dt 1/200 with decimation 4 = 50 Hz control,
  so 0.95 buys it a 0.40 s horizon. Atlas runs fps 120 / decimation 4 =
  30 Hz, where the same 0.95 is 0.67 s. (Their episode_length_s 6.0 is
  300 steps at 50 Hz; our 300 steps are 10 s.) Same class of error as the
  Froude retiming incident. Reverted to the repo default 0.99.
- discriminator_reward_w 0.5 -> 1.0: this was the whole of the claimed
  "style:diversity 1:1 -> 2:1" change; the accompanying mi_reward_w = 0.5
  line was a no-op, since ase/mlp.py already sets 0.5. Doubling it
  amplifies a discriminator signal that is measurably saturated (agent_acc
  0.988, style reward flat across 45k epochs) while relatively halving the
  encoder reward, which is the one term still moving. Reverted to 0.5.

NOT a port, deliberately kept (see the functions below): the power-penalty
removal, the atlas root-height obs zeroing, and the warm-start obs-norm
freeze. Those are ProtoMotions-side fixes with their own rationale.

For the record, the disc-vs-actor learning rates (actor 2e-5, disc/critic
1e-4) are NOT from the Template -- ase/mlp.py:423 and amp/mlp.py:244 both
set them, so the 5:1 ratio is ProtoMotions' own design. Likewise
num_mini_epochs=1 is the base_agent default, not a missed port.

Usage: same as ase/mlp.py (train_agent.py --experiment-path this file).
"""
import argparse

from examples.experiments.ase.mlp import (  # noqa: F401  (loader re-exports)
    terrain_config,
    scene_lib_config,
    motion_lib_config,
    env_config,
    apply_inference_overrides,
)
from examples.experiments.ase import mlp as _base


def _zero_root_height_obs(component) -> None:
    """Keep the 1-D root-height channel but zero it (dim stays the same)."""
    if component is not None and hasattr(component, "static_params"):
        component.static_params["root_height_obs"] = False


def env_config(robot_cfg, args: argparse.Namespace):
    """Base env config MINUS the power penalty.

    Atlas imitates MOCAP -- a real human already moves energy-efficiently,
    so imitation embeds the energy prior and pure AMP economics apply: every
    per-step reward >= 0, survival weakly dominates termination (Eric,
    2026-08-14, after the utahraptor learned to fall on purpose when the
    converged style reward 0.104 met the -0.097 power penalty).
    """
    cfg = _base.env_config(robot_cfg, args)
    if hasattr(cfg, "reward_components") and cfg.reward_components:
        cfg.reward_components.pop("pow_rew", None)
    # Atlas: do not observe root height. The max-coords slot is zeroed so
    # the actor/disc input size stays 493 and a v16 checkpoint can load.
    if getattr(args, "robot_name", None) == "atlas":
        for name in ("max_coords_obs", "historical_max_coords_obs"):
            _zero_root_height_obs(cfg.observation_components.get(name))
    return cfg


def agent_config(robot_config, env_config, args: argparse.Namespace):
    cfg = _base.agent_config(robot_config, env_config, args)
    # Warm starts must pin the actor's input normalization: the EMA obs
    # normalizer re-centers within epochs while the weights stay intact,
    # which collapsed four raptor warm starts before it was found (see
    # warm-start-obs-norm-freeze). Applied whenever warm-starting.
    cfg.freeze_actor_obs_norm = bool(getattr(args, "checkpoint", None))
    # gamma, entropy_coef and discriminator_reward_w are left at the
    # ProtoMotions defaults (0.99 / 0.005-gated-off / 0.5). See the module
    # docstring for why each Template port was reverted on 2026-09-03.
    if getattr(args, "robot_name", None) == "atlas":
        refs = getattr(cfg, "reference_obs_components", None) or {}
        _zero_root_height_obs(refs.get("historical_max_coords_obs"))
    return cfg


def additional_experiment_arguments(parser):
    # train_agent.py calls exactly ONE of these hooks, so defining it here
    # shadows the base file's -- without this delegation --disc-term-threshold
    # and --disc-term-decay-epochs silently vanish and the launch dies on an
    # "unrecognized arguments" error.
    _base.additional_experiment_arguments(parser)
    parser.add_argument(
        "--sim-dr", action="store_true",
        help="Engine-transfer domain randomization: per-env constant PD-target "
             "offsets (actuator calibration error, robust to implicit-vs-"
             "explicit PD differences) and foot friction/restitution buckets "
             "(robust to contact-model differences). For policies that must "
             "also run under Newton/MuJoCo, not just IsaacLab.")
    parser.add_argument(
        "--sim-dr-action-noise", type=float, default=0.02,
        help="Half-range (radians) of the per-env constant PD-target offset "
             "when --sim-dr is on. 0.02 rad ~= 1.1 deg per joint.")


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args: argparse.Namespace):
    if not getattr(args, "sim_dr", False):
        return
    from protomotions.simulator.base_simulator.config import (
        DomainRandomizationConfig,
        ActionNoiseDomainRandomizationConfig,
        FrictionDomainRandomizationConfig,
    )
    a = float(getattr(args, "sim_dr_action_noise", 0.02))
    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-a, a),
            dof_names=[".*"],
        ),
        friction=FrictionDomainRandomizationConfig(
            # Feet are the contact interface; engine transfer lives or dies
            # on ground contact. Ranges are the repo defaults (0.5-1.5 mu).
            body_names=["Foot_.*"],
        ),
    )
