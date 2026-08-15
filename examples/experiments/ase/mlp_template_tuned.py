# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""ASE LLC pretrain with IsaacLabASE-matched hyperparameters.

The atlas v6 pretrain froze at its discriminator equilibrium by epoch ~11k
(style reward flat at ~0.19 for 65k further epochs). Eric's IsaacLabASE
Template learned "most of the Reallusion moves" on the same data family —
with a more forgiving setup (rl_games_ase_cfg.yaml) plus hand-pruned data.
This experiment ports the trainer-side differences onto ase/mlp.py:

- gamma 0.99 -> 0.95            (Template LLC value)
- entropy_coef 0 -> 0.01        ("added to increase exploration")
- style:diversity reward mix 1:1 -> 2:1 (Template pre-scales disc rewards
  x2: disc_reward_w 0.5 * scale 2 vs enc 0.5)

Already matching (no change needed): disc grad penalty 5, weight decay
1e-4, logit reg 0.01, replay buffer 200k @ keep 0.01, latent dim 64,
latent steps 1-150, MLP sizes.

NOT ported (stage 2 if the plateau persists): the Template's REDUCED
discriminator features (~140/step vs our 493/step full max-coords — our
discriminator is ~3x better informed than the policy can defeat) and
rl_games' epsilon-greedy exploration.

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
    return cfg


def agent_config(robot_config, env_config, args: argparse.Namespace):
    cfg = _base.agent_config(robot_config, env_config, args)
    # Warm starts must pin the actor's input normalization: the EMA obs
    # normalizer re-centers within epochs while the weights stay intact,
    # which collapsed four raptor warm starts before it was found (see
    # warm-start-obs-norm-freeze). Applied whenever warm-starting.
    cfg.freeze_actor_obs_norm = bool(getattr(args, "checkpoint", None))
    cfg.gamma = 0.95
    cfg.entropy_coef = 0.01
    # Template effective mix: disc(x2 scale, w .5) : enc(w .5) = 2 : 1
    cfg.amp_parameters.discriminator_reward_w = 1.0
    cfg.ase_parameters.mi_reward_w = 0.5
    return cfg
