# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""MaskedMimic distillation that actually trains the sparse case.

Identical to transformer.py -- same model, same observations, same expert --
except for how often the body masks come out sparse.

Why. The steering and ball-chase tasks condition ONE body: the base link,
translation and rotation. Simulating the shipped masking sampler over 4M draws
(force_small_num_conditioned_bodies_prob 0.1, force_max 0.1, 13 conditionable
bodies on the go2):

    exactly one body conditioned                5.63%
    ...and that body is base_link               0.43%
    ...and translation+rotation (the tasks)     0.14%
    mean bodies conditioned per sample          7.2 of 13
    at least six bodies conditioned             61%

So the student is distilled on a dense diet and then driven with a single root
target. That is not a case it learned badly, it is one it barely saw -- and it
matches the shape of the failure measured on the chase: a ball AHEAD is
tracked at +0.94 path efficiency and 2 m/s, while a ball BEHIND produces
NEGATIVE efficiency, the dog falling back on the dominant "walk forward" prior
instead of pivoting. Seven different reshapings of the target failed to move
that band, which is what pointed at the training distribution rather than the
task.

0.4 lifts single-body samples from 5.6% to roughly 20%. Deliberately not
higher, and deliberately not fixed_conditioning pinned to base_link: the point
is to make root-only a case the student knows, not the only case it knows.
MaskedMimic still has to serve dense conditioning for everything else.

WARM START, not a fresh run: the masking knobs are env-side, so the network
and every observation width are unchanged and v4's weights stay valid. Warm
start rather than resume so v4 keeps its own history and stays as the
comparison baseline -- a true resume reloads the pickled configs and ignores
CLI overrides, so the only way to change the masking in place would be to
patch v4's pickle and silently rewrite what its first 10708 epochs were
trained under.

    python protomotions/train_agent.py --robot-name go2 \\
        --simulator isaaclab --physics physx --headless \\
        --motion-file data/motions/go2/go2_flat_mirrored_balanced.pt \\
        --experiment-path examples/experiments/masked_mimic/transformer_sparse.py \\
        --experiment-name go2_masked_mimic_v5 \\
        --checkpoint results/go2_masked_mimic_v4/last.ckpt \\
        --expert-model-path results/go2_tracker_v1/last.ckpt \\
        --num-envs 2048 --batch-size 8192 \\
        --training-max-steps 10000000000000

Judge it with the chase harness against v4: the number that has to move is the
120-180 deg path-efficiency band, currently -0.66.
"""

import argparse
import os

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig


# Single-body samples go from 5.6% to ~20% of the diet.
SPARSE_PROB = 0.4


def _transformer():
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transformer.py")
    spec = importlib.util.spec_from_file_location("masked_mimic_transformer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    _transformer().additional_experiment_arguments(parser)
    parser.add_argument(
        "--sparse-conditioning-prob", type=float, default=SPARSE_PROB,
        help="force_small_num_conditioned_bodies_prob. The shipped default is "
             "0.1, which leaves the single-body case at 5.6% of samples and "
             "base_link-only at 0.43%.")


def terrain_config(args):
    return _transformer().terrain_config(args)


def scene_lib_config(args):
    return _transformer().scene_lib_config(args)


def motion_lib_config(args):
    return _transformer().motion_lib_config(args)


def agent_config(robot_config: RobotConfig, env_config: EnvConfig, args):
    return _transformer().agent_config(robot_config, env_config, args)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    cfg = _transformer().env_config(robot_cfg, args)
    prob = getattr(args, "sparse_conditioning_prob", SPARSE_PROB)
    cfg.control_components["masked_mimic"].force_small_num_conditioned_bodies_prob = (
        prob
    )
    print(
        f"[sparse] force_small_num_conditioned_bodies_prob = {prob} "
        f"(transformer.py ships 0.1)",
        flush=True,
    )
    return cfg


def apply_inference_overrides(
    robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg,
    motion_lib_cfg, scene_lib_cfg, args,
):
    _transformer().apply_inference_overrides(
        robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg,
        motion_lib_cfg, scene_lib_cfg, args,
    )
