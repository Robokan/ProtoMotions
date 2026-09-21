# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Plain MaskedMimic clip playback, instrumented with the root-speed probe.

Identical to transformer.py in every respect except that it reports the
achieved root speed distribution, so playback can be compared against the ball
chase on the same numbers. That comparison is the thing that says whether a
slow chase is the POLICY's ceiling or something out-of-distribution about the
targets the chase synthesises.

    python protomotions/inference_agent.py \\
        --checkpoint results/go2_masked_mimic_v4/last.ckpt \\
        --experiment-path examples/experiments/masked_mimic/speed_probe.py \\
        --simulator isaaclab --physics physx --headless --num-envs 16
"""

import argparse
import os

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.base_env.config import EnvConfig


def _transformer():
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transformer.py")
    spec = importlib.util.spec_from_file_location("masked_mimic_transformer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    _transformer().additional_experiment_arguments(parser)


def terrain_config(args):
    return _transformer().terrain_config(args)


def scene_lib_config(args):
    return _transformer().scene_lib_config(args)


def motion_lib_config(args):
    return _transformer().motion_lib_config(args)


def agent_config(robot_config: RobotConfig, env_config: EnvConfig, args):
    return _transformer().agent_config(robot_config, env_config, args)


def _install_probe(cfg: EnvConfig) -> None:
    from protomotions.envs.control.speed_probe import RootSpeedProbeConfig

    cfg.control_components = dict(cfg.control_components)
    cfg.control_components["speed_probe"] = RootSpeedProbeConfig(label="mimic-playback")


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    cfg = _transformer().env_config(robot_cfg, args)
    _install_probe(cfg)
    return cfg


def apply_inference_overrides(
    robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg,
    motion_lib_cfg, scene_lib_cfg, args,
):
    _transformer().apply_inference_overrides(
        robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg,
        motion_lib_cfg, scene_lib_cfg, args,
    )
    if env_cfg is not None:
        _install_probe(env_cfg)
