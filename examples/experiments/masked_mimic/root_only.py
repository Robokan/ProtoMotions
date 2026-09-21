# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Clip playback with ONLY the torso conditioned -- the chase's exact setup.

Plain MaskedMimic playback, except the body mask is pinned instead of sampled:
base_link, translation AND rotation, on every step. That is precisely what the
steering and ball-chase tasks feed the policy, applied here to a real motion
the policy is known to track well.

It is the sharp version of the diagnostic. The masking sampler shows that
conditioning only base_link with both constraints is about 0.14% of training
samples, so the tasks run almost entirely outside the distilled distribution.
This isolates that one variable: same clip, same policy, everything else
normal, only the conditioning forced.

    Read it as:
      tracks the clip fine  -> sparse conditioning is NOT the problem, and the
                               chase's ball-behind failure is something else.
      falls apart           -> the tasks are running in a regime the student
                               barely saw, and transformer_sparse.py is the fix.

    python protomotions/inference_agent.py \\
        --checkpoint results/go2_masked_mimic_v4/last.ckpt \\
        --experiment-path examples/experiments/masked_mimic/root_only.py \\
        --simulator isaaclab --physics physx --num-envs 4

Compare against the same command on transformer.py, which samples masks
normally (7.2 of 13 bodies on average). The RootSpeedProbe is wired in, so
there is a number next to the visual impression.
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
    parser.add_argument(
        "--condition-body", type=str, default=None,
        help="Body to pin the conditioning to. Defaults to the robot's anchor "
             "body (base_link on the go2), which is what the tasks condition.")
    parser.add_argument(
        "--constraint-state", type=int, default=1, choices=[0, 1, 2],
        help="0 = translation only, 1 = translation AND rotation (what the "
             "tasks use), 2 = rotation only.")


def terrain_config(args):
    return _transformer().terrain_config(args)


def scene_lib_config(args):
    return _transformer().scene_lib_config(args)


def motion_lib_config(args):
    return _transformer().motion_lib_config(args)


def agent_config(robot_config: RobotConfig, env_config: EnvConfig, args):
    return _transformer().agent_config(robot_config, env_config, args)


def _pin_conditioning(cfg: EnvConfig, robot_cfg: RobotConfig, args) -> None:
    from protomotions.envs.control.masked_mimic_control import FixedBodyCondition
    from protomotions.envs.control.speed_probe import RootSpeedProbeConfig

    body = getattr(args, "condition_body", None) or robot_cfg.anchor_body_name
    state = getattr(args, "constraint_state", 1)
    assert body in robot_cfg.trackable_bodies_subset, (
        f"{body!r} is not in trackable_bodies_subset, so it cannot be "
        f"conditioned: {robot_cfg.trackable_bodies_subset}"
    )

    mm = cfg.control_components["masked_mimic"]
    mm.fixed_conditioning = [
        FixedBodyCondition(body_name=body, constraint_state=state)
    ]
    cfg.control_components["speed_probe"] = RootSpeedProbeConfig(label="root-only")
    kind = {0: "translation only", 1: "translation+rotation", 2: "rotation only"}[state]
    print(f"[root-only] conditioning pinned to {body!r}, {kind}", flush=True)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    cfg = _transformer().env_config(robot_cfg, args)
    _pin_conditioning(cfg, robot_cfg, args)
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
        _pin_conditioning(env_cfg, robot_cfg, args)
