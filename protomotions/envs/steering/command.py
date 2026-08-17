# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Omnidirectional velocity-command ("game controller") steering task.

Each env tracks a command vector [target_forward_vel, target_turn_vel,
target_side_vel] expressed in the robot's heading frame. Commands random-walk
within bounds on a per-env schedule (deltas reflected at the bounds, so the
walk "bounces" instead of saturating), and the live target ramps toward the
commanded value at a per-env rate — the tracked target is always smooth, never
a step. Robot-agnostic: reads root state only.

Wiring (see examples/experiments/ase/steering_ase_hlc.py):

    control_components = {"steering_cmd": SteeringCommandControlConfig(...)}
    observation_components = {"task_obs": steering_command_obs_factory()}
    reward_components = {"steering_command_rew": steering_command_reward_factory(...)}
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.context_views import SteeringCommandContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.utils import rotations

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv

_GRAVITY_MPS2 = 9.81


# =============================================================================
# Control component
# =============================================================================


@dataclass
class SteeringCommandControlConfig(ControlComponentConfig):
    """Configuration for the velocity-command steering control component.

    Attributes:
        forward_vel_min/max: Forward-command bounds (m/s; min < 0 = backward).
        turn_vel_max: Yaw-rate command bound (rad/s, symmetric).
        side_vel_max: Lateral command bound (m/s, symmetric).
        heading_change_steps_min/max: Per-env resample interval (steps).
        rate_frac_min/max: Per-step ramp rate toward the commanded value,
            sampled per env per channel as a fraction of that channel's full
            range (0.02 = a full-range transition takes 50 steps).
        difficulty_epochs: Command magnitudes scale by
            clamp(epoch / difficulty_epochs, 0.2, 1.0). NOTE: nothing calls
            set_epoch() yet, so values above 1 pin difficulty at 0.2 forever
            — wire the hook before using the curriculum. Default 1 means full
            difficulty immediately, which is what the ported configs use.
    """

    _target_: str = "protomotions.envs.steering.command.SteeringCommandControl"

    forward_vel_min: float = -1.0
    forward_vel_max: float = 4.0
    turn_vel_max: float = 2.0
    side_vel_max: float = 1.0
    heading_change_steps_min: int = 125
    heading_change_steps_max: int = 175
    rate_frac_min: float = 0.02
    rate_frac_max: float = 0.08
    difficulty_epochs: int = 1


class SteeringCommandControl(ControlComponent):
    """Manages the per-env velocity command random walk and target ramping."""

    def __init__(self, config: SteeringCommandControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self.config: SteeringCommandControlConfig = config

        num_envs, device = self.env.num_envs, self.env.device
        # Bounds per channel: [forward, turn, side]
        self._lo = torch.tensor(
            [config.forward_vel_min, -config.turn_vel_max, -config.side_vel_max],
            device=device, dtype=torch.float,
        )
        self._hi = torch.tensor(
            [config.forward_vel_max, config.turn_vel_max, config.side_vel_max],
            device=device, dtype=torch.float,
        )
        # Random-walk state (the commanded value) and the ramped live target.
        self._desired = torch.zeros(num_envs, 3, device=device, dtype=torch.float)
        self._target = torch.zeros(num_envs, 3, device=device, dtype=torch.float)
        # Per-env per-channel ramp rates (units per step).
        self._rates = torch.zeros(num_envs, 3, device=device, dtype=torch.float)
        self._change_steps = torch.zeros(
            num_envs, device=device, dtype=torch.int64
        )
        self._difficulty = 1.0 if config.difficulty_epochs <= 1 else 0.2

    def set_epoch(self, current_epoch: int):
        """Difficulty curriculum hook, called by the HLC env adapter each epoch."""
        n = max(1, self.config.difficulty_epochs)
        self._difficulty = min(max(current_epoch / n, 0.2), 1.0)

    @staticmethod
    def _reflect(x: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
        """Reflect values back into [lo, hi] ("bounce" at the bounds)."""
        span = hi - lo
        y = torch.remainder(x - lo, 2.0 * span)
        y = torch.where(y > span, 2.0 * span - y, y)
        return lo + y

    def reset(self, env_ids: Tensor):
        """Random-walk the commands for the given envs (also the mid-episode
        resample path, mirroring SteeringControl)."""
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        device = self.env.device
        span = self._hi - self._lo

        # Deltas: forward uniform [-1, 1] x range; turn Beta(4,4)-shaped
        # (center-heavy) x range; side uniform [-1, 1] x range.
        fwd_delta = (2.0 * torch.rand(n, device=device) - 1.0) * span[0]
        beta = torch.distributions.Beta(
            torch.tensor(4.0, device=device), torch.tensor(4.0, device=device)
        )
        turn_delta = (2.0 * beta.sample((n,)) - 1.0) * span[1]
        side_delta = (2.0 * torch.rand(n, device=device) - 1.0) * span[2]
        delta = torch.stack([fwd_delta, turn_delta, side_delta], dim=-1)

        self._desired[env_ids] = self._reflect(
            self._desired[env_ids] + delta, self._lo, self._hi
        )

        frac = self.config.rate_frac_min + (
            self.config.rate_frac_max - self.config.rate_frac_min
        ) * torch.rand(n, 3, device=device)
        self._rates[env_ids] = frac * span

        change_steps = torch.randint(
            low=self.config.heading_change_steps_min,
            high=self.config.heading_change_steps_max,
            size=(n,),
            device=device,
            dtype=torch.int64,
        )
        progress = self.env.progress_buf[env_ids]
        is_env_reset = self.env.reset_buf[env_ids] | self.env.terminate_buf[env_ids]
        progress = torch.where(is_env_reset, torch.zeros_like(progress), progress)
        self._change_steps[env_ids] = progress + change_steps
        # Fresh episodes ramp up from standstill; mid-episode resamples keep
        # ramping from the current target.
        self._target[env_ids[is_env_reset]] = 0.0

    def step(self):
        resample_mask = self.env.progress_buf >= self._change_steps
        env_ids = resample_mask.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset(env_ids)

        goal = self._desired * self._difficulty
        step = torch.clamp(goal - self._target, -self._rates, self._rates)
        self._target += step

    def populate_context(self, ctx) -> None:
        ctx.steering_cmd = SteeringCommandContext(
            fwd_cmd=self._target[:, 0],
            turn_cmd=self._target[:, 1],
            side_cmd=self._target[:, 2],
        )


# =============================================================================
# Observation kernel
# =============================================================================


def compute_steering_command_obs(
    root_rot: Tensor,
    root_vel: Tensor,
    root_local_ang_vel: Tensor,
    fwd_cmd: Tensor,
    turn_cmd: Tensor,
    side_cmd: Tensor,
    w_last: bool = True,
) -> Tensor:
    """Command + gait proprioception observation, 12 dims:
    [fwd_cmd, turn_cmd, side_cmd, projected_gravity(3), root_ang_vel(3, body
    frame), heading-frame local linear velocity(3)].
    """
    from protomotions.envs.obs.humanoid import root_projected_gravity

    heading_inv = rotations.calc_heading_quat_inv(root_rot, w_last)
    local_vel = rotations.quat_rotate(heading_inv, root_vel, w_last)
    proj_gravity = root_projected_gravity(root_rot, w_last)

    return torch.cat(
        [
            fwd_cmd.unsqueeze(-1),
            turn_cmd.unsqueeze(-1),
            side_cmd.unsqueeze(-1),
            proj_gravity,
            root_local_ang_vel,
            local_vel,
        ],
        dim=-1,
    )


# =============================================================================
# Reward kernel
# =============================================================================


def compute_steering_command_reward(
    root_rot: Tensor,
    root_vel: Tensor,
    root_ang_vel: Tensor,
    fwd_cmd: Tensor,
    turn_cmd: Tensor,
    side_cmd: Tensor,
    forward_vel_min: float,
    forward_vel_max: float,
    turn_vel_max: float,
    side_vel_max: float,
    friction_mu: float = 0.6,
    w_last: bool = True,
) -> Tensor:
    """Mean of three exponential velocity-tracking terms in the heading frame.

    Targets are reshaped (not the robot penalized) before scoring:
    - safe velocity: forward target capped by the centripetal friction limit
      mu*g/|yaw_rate| during sharp turns;
    - appropriate side velocity: lateral target scaled linearly down to 10%
      as the forward target approaches forward_vel_max.
    """
    heading_inv = rotations.calc_heading_quat_inv(root_rot, w_last)
    local_vel = rotations.quat_rotate(heading_inv, root_vel, w_last)
    cur_fwd = local_vel[:, 0]
    cur_side = local_vel[:, 1]
    cur_turn = root_ang_vel[:, 2]

    safe_cap = friction_mu * _GRAVITY_MPS2 / turn_cmd.abs().clamp_min(1e-3)
    fwd_tgt = torch.minimum(fwd_cmd, safe_cap)
    side_scale = 1.0 - 0.9 * (fwd_tgt.abs() / forward_vel_max).clamp(0.0, 1.0)
    side_tgt = side_cmd * side_scale

    backward = fwd_tgt < 0
    fwd_scale = torch.where(
        backward,
        1.0 / max(abs(forward_vel_min), 1e-3),
        1.0 / forward_vel_max,
    )
    fwd_scale2 = torch.where(
        backward,
        torch.full_like(fwd_tgt, 2.0),
        torch.ones_like(fwd_tgt),
    )

    r_fwd = torch.exp(-(((cur_fwd - fwd_tgt) * fwd_scale) ** 2) * fwd_scale2)
    r_turn = torch.exp(-(((cur_turn - turn_cmd) / turn_vel_max) ** 2) * 2.0)
    r_side = torch.exp(-(((cur_side - side_tgt) / side_vel_max) ** 2) * 2.0)

    return (r_fwd + r_turn + r_side) / 3.0


# =============================================================================
# MdpComponent factories
# =============================================================================


def steering_command_obs_factory():
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    return MdpComponent(
        compute_func=compute_steering_command_obs,
        dynamic_vars={
            "root_rot": EnvContext.current.root_rot,
            "root_vel": EnvContext.current.root_vel,
            "root_local_ang_vel": EnvContext.current.root_local_ang_vel,
            "fwd_cmd": EnvContext.steering_cmd.fwd_cmd,
            "turn_cmd": EnvContext.steering_cmd.turn_cmd,
            "side_cmd": EnvContext.steering_cmd.side_cmd,
        },
        static_params={"w_last": True},
    )


def steering_command_reward_factory(
    forward_vel_min: float,
    forward_vel_max: float,
    turn_vel_max: float,
    side_vel_max: float,
    friction_mu: float = 0.6,
    weight: float = 1.0,
):
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    return MdpComponent(
        compute_func=compute_steering_command_reward,
        dynamic_vars={
            "root_rot": EnvContext.current.root_rot,
            "root_vel": EnvContext.current.root_vel,
            "root_ang_vel": EnvContext.current.root_ang_vel,
            "fwd_cmd": EnvContext.steering_cmd.fwd_cmd,
            "turn_cmd": EnvContext.steering_cmd.turn_cmd,
            "side_cmd": EnvContext.steering_cmd.side_cmd,
        },
        static_params={
            "forward_vel_min": forward_vel_min,
            "forward_vel_max": forward_vel_max,
            "turn_vel_max": turn_vel_max,
            "side_vel_max": side_vel_max,
            "friction_mu": friction_mu,
            "w_last": True,
            "weight": weight,
        },
    )


__all__ = [
    "SteeringCommandContext",
    "SteeringCommandControlConfig",
    "SteeringCommandControl",
    "compute_steering_command_obs",
    "compute_steering_command_reward",
    "steering_command_obs_factory",
    "steering_command_reward_factory",
]
