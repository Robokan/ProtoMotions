# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Chase-the-ball task: a red ball, and a dog whose job is to reach it.

Two pieces, deliberately separate:

* ``BallChaseCommandSource`` -- where the ball goes. Extends the ordinary
  random target sampler with the thing that makes it a CHASE rather than a
  waypoint: the ball respawns as soon as the torso gets within the success
  radius, so the task never ends and the robot is always pursuing. Counts
  catches per env.

* ``MaskedMimicGoalControl`` -- where the DOG is told to be. It emits
  MaskedMimic base-link targets directly: "put my torso here, by t+dt_k",
  walking a ladder of waypoints along the line to the ball. No learning.

Going straight to positional targets is the point. MaskedMimic is natively
conditioned on poses-at-times, so routing a goal through a velocity command
and re-integrating it back into positions (which is what the steering harness
does) is a lossy round trip through a representation that cannot express
"be 2 feet from the ball" at all -- and the velocity route needs pursuit gains
to tune, where the direct one needs none: the waypoint ladder saturates at the
goal, so arrival slows by construction.

It is also the format a VLA will emit, which matters for what this is FOR.

That split is the point for data collection. The pursuit controller is a
demonstrator: it produces (what the robot sees, where the ball is) ->
(MaskedMimic targets) pairs at whatever scale you want to run envs, which is
exactly the supervision a VLA needs to learn to emit those targets itself.
Swap BallPursuitControl for a learned high-level policy later and the rest of
the task -- ball, success radius, markers, obs, reward -- is unchanged.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple

import torch
from torch import Tensor

from protomotions.envs.steering.masked_mimic_command import (
    MaskedMimicSteeringControl,
    MaskedMimicSteeringControlConfig,
)
from protomotions.envs.control.target_control import (
    RandomTargetCommandSource,
    RandomTargetCommandSourceConfig,
    TargetControlConfig,
)
from protomotions.utils import rotations

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


# 2 feet, the brief. Kept as a named constant because it is a task definition,
# not a tuning knob.
TWO_FEET_M = 0.6096


@dataclass
class BallChaseCommandSourceConfig(RandomTargetCommandSourceConfig):
    """Ball placement: random, and respawned the moment it is caught."""

    _target_: str = (
        "protomotions.envs.control.ball_chase.BallChaseCommandSource"
    )

    # Re-throw distance. Far enough that catching it means travelling, close
    # enough that the robot can plausibly see it.
    tar_dist_min: float = 2.0
    tar_dist_max: float = 8.0
    # Give up and re-throw if the ball has not been caught in this long. The
    # base class' tar_change_time_* do this; the defaults here are much longer
    # because a chase should be allowed to take a while.
    tar_change_time_min: float = 20.0
    tar_change_time_max: float = 30.0
    # Print catches / mean range / mean time-to-catch every N steps. This is
    # the only thing that distinguishes a dog that is CHASING from one merely
    # wandering near a ball. 0 disables it.
    report_every_steps: int = 250


class BallChaseCommandSource(RandomTargetCommandSource):
    """Random ball placement that respawns on a catch."""

    config: BallChaseCommandSourceConfig

    def __init__(self, config: BallChaseCommandSourceConfig, control):
        super().__init__(config, control)
        self.catches = torch.zeros(
            control.env.num_envs, dtype=torch.long, device=control.env.device
        )
        # Per-env step count since the current ball appeared, so "how long did
        # that catch take" is recoverable for logging and for dataset filtering.
        self.steps_since_throw = torch.zeros(
            control.env.num_envs, dtype=torch.long, device=control.env.device
        )
        self.last_catch_steps = torch.zeros(
            control.env.num_envs, dtype=torch.long, device=control.env.device
        )
        self._report_steps = 0
        self._report_catches = 0
        self._report_range_sum = 0.0
        self._report_catch_steps = []

    def reset(self, env_ids: Tensor) -> None:
        super().reset(env_ids)
        self.steps_since_throw[env_ids] = 0

    def step(self) -> None:
        # Timeout re-throw (the base class' behaviour).
        super().step()

        self.steps_since_throw += 1
        rng = self.control.distance_to_target()
        self._report(rng)
        caught = rng <= self.control.config.tar_proximity_threshold
        ids = caught.nonzero(as_tuple=False).flatten()
        if len(ids) == 0:
            return
        self.catches[ids] += 1
        self.last_catch_steps[ids] = self.steps_since_throw[ids]
        self._report_catch_steps.extend(self.steps_since_throw[ids].tolist())
        self._report_catches += len(ids)
        self.steps_since_throw[ids] = 0
        self._set_random_target(ids)

    def _report(self, rng: Tensor) -> None:
        every = self.config.report_every_steps
        if every <= 0:
            return
        self._report_steps += 1
        self._report_range_sum += float(rng.mean())
        if self._report_steps < every:
            return
        n_env = self.control.env.num_envs
        secs = self._report_steps * self.control.env.dt
        per_min = self._report_catches / max(secs, 1e-6) / n_env * 60.0
        ttc = self._report_catch_steps
        ttc_s = (sum(ttc) / len(ttc) * self.control.env.dt) if ttc else float("nan")
        print(
            f"[ball-chase] {self._report_catches} catches in {secs:.0f}s x "
            f"{n_env} dogs ({per_min:.1f}/min/dog), mean range "
            f"{self._report_range_sum / self._report_steps:.2f} m, mean "
            f"time-to-catch {ttc_s:.1f} s",
            flush=True,
        )
        self._report_steps = 0
        self._report_catches = 0
        self._report_range_sum = 0.0
        self._report_catch_steps = []

    def _sample_heading_relative_target(
        self, env_ids: Tensor, root_pos: Tensor, root_rot: Tensor
    ) -> None:
        """Throw the ball to an annulus around the robot, not a disc.

        The base sampler draws the distance from U(0, tar_dist_max), so a
        meaningful share of throws land inside the success radius and are
        caught instantly -- the robot would "win" without moving, and a
        dataset built from it would be mostly standing still.
        """
        num = len(env_ids)
        device = self.control.env.device
        lo = max(self.config.tar_dist_min, self.control.config.tar_proximity_threshold)
        dist = lo + torch.rand(num, device=device) * max(
            self.config.tar_dist_max - lo, 0.0
        )
        angle = torch.rand(num, device=device) * 2 * torch.pi

        local = torch.zeros(num, 3, device=device)
        local[:, 0] = dist * torch.cos(angle)
        local[:, 1] = dist * torch.sin(angle)
        heading_rot = rotations.calc_heading_quat(root_rot, w_last=True)
        world = rotations.quat_rotate(heading_rot, local, w_last=True)
        self.control._tar_pos[env_ids, :2] = root_pos[:, :2] + world[:, :2]

        if self._target_bounds is not None:
            x_min, x_max, y_min, y_max = self._target_bounds
            self.control._tar_pos[env_ids, 0] = self.control._tar_pos[
                env_ids, 0
            ].clamp(x_min, x_max)
            self.control._tar_pos[env_ids, 1] = self.control._tar_pos[
                env_ids, 1
            ].clamp(y_min, y_max)
        self.control._update_target_heights(env_ids)


def ball_chase_target_config(
    success_radius: float = TWO_FEET_M,
    throw_min: float = 2.0,
    throw_max: float = 8.0,
) -> TargetControlConfig:
    """A red ball, caught at success_radius, re-thrown on every catch."""
    return TargetControlConfig(
        tar_proximity_threshold=success_radius,
        marker_color=(1.0, 0.1, 0.1),
        marker_size="huge",
        proximity_planar=True,
        # The chase is the task; falling over is failure, but wandering is not.
        enable_fall_termination=False,
        command_source=BallChaseCommandSourceConfig(
            tar_dist_min=throw_min,
            tar_dist_max=throw_max,
        ),
    )


@dataclass
class MaskedMimicGoalControlConfig(MaskedMimicSteeringControlConfig):
    """MaskedMimic conditioning that walks the dog to a goal position.

    Replaces the velocity roll-out of the steering harness with a goal-seeking
    one: the k-th conditioned target is placed along the line to the ball, as
    far as the dog could plausibly travel in dt_k, and the whole ladder
    saturates once it reaches the aim point. Nothing tells the robot to keep
    going after it arrives, so approach slows on its own.

    Attributes:
        target_component: Key of the TargetControl holding the ball.
        stop_distance: How far short of the ball the waypoint ladder aims.
            Default 0: aim AT the ball and let the two-foot success radius be
            tripped on the way in. Aiming at the success boundary instead asks
            the dog to stop exactly on the threshold, and since the policy
            realises only about two thirds of a commanded motion it then parks
            just outside it -- measured: mean range stuck at 1.2-2.8 m and
            23-72 s per catch. Non-zero only to hold station off the ball.
        max_speed: Ceiling on how far ahead a waypoint may be placed, as
            metres per second of lead time. This is what stops a ball 8 m away
            from generating a "be there in 0.5 s" target the corpus has no
            answer for.
        max_yaw_rate: Ceiling on how far the commanded facing may swing per
            second of lead time, so the yaw targets ramp toward the ball
            rather than snapping 180 degrees in the nearest slot.
    """

    _target_: str = "protomotions.envs.control.ball_chase.MaskedMimicGoalControl"

    target_component: str = "ball"
    stop_distance: float = 0.0
    max_speed: float = 1.6
    max_yaw_rate: float = 1.5


class MaskedMimicGoalControl(MaskedMimicSteeringControl):
    """Base-link targets that lead to the ball, emitted directly."""

    config: MaskedMimicGoalControlConfig

    def _goal_xy(self) -> Tensor:
        target = self.env.control_manager.components[self.config.target_component]
        return target._tar_pos[:, :2]

    def _bearing_and_range(self, root_pos: Tensor, root_rot: Tensor):
        """Range to the aim point and the heading error toward the ball."""
        delta = self._goal_xy() - root_pos[:, :2]
        rng = torch.linalg.norm(delta, dim=-1, keepdim=True)
        direction = delta / rng.clamp_min(1e-6)
        goal_heading = torch.atan2(direction[:, 1], direction[:, 0]).unsqueeze(-1)
        heading = rotations.calc_heading(root_rot, True).unsqueeze(-1)
        # Wrapped to (-pi, pi] so a ball just behind the left shoulder is a
        # small right turn, not an almost-full circle to the left.
        error = torch.atan2(
            torch.sin(goal_heading - heading), torch.cos(goal_heading - heading)
        )
        aim = (rng - self.config.stop_distance).clamp_min(0.0)
        return direction, aim, heading, error

    def _rollout(self, root_pos: Tensor, root_rot: Tensor) -> Tuple[Tensor, Tensor]:
        direction, aim, heading, error = self._bearing_and_range(root_pos, root_rot)
        t = self._offsets.unsqueeze(0)  # [1, steps]

        # Facing ramps toward the ball at a bounded rate.
        swing = (self.config.max_yaw_rate * t).clamp(max=1e3)
        target_heading = heading + error.sign() * torch.minimum(error.abs(), swing)

        # Distance travelled by dt_k, saturating at the aim point. Scaled by
        # how well the dog is already facing the ball: planning to translate
        # toward a goal you have your back to just asks for a walk backwards,
        # so it turns first and the ladder collapses to a pivot.
        align = torch.cos(error).clamp_min(0.0)
        travel = torch.minimum(aim, self.config.max_speed * t * align)
        target_xy = root_pos[:, :2].unsqueeze(1) + direction.unsqueeze(1) * (
            travel.unsqueeze(-1)
        )
        return target_xy, target_heading

    def _command(self) -> Tensor:
        """The velocity this goal implies -- for the tracking readout only.

        Nothing consumes it to drive the robot; it exists so the inherited
        commanded-vs-achieved report still means something when no steering
        component is in the loop.
        """
        root_state = self.env.simulator.get_root_state()
        _, aim, _, error = self._bearing_and_range(
            root_state.root_pos, root_state.root_rot
        )
        lead = float(self._offsets[0])
        speed = torch.minimum(
            aim.squeeze(-1) / lead,
            torch.full_like(aim.squeeze(-1), self.config.max_speed),
        ) * torch.cos(error.squeeze(-1)).clamp_min(0.0)
        turn = (error.squeeze(-1) / lead).clamp(
            -self.config.max_yaw_rate, self.config.max_yaw_rate
        )
        return torch.stack([speed, turn, torch.zeros_like(speed)], dim=-1)
