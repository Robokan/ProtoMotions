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
  MaskedMimic base-link targets directly: the ball, by t+dt_k. No learning.

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
from typing import Optional, TYPE_CHECKING, Tuple

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
        self._report_closing_sum = 0.0
        self._report_prev_range = None
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
        # Closing speed: how fast the gap to the ball is actually shrinking.
        # This is the number that separates "running" from "taking its
        # time", and a dog merely moving cannot fake it. Re-throws are
        # dropped: a respawn jumps the range and would read as a huge
        # negative closing rate.
        if self._report_prev_range is not None:
            rate = (self._report_prev_range - rng) / self.control.env.dt
            sane = rate.abs() < 10.0
            if bool(sane.any()):
                self._report_closing_sum += float(rate[sane].mean())
        self._report_prev_range = rng.clone()
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
            f"closing speed "
            f"{self._report_closing_sum / self._report_steps:.2f} m/s, mean "
            f"time-to-catch {ttc_s:.1f} s",
            flush=True,
        )
        self._report_steps = 0
        self._report_catches = 0
        self._report_range_sum = 0.0
        self._report_closing_sum = 0.0
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
    """MaskedMimic conditioning that points the dog at a goal position.

    The target IS the ball, at each conditioned lead time, facing it. That is
    the whole rule. No speed cap, no waypoint ladder, no bearing dead band, no
    yaw-rate ramp: give the policy the position and the time and let it work
    out how fast to run and how to turn (Eric, 2026-09-20 -- an earlier
    version had all four and the dog crawled a wide arc, because each one was
    another place for me to quietly throttle it).

    Approach still slows on its own: the target is re-anchored every step, so
    as the dog closes, the remaining distance -- and with it the speed the
    target implies -- shrinks to nothing.

    Attributes:
        target_component: Key of the TargetControl holding the ball.
        stop_distance: How far short of the ball to aim. Default 0: aim AT the
            ball and let the two-foot success radius be tripped on the way in.
            Aiming at the boundary asks the dog to stop exactly on the
            threshold and parks it just outside -- measured, mean range stuck
            at 1.2-2.8 m against 4.4-5.3 m when aiming at the ball.
        max_speed: The robot's top speed, used ONLY to turn a distance into
            a deadline: horizon = range / max_speed. Not a cap -- nothing
            clamps what the dog attempts, and it may beat the deadline or miss
            it. Using the TOP speed rather than a comfortable one is the point
            (Eric): it makes the deadline the most urgent one physics allows,
            so the instruction is "get there as fast as you can" rather than
            "amble over". None measures it from the corpus (p99 of root speed
            -- 2.44 m/s on the go2, against a 2.60 m/s fastest clip mean and a
            4.10 m/s single-frame spike), which keeps it robot-agnostic.
        min_horizon_sec: Floor on the deadline, so an almost-reached ball
            does not collapse the target onto the robot's current position.
        max_horizon_sec: Ceiling on the deadline. Working mimic playback sits
            far further out than a fixed ladder suggests -- measured in steady
            state, the furthest slot has a median of 5.54 s and a p90 of
            22.2 s, and only 21.9% of all lead times are under 1 s. 15 s is
            comfortably inside that.
    """

    _target_: str = "protomotions.envs.control.ball_chase.MaskedMimicGoalControl"

    target_component: str = "ball"
    stop_distance: float = 0.0
    max_speed: Optional[float] = None
    min_horizon_sec: float = 0.5
    max_horizon_sec: float = 15.0


class MaskedMimicGoalControl(MaskedMimicSteeringControl):
    """Base-link targets that say "be at the ball", and nothing else."""

    config: MaskedMimicGoalControlConfig

    def __init__(self, config: MaskedMimicGoalControlConfig, env):
        super().__init__(config, env)
        # There is ONE target: the torso, at the ball, by the deadline.
        #
        # MaskedMimic's five conditioning slots are a property of the trained
        # checkpoint, not of this task -- NUM_FUTURE_STEPS=5 fixes the
        # observation widths and the transformer's sequence length, so losing
        # them means retraining. All this task can do is refuse to use them:
        # every slot but one is masked out of the attention mask AND of the
        # pose features, so nothing downstream sees more than a single target.
        self.masked_mimic_target_poses_masks[:] = False
        self.masked_mimic_target_poses_masks[:, -1] = True
        bodies = self.masked_mimic_target_bodies_masks.view(
            env.num_envs, config.num_masked_future_steps, -1
        )
        bodies[:, :-1, :] = False

    def _top_speed(self) -> float:
        """The robot's top speed, measured from the corpus unless configured."""
        if self.config.max_speed is not None:
            return max(float(self.config.max_speed), 1e-6)
        if self._corpus_max_speed is None:
            self._measure_height_fit()
        return max(self._corpus_max_speed or 1.0, 1e-6)

    def _goal_xy(self) -> Tensor:
        target = self.env.control_manager.components[self.config.target_component]
        return target._tar_pos[:, :2]

    def _lead_times(self) -> Tensor:
        """Give it the time to get there: the deadline scales with distance.

        The inherited fixed ladder (0.2 .. 1.0 s) demanded the same second of
        a ball 1 m away and one 8 m away -- the latter works out at 8 m/s,
        which nothing in the corpus can do, so every slot held an unreachable
        target and the bearing signal was swamped by targets that sat far
        ahead however the dog was pointing.

Measured over actual mimic playback -- where the policy works well --
        lead times have a median of 3.42 s and a p90 of 16.5 s, with only
        21.9% under 1 s, so that ladder lived entirely inside the shortest
        fifth of what the policy knows.

        The deadline is range / top speed: the most urgent one the robot could
        actually meet. A ball 5 m out at the go2's measured 2.44 m/s is asked
        for in 2.0 s, inside the 3.42 s playback median.
        """
        root_state = self.env.simulator.get_root_state()
        delta = self._goal_xy() - root_state.root_pos[:, :2]
        rng = torch.linalg.norm(delta, dim=-1)
        aim = (rng - self.config.stop_distance).clamp_min(0.0)
        horizon = (aim / self._top_speed()).clamp(
            self.config.min_horizon_sec, self.config.max_horizon_sec
        )
        # One deadline, not a ladder. The masked-out slots carry the same
        # value so there is a single number in play anywhere in this task.
        return horizon.unsqueeze(-1).expand(
            -1, self.config.num_masked_future_steps
        )

    def _rollout(
        self, root_pos: Tensor, root_rot: Tensor, lead: Tensor
    ) -> Tuple[Tensor, Tensor]:
        delta = self._goal_xy() - root_pos[:, :2]
        rng = torch.linalg.norm(delta, dim=-1, keepdim=True)
        direction = delta / rng.clamp_min(1e-6)

        aim = (rng - self.config.stop_distance).clamp_min(0.0)
        goal = root_pos[:, :2] + direction * aim
        heading = torch.atan2(direction[:, 1], direction[:, 0])

        steps = self.config.num_masked_future_steps
        # Same goal at every lead time. The near slots are the urgent ones and
        # the far slots the relaxed ones; the policy decides what it can do
        # about that.
        return (
            goal.unsqueeze(1).expand(-1, steps, -1),
            heading.unsqueeze(-1).expand(-1, steps),
        )

    def _command(self) -> Tensor:
        """The velocity the goal implies -- for the inherited readout only.

        Never drives anything. Reported against the nearest lead time, so a
        distant ball shows a large implied command; that is honest, it IS what
        the target asks for.
        """
        root_state = self.env.simulator.get_root_state()
        delta = self._goal_xy() - root_state.root_pos[:, :2]
        rng = torch.linalg.norm(delta, dim=-1)
        heading = rotations.calc_heading(root_state.root_rot, True)
        goal_heading = torch.atan2(delta[:, 1], delta[:, 0])
        error = torch.atan2(
            torch.sin(goal_heading - heading), torch.cos(goal_heading - heading)
        )
        horizon = self._lead_times()[:, -1].clamp_min(1e-6)
        return torch.stack(
            [rng / horizon, error / horizon, torch.zeros_like(rng)], dim=-1
        )
