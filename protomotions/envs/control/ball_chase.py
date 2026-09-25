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

Which is why the chase can also be run UNPRIVILEGED (``--unprivileged``): the
demonstrator is then restricted to what a front camera would show it, and when
the ball is not in frame it believes -- correctly, most of the time -- that the
ball is behind, which makes it turn and look. A demonstrator that tracks a ball
through the back of its own head teaches a student that cannot. See
``MaskedMimicGoalControlConfig.fov_deg``.

That split is the point for data collection. The pursuit controller is a
demonstrator: it produces (what the robot sees, where the ball is) ->
(MaskedMimic targets) pairs at whatever scale you want to run envs, which is
exactly the supervision a VLA needs to learn to emit those targets itself.
Swap BallPursuitControl for a learned high-level policy later and the rest of
the task -- ball, success radius, markers, obs, reward -- is unchanged.
"""

import math
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

# Range bands for the closing-speed readout. An average hides a chase that
# sprints the first four metres and then crawls the last one.
_RANGE_BANDS = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 99.0)]

# Bearing-error bands (degrees) for the path-efficiency readout. Efficiency is
# closing rate divided by actual speed: 1.0 is running straight at the ball,
# 0.0 is moving without shortening the gap at all. It separates "slow" from
# "fast but pointed wrong", which a speed number alone cannot -- the chase runs
# at a median 1.86 m/s while the gap closes at ~1.2, so about a third of the
# motion is going somewhere other than the ball.
_BEARING_BANDS = [(0.0, 30.0), (30.0, 60.0), (60.0, 120.0), (120.0, 180.0)]


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
    # Moving ball. Each throw also draws a velocity -- uniform random
    # direction, speed in [ball_speed_min, ball_speed_max] m/s -- which the
    # ball keeps until it is caught or re-thrown. The dog does not chase the
    # ball's current position: MaskedMimicGoalControl aims at where the ball
    # WILL be when the deadline arrives (an intercept), see _set_deadline.
    # With target_bounds set the ball reflects off the edges; unbounded
    # otherwise.
    moving: bool = False
    ball_speed_min: float = 0.5
    ball_speed_max: float = 2.0
    # A moving ball occasionally changes direction (and speed): a Poisson
    # process with this mean interval in seconds. Each change is a new plan
    # for the dog -- intercept and deadline are re-solved once. 0 disables.
    ball_turn_mean_sec: float = 5.0


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
        self._report_band_sum = [0.0] * len(_RANGE_BANDS)
        self._report_band_n = [0] * len(_RANGE_BANDS)
        self._eff_sum = [0.0] * len(_BEARING_BANDS)
        self._eff_spd = [0.0] * len(_BEARING_BANDS)
        self._eff_n = [0] * len(_BEARING_BANDS)
        self._report_catch_steps = []
        # Ball velocity (m/s, planar); zeros unless config.moving.
        self._tar_vel = torch.zeros(
            control.env.num_envs, 2, device=control.env.device
        )
        # Incremented on every throw AND every direction change. The goal
        # control keys its deadline to this, not to the ball's position, so a
        # moving ball does not restart the deadline every step -- only when
        # the plan genuinely changes.
        self.plan_id = torch.zeros(
            control.env.num_envs, dtype=torch.long, device=control.env.device
        )

    def reset(self, env_ids: Tensor) -> None:
        super().reset(env_ids)
        self.steps_since_throw[env_ids] = 0

    def _set_random_target(self, env_ids: Tensor) -> None:
        """A throw: place the ball (base behaviour), then give it a velocity."""
        super()._set_random_target(env_ids)
        self.plan_id[env_ids] += 1
        if not self.config.moving:
            self._tar_vel[env_ids] = 0.0
            return
        self._sample_velocity(env_ids)

    def _sample_velocity(self, env_ids: Tensor) -> None:
        num = len(env_ids)
        device = self.control.env.device
        speed = self.config.ball_speed_min + torch.rand(num, device=device) * max(
            self.config.ball_speed_max - self.config.ball_speed_min, 0.0
        )
        angle = torch.rand(num, device=device) * 2 * torch.pi
        self._tar_vel[env_ids, 0] = speed * torch.cos(angle)
        self._tar_vel[env_ids, 1] = speed * torch.sin(angle)

    def _advance_ball(self) -> None:
        """Integrate the moving ball one control step; sometimes change course."""
        control = self.control
        if self.config.ball_turn_mean_sec > 0:
            p = control.env.dt / self.config.ball_turn_mean_sec
            turn = torch.rand(control.env.num_envs, device=control.env.device) < p
            ids = turn.nonzero(as_tuple=False).flatten()
            if len(ids) > 0:
                self._sample_velocity(ids)
                self.plan_id[ids] += 1   # the dog must re-solve its intercept
        control._tar_pos[:, :2] += self._tar_vel * control.env.dt
        if self._target_bounds is not None:
            x_min, x_max, y_min, y_max = self._target_bounds
            pos = control._tar_pos
            for axis, (lo, hi) in enumerate(((x_min, x_max), (y_min, y_max))):
                out = (pos[:, axis] < lo) | (pos[:, axis] > hi)
                self._tar_vel[out, axis] = -self._tar_vel[out, axis]
                pos[:, axis] = pos[:, axis].clamp(lo, hi)
        control._update_target_heights(
            torch.arange(control.env.num_envs, device=control.env.device)
        )

    def step(self) -> None:
        # Timeout re-throw (the base class' behaviour).
        super().step()
        if self.config.moving:
            self._advance_ball()

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

    def _accumulate_efficiency(self, rate: Tensor, sane: Tensor) -> None:
        """How much of the dog's motion actually shortens the gap, by bearing.

        High speed with a low closing rate means it is running hard in the
        wrong direction -- arcing toward the ball rather than pivoting and
        then sprinting. Speed alone cannot tell those apart.
        """
        env = self.control.env
        root = env.simulator.get_root_state()
        speed = root.root_vel[:, :2].norm(dim=-1)
        delta = self.control._tar_pos[:, :2] - root.root_pos[:, :2]
        heading = rotations.calc_heading(root.root_rot, True)
        goal_heading = torch.atan2(delta[:, 1], delta[:, 0])
        bearing = torch.rad2deg(
            torch.atan2(
                torch.sin(goal_heading - heading),
                torch.cos(goal_heading - heading),
            ).abs()
        )
        moving = sane & (speed > 0.2)
        for i, (lo, hi) in enumerate(_BEARING_BANDS):
            m = moving & (bearing >= lo) & (bearing < hi)
            if bool(m.any()):
                self._eff_sum[i] += float((rate[m] / speed[m]).sum())
                self._eff_spd[i] += float(speed[m].sum())
                self._eff_n[i] += int(m.sum())

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
            for i, (lo, hi) in enumerate(_RANGE_BANDS):
                m = sane & (rng >= lo) & (rng < hi)
                if bool(m.any()):
                    self._report_band_sum[i] += float(rate[m].sum())
                    self._report_band_n[i] += int(m.sum())
            self._accumulate_efficiency(rate, sane)
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
        bands = "  ".join(
            (f"{lo:g}-{hi:g}m {self._report_band_sum[i]/self._report_band_n[i]:5.2f}"
             if self._report_band_n[i] else f"{lo:g}-{hi:g}m    --")
            for i, (lo, hi) in enumerate(_RANGE_BANDS)
        )
        print(f"[ball-chase]   closing m/s by range: {bands}", flush=True)
        eff = "  ".join(
            (f"{lo:g}-{hi:g}deg eff {self._eff_sum[i]/self._eff_n[i]:+.2f} "
             f"spd {self._eff_spd[i]/self._eff_n[i]:.2f}"
             if self._eff_n[i] else f"{lo:g}-{hi:g}deg --")
            for i, (lo, hi) in enumerate(_BEARING_BANDS)
        )
        print(f"[ball-chase]   by bearing: {eff}", flush=True)
        self._eff_sum = [0.0] * len(_BEARING_BANDS)
        self._eff_spd = [0.0] * len(_BEARING_BANDS)
        self._eff_n = [0] * len(_BEARING_BANDS)
        self._report_steps = 0
        self._report_catches = 0
        self._report_range_sum = 0.0
        self._report_closing_sum = 0.0
        self._report_band_sum = [0.0] * len(_RANGE_BANDS)
        self._report_band_n = [0] * len(_RANGE_BANDS)
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
    moving: bool = False,
    ball_speed_min: float = 0.5,
    ball_speed_max: float = 2.0,
    ball_turn_mean_sec: float = 5.0,
) -> TargetControlConfig:
    """A red ball, caught at success_radius, re-thrown on every catch."""
    return TargetControlConfig(
        tar_proximity_threshold=success_radius,
        marker_color=(1.0, 0.1, 0.1),
        marker_size="huge",
        # A real prim per env so the ball casts a shadow (instancer markers
        # do not under RTX). The chase runs at viewer env counts.
        marker_cast_shadows=True,
        # "huge" is a 0.12 m radius; rest the ball on the ground (visual only,
        # the catch test uses the planar target position).
        marker_z_offset=0.12,
        proximity_planar=True,
        # The chase is the task; falling over is failure, but wandering is not.
        enable_fall_termination=False,
        command_source=BallChaseCommandSourceConfig(
            tar_dist_min=throw_min,
            tar_dist_max=throw_max,
            moving=moving,
            ball_speed_min=ball_speed_min,
            ball_speed_max=ball_speed_max,
            ball_turn_mean_sec=ball_turn_mean_sec,
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
            a deadline: horizon = |bearing| / max_yaw_rate + range / max_speed
            (see _set_deadline; max_yaw_rate defaults to the corpus p99 yaw
            rate, measured alongside this). Not a cap -- nothing
            clamps what the dog attempts, and it may beat the deadline or miss
            it. Using the TOP speed rather than a comfortable one is the point
            (Eric): it makes the deadline the most urgent one physics allows,
            so the instruction is "get there as fast as you can" rather than
            "amble over". None measures it from the corpus (p99 of root speed
            -- 2.44 m/s on the go2, against a 2.60 m/s fastest clip mean and a
            4.10 m/s single-frame spike), which keeps it robot-agnostic.
        min_horizon_sec: Numerical guard only -- it keeps the deadline off
            zero so nothing divides by it. NOT a brake. At 0.05 s it sits
            below the p10 of the nearest lead time the policy trained on, so
            the implied speed holds at top speed the whole way in and the dog
            runs AT the ball instead of easing off. The old 0.5 s floor was a
            deceleration ramp in disguise: inside 1.2 m the implied speed
            became range/0.5 and decayed to nothing (Eric: "no deceleration
            ramp!!!"). The dog does not need to STOP at the ball, it needs to
            REACH it -- the ball is re-thrown the instant it does.
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
    max_yaw_rate: Optional[float] = None
    # How many conditioned slots carry a target. 1 gives the endpoint only,
    # which leaves the PACE undetermined: "be at X in 2 s" is satisfied just
    # as well by ambling there in 2 s as by sprinting and waiting, so nothing
    # prefers a natural gait (Eric: even after turning it "often moves slowly
    # to the target in an unantural way"). More than one pins where the dog
    # should be ALONG the way, and two positions at two times ARE a velocity
    # -- which is how mimic playback never leaves the pace open.
    visible_targets: int = 1
    # Bearing gate on the POSITION target only; the rotation target always
    # faces the ball. Measured, the dog's motion is counterproductive when the
    # ball is behind it:
    #
    #     bearing        efficiency   speed
    #     0-30 deg         +0.94      2.02
    #     30-60 deg        +0.78      1.44
    #     60-120 deg       +0.25      1.10
    #     120-180 deg      -0.64      1.35   <- running AWAY at 1.35 m/s
    #
    # Efficiency is closing rate over speed. Negative means it drives forward
    # and curves round instead of pivoting, so a position commanded BEHIND the
    # dog produces velocity pointing away from it. Holding the position target
    # near the robot while the bearing is large deletes that instruction: only
    # the facing stays live, so the dog pivots, and the position extends toward
    # the ball as it comes round.
    #
    # This is structurally the cos(bearing) scale Eric cut earlier, but that
    # was framed as a speed throttle and tested under an impossible deadline.
    # It is not a throttle: at 120-180 deg the useful speed is already
    # negative. Set position_gate_zero_deg <= position_gate_full_deg to
    # disable it.
    # How much of a turn must be finished BEFORE running, as a function of
    # bearing. A ball behind has to be turned to on the spot; one 45 deg off
    # can be turned into gradually, while running (Eric).
    #
    # The gate scales TWO things together, and that is the point. Gating only
    # the position (measured, earlier) made things WORSE: the deadline stayed
    # at turn+run, so the instruction became "be here, facing the ball, in 3
    # seconds", which invites loitering -- and the dog wandered forward
    # instead. Gating the deadline too makes a large bearing mean "pivot, and
    # you have |bearing|/top_yaw to do it": 0.89 s for 180 deg on the go2.
    #
    # Staging then falls out by itself: the short pivot-only deadline expires,
    # a new one is set from the now-smaller bearing, and it carries more of
    # the run each time.
    #
    # DISABLED by default (zero <= full). Measured three ways against the
    # ungated build, on one checkpoint:
    #
    #                              catches   0-30   30-60  60-120  120-180
    #     ungated                  13.6/min  +0.87  +0.61  +0.11   -0.66
    #     position gated only      13.4/min  +0.96  +0.78  +0.15   -0.88
    #     turn-first (both gated)  13.3/min  +0.91  +0.72  +0.30   -0.82
    #
    # Gating buys efficiency in the middle bands and costs speed everywhere,
    # netting the same catch rate -- and it never produced the pivot it was
    # built for: 120-180 deg stays strongly negative under all three. Even
    # with the position target sitting on the robot AND a 0.89 s pivot-only
    # deadline, an instruction saying nothing but "turn, now", the dog still
    # drives away at ~1.1 m/s. That motion is the policy's own prior and no
    # reshaping of the target reaches it.
    #
    # Ungated is also what Eric judged in the viewer: "pretty good... it only
    # makes mistakes sometimes". Naturalness is the criterion here and the
    # catch-rate metric cannot see it.
    #
    # full: at or below this bearing the run is fully in the budget and the
    #       position target reaches the ball.
    # zero: at or above it neither is -- a pure pivot.
    # Set zero > full to enable.
    position_gate_full_deg: float = 45.0
    position_gate_zero_deg: float = 0.0
    min_horizon_sec: float = 0.05
    max_horizon_sec: float = 15.0

    # ------------------------------------------------------------------
    # Unprivileged sight: what the dog would actually know from a camera
    # ------------------------------------------------------------------
    # The chase as shipped is PRIVILEGED -- the ball's position is read
    # straight out of the simulator, including when it is behind the robot.
    # A go2 cannot see behind itself, and a VLA watching its front camera
    # will not be handed that either, so a demonstrator trained on
    # privileged targets teaches a behaviour the student cannot reproduce:
    # the moment the ball leaves the frame the expert keeps tracking it and
    # the student has nothing to track it with.
    #
    # fov_deg > 0 takes the truth away. The ball is KNOWN only while it
    # falls inside a forward cone of this total width (yaw only -- the
    # camera is modelled as a heading-frame wedge, not a pinhole: pitch and
    # roll do not gate it, and nothing occludes). Outside it the dog holds
    # a BELIEF instead, and the belief is the one a dog actually acts on:
    # if I cannot see it, it is behind me. That belief is a position target
    # behind the robot, so the existing machinery turns it into a pivot,
    # and pivoting sweeps the camera -- the search falls out of the belief
    # rather than being a separate mode with its own controller.
    #
    # 0 disables all of this and restores the privileged chase exactly.
    fov_deg: float = 0.0
    # How far the ball can be recognised at all. 0 = unlimited (the cone is
    # then the only gate).
    sight_range_m: float = 0.0
    # The belief when blind: the ball is this many degrees off the current
    # heading (180 = directly behind, the default and the honest prior) at
    # search_range_m. Latched in WORLD coordinates when the plan is issued,
    # not recomputed per step -- a belief that rotates with the robot is a
    # carrot on a stick and the dog spins without ever arriving at it.
    # Below 180 the sign matters and the search keeps turning the way the
    # ball was last seen leaving, which sweeps one direction consistently.
    search_turn_deg: float = 180.0
    # Near enough that the leg is mostly pivot (the part that actually
    # searches) and the target stays inside the trained displacement range.
    search_range_m: float = 1.5
    # How fast to sweep. A search turned at the dog's top yaw rate is a
    # search run blind: the VLA that has to do the seeing will be sampling at
    # vla_hz, and between two of its frames the whole scene rotates by
    # (yaw rate / vla_hz). At the corpus top yaw of ~3.2 rad/s that is 18 deg
    # a frame, which is a lot of apparent motion to detect a small ball
    # across -- and the answer it returns is already that stale by the time
    # it acts on it. So the SEARCH leg is budgeted at a deliberate rate
    # instead of the top one:
    #
    #     search yaw = search_deg_per_frame * vla_hz
    #
    # 10 deg at 10 Hz = 100 deg/s, which holds any bearing inside a 120 deg
    # cone for 1.2 s -- 12 frames to notice a ball in, against 6 at top yaw,
    # and consecutive frames overlap by more than 90% of the cone.
    #
    # The budget is the only thing that sets pace here (there is no rate
    # limiter anywhere): a longer deadline for the same turn IS a slower
    # turn, which is exactly how the slow-turn bug worked in reverse. Clamped
    # to the top yaw rate, since budgeting faster than the robot can pivot
    # just makes the deadline unmeetable. Only the search is slowed -- the
    # moment the ball is acquired the chase re-plans at full speed.
    vla_hz: float = 10.0
    search_deg_per_frame: float = 10.0


class MaskedMimicGoalControl(MaskedMimicSteeringControl):
    """Base-link targets that say "be at the ball, facing it, by then".

    As shipped the dog is told three things and no more:

      * WHERE  -- the ball, ungated, at full distance.
      * FACING -- straight at the ball, every slot, no rate limit.
      * WHEN   -- |bearing|/top_yaw + range/top_speed, so the turn is paid
                  for on top of the run.

    Unprivileged (fov_deg > 0) adds a fourth: when the ball is not in view,
    WHERE and FACING come from the belief that it is behind, and WHEN is
    budgeted at a deliberate sweep rate rather than the top one, so whatever
    is doing the looking gets enough frames to look in.

    It is never told to finish turning BEFORE running. That sequencing is what
    the bearing gates would add, and they are off by default because they were
    measured and did not deliver it.
    """

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
        self._visible = min(
            max(int(config.visible_targets), 1), config.num_masked_future_steps
        )
        self.masked_mimic_target_poses_masks[:] = False
        self.masked_mimic_target_poses_masks[:, -self._visible:] = True
        bodies = self.masked_mimic_target_bodies_masks.view(
            env.num_envs, config.num_masked_future_steps, -1
        )
        bodies[:, :-self._visible, :] = False

        # Absolute deadline per env, in elapsed seconds. Set once per throw
        # and then counted DOWN -- see _lead_times.
        self._deadline = torch.zeros(env.num_envs, device=env.device)
        # Which throw the deadline belongs to (-1: none yet). Keyed to the
        # throw, not the ball's position, so a moving ball does not restart
        # the deadline every step.
        self._deadline_plan = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        # Where the ball was predicted to be at the deadline, as solved when
        # the plan was issued. Kept for readouts; the live target is
        # _target_xy(), which re-evaluates the prediction from the current
        # ball measurement every step (identical in the sim, better on a
        # robot whose estimate improves as it closes).
        self._intercept_xy = torch.zeros(env.num_envs, 2, device=env.device)

        # --- belief state (unprivileged mode; inert when fov_deg == 0) ---
        # Is the ball in view RIGHT NOW. Privileged mode pins this True.
        self._ball_seen = torch.full(
            (env.num_envs,), not self._unprivileged(), dtype=torch.bool,
            device=env.device,
        )
        # The latched "it must be behind me" point, used while blind.
        self._search_xy = torch.zeros(env.num_envs, 2, device=env.device)
        # Which way to sweep: +1 left, -1 right. Set to the side the ball
        # was last seen leaving on.
        self._search_sign = torch.ones(env.num_envs, device=env.device)
        # Bearing at the last honest sighting -- which side it was on when it
        # was last actually visible. Not the bearing at the moment of loss:
        # a catch re-throws the ball before the next look, so that reading
        # would be of the NEW ball and would leak its position into the
        # search direction.
        self._last_bearing = torch.zeros(env.num_envs, device=env.device)
        # Bumped whenever the BELIEF changes (acquired / lost). The deadline
        # is keyed to this as well as to the ball's plan id, so acquiring the
        # ball re-plans immediately instead of finishing the search leg.
        self._epoch = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self._deadline_epoch = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self._sight_steps = 0
        self._sight_seen = 0
        self._blind_run = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self._blind_spans = []

    def reset(self, env_ids: Tensor) -> None:
        """Forget the throw's deadline so the first real step re-issues it.

        The inherited reset calls _lead_times() while progress_buf still holds
        its PRE-reset value (env.reset zeroes it only after the control
        components reset). After the R key that value is the 100000000000-step
        sentinel user_reset() plants: a clock of ~2e9 s, where float32 spacing
        is 128 s and the whole budget rounds away. Left alone, the deadline is
        stamped two billion seconds out, the ball re-throw is stamped against
        the same clock, nothing ever moves the ball again, and the dog ambles
        at clip-playback pace for the rest of the run. Clearing the anchors
        here forces a clean restart on the next step, when the clock is zero.
        """
        # Look first: otherwise the inherited reset plans this env's first
        # deadline against a stale belief and the dog opens with a search leg
        # while the ball sits in front of it.
        self._update_belief()
        super().reset(env_ids)
        self._deadline[env_ids] = 0.0
        self._deadline_plan[env_ids] = -1
        self._deadline_epoch[env_ids] = -1

    def _top_speed(self) -> float:
        """The robot's top speed, measured from the corpus unless configured."""
        if self.config.max_speed is not None:
            return max(float(self.config.max_speed), 1e-6)
        if self._corpus_max_speed is None:
            self._measure_height_fit()
        return max(self._corpus_max_speed or 1.0, 1e-6)

    def _fractions(self) -> Tensor:
        """Where each visible slot sits, as a fraction of the remaining
        distance AND of the remaining time -- so the pace they imply is
        constant and reachable by construction."""
        n = self.config.num_masked_future_steps
        f = torch.ones(n, device=self.env.device)
        f[-self._visible:] = (
            torch.arange(1, self._visible + 1, device=self.env.device).float()
            / self._visible
        )
        return f

    def _bearing_gate(self, bearing: Tensor) -> Tensor:
        """1 when the dog is pointed at the ball, 0 when it is behind.

        bearing in RADIANS. Scales the position target and the run term of the
        deadline together, so the two never disagree about whether the dog is
        pivoting or running.

        Returns all ones when disabled, which is the default -- see
        position_gate_full_deg for the measurements that turned it off.
        """
        full = self.config.position_gate_full_deg
        zero = self.config.position_gate_zero_deg
        if zero <= full:
            return torch.ones_like(bearing)
        deg = torch.rad2deg(bearing.abs())
        return ((zero - deg) / (zero - full)).clamp(0.0, 1.0)

    def _search_yaw(self) -> float:
        """Yaw rate to budget a SEARCH leg at, in rad/s.

        Slow enough that the policy doing the looking gets a usable number of
        frames per cone-width of sweep -- see search_deg_per_frame. Never
        faster than the robot can actually pivot.
        """
        rate = math.radians(
            max(self.config.search_deg_per_frame, 1e-3)
            * max(self.config.vla_hz, 1e-3)
        )
        return min(rate, self._top_yaw())

    def _top_yaw(self) -> float:
        """Top yaw rate, measured from the corpus unless configured."""
        if self.config.max_yaw_rate is not None:
            return max(float(self.config.max_yaw_rate), 1e-6)
        if self._corpus_max_yaw is None:
            self._measure_height_fit()
        return max(self._corpus_max_yaw or 1.0, 1e-6)

    def _goal_xy(self) -> Tensor:
        """The ball's CURRENT planar position (catch test, readouts)."""
        target = self.env.control_manager.components[self.config.target_component]
        return target._tar_pos[:, :2]

    def _ball_source(self):
        target = self.env.control_manager.components[self.config.target_component]
        return target.command_source

    def _ball_vel(self) -> Tensor:
        """Planar ball velocity; zeros for a static ball or a plain source."""
        vel = getattr(self._ball_source(), "_tar_vel", None)
        if vel is None:
            return torch.zeros_like(self._goal_xy())
        return vel

    def _plan_id(self) -> Tensor:
        """Bumps on every throw and every ball direction change."""
        pid = getattr(self._ball_source(), "plan_id", None)
        if pid is None:
            # Plain sources never re-throw on their own; treat as one plan.
            return torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)
        return pid

    # ------------------------------------------------------------------
    # Sight and belief
    # ------------------------------------------------------------------

    def _unprivileged(self) -> bool:
        """True when the dog has to earn the ball's position by looking."""
        return self.config.fov_deg > 0.0

    def _bearing_to(self, xy: Tensor) -> Tensor:
        """Signed bearing to a planar point, in the robot's heading frame."""
        root_state = self.env.simulator.get_root_state()
        delta = xy - root_state.root_pos[:, :2]
        heading = rotations.calc_heading(root_state.root_rot, True)
        to = torch.atan2(delta[:, 1], delta[:, 0])
        return torch.atan2(torch.sin(to - heading), torch.cos(to - heading))

    def _in_view(self) -> Tensor:
        """Is the ball inside the forward cone (and close enough to see)."""
        ball = self._goal_xy()
        seen = self._bearing_to(ball).abs() <= math.radians(self.config.fov_deg) * 0.5
        if self.config.sight_range_m > 0:
            root_state = self.env.simulator.get_root_state()
            rng = torch.linalg.norm(ball - root_state.root_pos[:, :2], dim=-1)
            seen = seen & (rng <= self.config.sight_range_m)
        return seen

    def _update_belief(self) -> None:
        """Look, and note the moments the answer changes.

        Only the CHANGES matter downstream: acquiring or losing the ball
        bumps the epoch, which is what makes the deadline re-plan. While the
        answer holds steady nothing re-plans, exactly as with a moving ball
        whose velocity is known -- see _lead_times.
        """
        if not self._unprivileged():
            return
        seen = self._in_view()
        lost = self._ball_seen & ~seen
        if bool(lost.any()):
            # Sweep after it: it left the frame on one side, so keep turning
            # that way rather than guessing. (Only bites below 180 deg.)
            side = torch.where(
                self._last_bearing >= 0,
                torch.ones_like(self._search_sign),
                -torch.ones_like(self._search_sign),
            )
            self._search_sign = torch.where(lost, side, self._search_sign)
        changed = lost | (seen & ~self._ball_seen)
        if bool(changed.any()):
            self._epoch[changed] += 1
        self._ball_seen = seen
        if bool(seen.any()):
            self._last_bearing = torch.where(
                seen, self._bearing_to(self._goal_xy()), self._last_bearing
            )

    def _latch_search(self, env_ids: Tensor) -> None:
        """Believe the ball is behind, and nail that belief to the ground.

        Placing it in world coordinates is the whole trick: the dog turns
        toward a point that stays put, so the turn ends. Re-deriving "behind
        me" every step would move the point with the robot and the dog would
        rotate forever at a fixed bearing error.
        """
        root_state = self.env.simulator.get_root_state()
        heading = rotations.calc_heading(root_state.root_rot, True)
        ang = heading + self._search_sign * math.radians(self.config.search_turn_deg)
        point = root_state.root_pos[:, :2] + self.config.search_range_m * torch.stack(
            [torch.cos(ang), torch.sin(ang)], dim=-1
        )
        self._search_xy[env_ids] = point[env_ids]

    def _belief_xy(self) -> Tensor:
        """Where the dog thinks the ball is. The truth when privileged."""
        if not self._unprivileged():
            return self._goal_xy()
        return torch.where(
            self._ball_seen.unsqueeze(-1), self._goal_xy(), self._search_xy
        )

    def _belief_vel(self) -> Tensor:
        """Believed ball velocity -- zero for a belief, which is not moving."""
        if not self._unprivileged():
            return self._ball_vel()
        return torch.where(
            self._ball_seen.unsqueeze(-1),
            self._ball_vel(),
            torch.zeros_like(self._search_xy),
        )

    def _target_xy(self) -> Tensor:
        """The point to run at: where the ball will be when the deadline arrives.

        Re-evaluated EVERY step from the current ball state -- position and
        velocity as measured now -- against the fixed deadline. With the
        velocity known exactly (the sim) this is the same world point every
        step, so nothing visibly re-plans. The point of evaluating it live is
        the real robot: a perception estimate of the ball gets better as the
        dog closes, and this folds each better measurement straight into the
        target without touching the deadline. Only the DEADLINE is per plan;
        if a better estimate reveals it cannot be met, it expires and
        _lead_times re-plans from where the dog actually is.
        (_ball_vel() is where an estimated velocity would plug in.)

        Unprivileged: this is the BELIEF, not the ball -- the sighting while
        it is in view, the latched point behind the robot while it is not.
        """
        have = (self._deadline_plan >= 0).unsqueeze(-1)
        remaining = (self._deadline - self._now()).clamp(
            self.config.min_horizon_sec, self.config.max_horizon_sec
        )
        ball = self._belief_xy()
        live = ball + self._belief_vel() * remaining.unsqueeze(-1)
        return torch.where(have, live, ball)

    def _intercept_run_time(self, delta: Tensor, vel: Tensor, tau: Tensor) -> Tensor:
        """Running time s (after a turn of tau) to meet a ball moving at vel.

        Solves |delta + vel (tau + s)| = top_speed s + stop_distance for the
        smallest s >= 0. With vel = 0 this is exactly (range - stop) /
        top_speed, the static-ball budget. If the ball outruns the dog
        (no positive root) the run term falls back to the horizon ceiling:
        the dog is still sent after it, just with the longest deadline.
        """
        vmax = self._top_speed()
        stop = self.config.stop_distance
        q = delta + vel * tau.unsqueeze(-1)
        a = (vel * vel).sum(-1) - vmax * vmax
        b = 2.0 * ((q * vel).sum(-1) - vmax * stop)
        c = (q * q).sum(-1) - stop * stop
        fallback = (self.config.max_horizon_sec - tau).clamp_min(0.0)
        inf = torch.full_like(a, float("inf"))

        # Degenerate (|vel| == top_speed): linear.
        linear = torch.where(b.abs() > 1e-9, -c / torch.where(b.abs() > 1e-9, b, torch.ones_like(b)), inf)
        disc = b * b - 4.0 * a * c
        safe_a = torch.where(a.abs() > 1e-9, a, torch.ones_like(a))
        root = disc.clamp_min(0.0).sqrt()
        r1 = (-b - root) / (2.0 * safe_a)
        r2 = (-b + root) / (2.0 * safe_a)
        pos1 = torch.where(r1 >= 0, r1, inf)
        pos2 = torch.where(r2 >= 0, r2, inf)
        quad = torch.where(disc >= 0, torch.minimum(pos1, pos2), inf)
        s = torch.where(a.abs() > 1e-9, quad, linear)
        s = torch.where(torch.isfinite(s), s, fallback)
        return s.clamp(0.0, self.config.max_horizon_sec)

    def _now(self) -> Tensor:
        """Elapsed seconds per env. Monotonic: this task never resets."""
        return self.env.progress_buf.float() * self.env.dt

    def _lead_times(self) -> Tensor:
        """Time remaining until the deadline -- counting DOWN.

        The deadline is ABSOLUTE, set once when the ball is thrown, exactly as
        MaskedMimic target times are absolute moments in a clip. What the
        policy sees is therefore 2.00 s, 1.98 s, 1.96 s ... and that countdown
        is the urgency signal it was distilled against.

        Recomputing range / top_speed every step -- which is what this did
        before -- makes the deadline recede perpetually: always "get there in
        2 seconds from now", never arriving, so the countdown never happens
        (Eric: "you are always saying: be at the target immediately? That
        isn't how mimic is supposed to work"). It is a standing demand for
        maximum urgency re-issued fifty times a second, not a deadline.

        A fresh throw sets a new deadline. So does letting one expire, which
        is the miss case -- the analogue of MaskedMimic shifting its slot on
        to the next target once the current one is passed.
        """
        now = self._now()
        expired = (self._deadline - now) <= self.config.min_horizon_sec
        # Restart per THROW, never per ball movement: a moving ball changes
        # position every step, and a deadline re-issued every step is the
        # standing maximum-urgency demand the docstring above warns of.
        # A throw or course change is only news if the dog can SEE it. When
        # blind, the ball's plan id is privileged information -- keying off it
        # would let the dog re-plan on a re-throw it has no way of noticing.
        plan_changed = self._plan_id() != self._deadline_plan
        if self._unprivileged():
            plan_changed = plan_changed & self._ball_seen
        restart = expired | plan_changed | (self._epoch != self._deadline_epoch)
        if bool(restart.any()):
            self._set_deadline(restart)
        # Both bounds, not just the floor: max_horizon_sec caps the BUDGET at
        # issue time, but a stale deadline (clock rewound under it) would
        # otherwise pass straight through here as an absurd lead time.
        remaining = (self._deadline - now).clamp(
            self.config.min_horizon_sec, self.config.max_horizon_sec
        )
        return remaining.unsqueeze(-1) * self._fractions().unsqueeze(0)

    def _set_deadline(self, env_ids: Tensor) -> None:
        """Budget range / top_speed from now: the most urgent time the robot
        could actually meet, fixed for the rest of this throw."""
        mask = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        mask[env_ids] = True
        if self._unprivileged():
            # Planning with nothing in view: (re-)latch the belief behind the
            # robot. Doing it HERE and only here means every search leg --
            # the first one, and each one after the previous expires -- is
            # measured from the heading the dog has now, so consecutive legs
            # keep the sweep going instead of re-aiming at the same point.
            blind = mask & ~self._ball_seen
            if bool(blind.any()):
                self._latch_search(blind)
        root_state = self.env.simulator.get_root_state()
        root_pos = root_state.root_pos
        ball = self._belief_xy()
        delta = ball - root_pos[:, :2]
        aim = (
            torch.linalg.norm(delta, dim=-1) - self.config.stop_distance
        ).clamp_min(0.0)

        # Budget the TURN as well as the run. range/top_speed alone assumes a
        # straight sprint from a standing start already facing the ball, so a
        # ball behind is demanded in 70% of the time it genuinely needs -- and
        # the whole shortfall lands exactly when the dog should be turning. A
        # 180 deg turn takes 0.89 s at the go2's measured 3.51 rad/s; the old
        # budget allowed 0.00 s for it (Eric: with a ball behind it "choses to
        # turn slowly and sometimes not in a natural way").
        heading = rotations.calc_heading(root_state.root_rot, True)
        goal_heading = torch.atan2(delta[:, 1], delta[:, 0])
        bearing = torch.atan2(
            torch.sin(goal_heading - heading), torch.cos(goal_heading - heading)
        ).abs()
        # The turn is ALWAYS paid for, gate or no gate: a ball behind gets
        # |bearing|/top_yaw more than the same distance ahead. The gate, when
        # enabled, additionally removes the run term so the deadline becomes
        # pivot-only; off by default, so the budget here is turn + run.
        # Moving ball: the run term is an INTERCEPT, not the current range.
        # Solve for the running time to meet the ball, then refine the turn
        # against the intercept point itself (the dog turns toward where it is
        # going, not toward where the ball is now) and solve once more.
        vel = self._belief_vel()
        # A search leg is swept at the deliberate rate; a leg planned against
        # a real sighting is turned as fast as the dog can.
        yaw_rate = torch.full_like(bearing, self._top_yaw())
        if self._unprivileged():
            yaw_rate = torch.where(
                self._ball_seen, yaw_rate, torch.full_like(bearing, self._search_yaw())
            )
        tau = bearing / yaw_rate
        run = self._intercept_run_time(delta, vel, tau)
        # Fixed point: the turn depends on where the intercept is, and the
        # intercept depends on how long the turn takes. A few vectorized
        # iterations converge it; for a static ball it is exact at once.
        for _ in range(8):
            point = ball + vel * (tau + run).unsqueeze(-1)
            d2 = point - root_pos[:, :2]
            heading2 = torch.atan2(d2[:, 1], d2[:, 0])
            bearing = torch.atan2(
                torch.sin(heading2 - heading), torch.cos(heading2 - heading)
            ).abs()
            tau = bearing / yaw_rate
            run = self._intercept_run_time(delta, vel, tau)
        # (aim / top_speed is what `run` reduces to for a static ball.)
        budget = (tau + self._bearing_gate(bearing) * run).clamp(
            self.config.min_horizon_sec, self.config.max_horizon_sec
        )
        now = self._now()
        self._deadline[env_ids] = now[env_ids] + budget[env_ids]
        # The prediction at issue time, from the CLAMPED budget -- if the
        # ceiling bit, this is still where the ball will be when the deadline
        # arrives. _target_xy() recomputes the same quantity live each step.
        self._intercept_xy[env_ids] = (ball + vel * budget.unsqueeze(-1))[env_ids]
        self._deadline_plan[env_ids] = self._plan_id()[env_ids]
        self._deadline_epoch[env_ids] = self._epoch[env_ids]

    def step(self) -> None:
        """Look first, then plan. The ball component has already moved."""
        self._update_belief()
        super().step()
        self._report_sight()

    def _report_sight(self) -> None:
        """How much of the chase is spent blind, and how long a search takes."""
        every = self.config.report_every_steps
        if every <= 0 or not self._unprivileged():
            return
        self._sight_steps += 1
        self._sight_seen += int(self._ball_seen.sum())
        self._blind_run[~self._ball_seen] += 1
        done = self._ball_seen & (self._blind_run > 0)
        if bool(done.any()):
            self._blind_spans.extend(self._blind_run[done].tolist())
            self._blind_run[done] = 0
        if self._sight_steps < every:
            return
        steps = max(self._sight_steps * self.env.num_envs, 1)
        spans = self._blind_spans
        blind_s = (
            sum(spans) / len(spans) * self.env.dt if spans else float("nan")
        )
        sweep = math.degrees(self._search_yaw())
        dwell = self.config.fov_deg / max(sweep, 1e-6)
        print(
            f"[chase-vision] ball in view {100.0 * self._sight_seen / steps:.0f}% "
            f"of steps, {len(spans)} reacquisitions, mean {blind_s:.1f} s to "
            f"find it (fov {self.config.fov_deg:.0f} deg, search "
            f"{self.config.search_turn_deg:.0f} deg at "
            f"{self.config.search_range_m:.1f} m, sweep {sweep:.0f} deg/s = "
            f"{dwell:.1f} s / {dwell * self.config.vla_hz:.0f} frames in view "
            f"at {self.config.vla_hz:.0f} Hz)",
            flush=True,
        )
        self._sight_steps = 0
        self._sight_seen = 0
        self._blind_spans = []

    def _rollout(
        self, root_pos: Tensor, root_rot: Tensor, lead: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # The intercept point, not the ball: for a static ball they coincide.
        delta = self._target_xy() - root_pos[:, :2]
        rng = torch.linalg.norm(delta, dim=-1, keepdim=True)
        direction = delta / rng.clamp_min(1e-6)

        aim = (rng - self.config.stop_distance).clamp_min(0.0)
        goal = root_pos[:, :2] + direction * aim
        heading = torch.atan2(direction[:, 1], direction[:, 0])

        steps = self.config.num_masked_future_steps
        # Optional bearing gate, OFF by default: it would hold the position
        # target near the robot while the ball is behind, leaving "turn" as
        # the only live instruction. Measured, it never produced that pivot
        # (see position_gate_full_deg). Disabled, _bearing_gate returns ones
        # and this is a no-op, so the target below is the plain one: the ball,
        # faced, by the deadline.
        cur = rotations.calc_heading(root_rot, True)
        bearing = torch.atan2(
            torch.sin(heading - cur), torch.cos(heading - cur)
        )
        gate = self._bearing_gate(bearing)
        frac = self._fractions().view(1, steps, 1) * gate.view(-1, 1, 1)
        # Each visible slot sits the same fraction of the way along as it does
        # through the remaining time, so together they state a steady PACE to
        # the ball rather than only its endpoint. With the default
        # visible_targets=1 there is a single slot at fraction 1.0, i.e. the
        # ball itself -- the fractions only bite when more are shown.
        along = root_pos[:, :2].unsqueeze(1) + (
            goal - root_pos[:, :2]
        ).unsqueeze(1) * frac
        return along, heading.unsqueeze(-1).expand(-1, steps)

    def _command(self) -> Tensor:
        """The velocity the goal implies -- for the inherited readout only.

        Never drives anything. Reported against the nearest lead time, so a
        distant ball shows a large implied command; that is honest, it IS what
        the target asks for.
        """
        root_state = self.env.simulator.get_root_state()
        delta = self._belief_xy() - root_state.root_pos[:, :2]
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
