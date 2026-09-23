# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Velocity-command conditioning for a trained MaskedMimic policy.

MaskedMimic is conditioned on sparse future body poses. This component swaps
the motion-library lookup for a kinematic roll-out of the steering command:
the base link is the ONLY conditioned body, and its target at t+dt_k is where
the robot would be if it held the commanded (forward, yaw-rate, lateral)
velocity for dt_k seconds starting from where it is RIGHT NOW.

The roll-out is re-anchored on the live root pose every step, so the targets
carry no accumulated tracking error -- they encode a velocity, not a path.
Put the target 0.2 s in front at 1 m/s and the policy walks at 1 m/s.

The command itself (random walk, ramping, gamepad teleop, the green direction
arrow and the spinning turn dial) comes from the ordinary steering component
in this package, which must be registered ALONGSIDE this one:

    control_components = {
        "steering_cmd": SteeringCommandControlConfig(...),
        "masked_mimic": MaskedMimicSteeringControlConfig(...),
    }

Nothing here is trained: it is an evaluation harness for an existing
MaskedMimic checkpoint, so the observation and model configs must stay
byte-identical to the ones it was distilled with.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext, MaskedMimicContext
from protomotions.envs.control.masked_mimic_control import (
    MaskedMimicControl,
    MaskedMimicControlConfig,
)
from protomotions.envs.control.mimic_control import MimicControl
from protomotions.utils import rotations
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


# Below this yaw rate the arc roll-out is replaced by its straight-line limit
# (the closed form divides by omega).
_STRAIGHT_EPS = 1e-4


@dataclass
class MaskedMimicSteeringControlConfig(MaskedMimicControlConfig):
    """Configuration for velocity-command MaskedMimic conditioning.

    Attributes:
        command_component: Key of the sibling SteeringCommandControl in the
            env's control_components dict. Its ramped command drives the
            roll-out.
        horizon_sec: Time of the FARTHEST conditioned target. The
            num_masked_future_steps targets are spread evenly over
            (0, horizon_sec], so the default 1.0 s with 5 steps gives
            0.2/0.4/0.6/0.8/1.0 s.
        height_mode: Where the commanded base-link height comes from.
            "current" keeps the target at the height the robot is ALREADY at,
            so the z channel carries no instruction and the command is purely
            planar -- the closest thing to masking height out, which the mask
            itself cannot express (one bit covers the whole translation
            vector, so dropping z would drop x and y with it). Height
            regulation is then left to the policy's own prior. Still clamped
            to the corpus band, so a robot that sinks is pulled back up
            instead of dragging its own target into the floor.
            "corpus" commands the height the corpus carries at the commanded
            speed (a linear fit, see _measure_height_fit).
            "fixed" commands target_root_height.
        target_root_height: The height used by height_mode="fixed".
        height_fit_samples: How many random corpus poses the fit and the
            clamp band are built from.
        condition_rotation: Condition the base link's orientation as well as
            its position. Off = position only, letting the policy pick its own
            facing. Measured worse on the chase (11.2-14.2 catches/min against
            15.8-23.2), so the facing command earns its place.
        preserve_tilt: Build the rotation target by turning the robot's
            CURRENT orientation about world up to the desired heading, so the
            target keeps the roll and pitch the body actually has. Off builds
            a pure-yaw quaternion, which also commands perfectly level -- an
            attitude no real dog holds while turning, and nothing like what
            MaskedMimic is conditioned on in training, where the rotation
            target is the CLIP's own rigid_body_rot with its real tilt
            (corpus mean 7.8 deg; only 38.7% of frames within 5 deg of level).
        report_every_steps: Print a commanded-vs-achieved velocity summary
            every N env steps. 0 disables it.
    """

    _target_: str = (
        "protomotions.envs.steering.masked_mimic_command.MaskedMimicSteeringControl"
    )

    command_component: str = "steering_cmd"
    horizon_sec: float = 1.0
    height_mode: str = "current"
    target_root_height: Optional[float] = None
    height_fit_samples: int = 8192
    condition_rotation: bool = True
    preserve_tilt: bool = True
    report_every_steps: int = 0


class MaskedMimicSteeringControl(MaskedMimicControl):
    """MaskedMimic conditioning driven by a velocity command, not a clip.

    Overrides the three motion-library behaviours of the parent:
      * target times are a fixed ladder instead of beta-sampled clip times,
      * body masks are fixed to "base link only" instead of resampled,
      * target poses are integrated from the command instead of looked up.

    ctx.mimic is still populated by MimicControl (the motion library is the
    reset/RSI source), but nothing the MaskedMimic prior reads comes from it.
    """

    config: MaskedMimicSteeringControlConfig

    def __init__(self, config: MaskedMimicSteeringControlConfig, env: "BaseEnv"):
        super().__init__(config, env)

        num_envs, device = self.env.num_envs, self.env.device
        num_steps = self.config.num_masked_future_steps

        # Evenly spaced lead times over (0, horizon].
        self._offsets = torch.linspace(
            self.config.horizon_sec / num_steps,
            self.config.horizon_sec,
            num_steps,
            device=device,
            dtype=torch.float,
        )

        # The conditioned body. build_sparse_target_poses() always treats body
        # 0 as the root frame, so anchor_body_name is expected to be body 0
        # (true for every quadruped config here); the mask index is its slot
        # inside trackable_bodies_subset.
        body_names = self.env.robot_config.kinematic_info.body_names
        self._root_body_id = body_names.index(self.env.robot_config.anchor_body_name)
        cond_ids = self.conditionable_body_ids.tolist()
        assert self._root_body_id in cond_ids, (
            f"anchor body {self.env.robot_config.anchor_body_name!r} is not in "
            "trackable_bodies_subset -- MaskedMimic cannot be conditioned on it"
        )
        self._root_cond_idx = cond_ids.index(self._root_body_id)
        self._num_bodies = len(body_names)

        # Height fit, measured lazily from the corpus on the first reset.
        # robot_config.default_root_height is NOT usable here: it is the
        # KINEMATIC standing height of the default joint pose (0.2868 m on the
        # go2), while the repacked corpus carries the body 6-9 cm higher and
        # rises with speed (0.345 m at rest, 0.376 m above 2 m/s). Commanding
        # the kinematic height asks the policy to crouch below anything it was
        # ever shown, at every speed, which flattens the gait it will offer
        # (Eric, 2026-09-19: "the commanded speed immediately slows it down to
        # a walk").
        self._height_a = None
        self._height_b = 0.0
        self._height_lo = 0.0
        self._height_hi = 0.0
        self._corpus_max_speed = None
        self._corpus_max_yaw = None

        # Every conditioned pose is visible, and only the base link is
        # conditioned within it. Both are constant, so build them once.
        self.masked_mimic_target_poses_masks[:] = True
        fixed = torch.zeros(
            num_envs,
            num_steps,
            self.num_conditionable_bodies,
            2,
            dtype=torch.bool,
            device=device,
        )
        fixed[:, :, self._root_cond_idx, 0] = True  # translation
        if self.config.condition_rotation:
            fixed[:, :, self._root_cond_idx, 1] = True  # rotation
        self.masked_mimic_target_bodies_masks[:] = fixed.view(num_envs, -1)

        # Identity quaternion (xyzw) used to fill the masked-out bodies. Zeros
        # would be multiplied out by the mask, but only after passing through
        # quat_rotate, which turns a zero quaternion into NaN.
        self._identity_quat = torch.zeros(4, device=device, dtype=torch.float)
        self._identity_quat[3] = 1.0

        self._initialized = False

        # Commanded-vs-achieved accumulators for the periodic readout.
        self._report_steps = 0
        self._report_abs_err = torch.zeros(3, device=device, dtype=torch.float)
        self._report_abs_cmd = torch.zeros(3, device=device, dtype=torch.float)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _measure_height_fit(self):
        """Fit commanded root height against speed, from the motion library.

        Least squares on random corpus poses, clamped to the observed 5th-95th
        percentile band so an extrapolated command can never ask for a height
        the corpus does not contain.
        """
        if self.config.height_mode == "fixed":
            fixed = self.config.target_root_height
            assert fixed is not None, 'height_mode="fixed" needs target_root_height'
            self._height_a = float(fixed)
            self._height_b = 0.0
            self._height_lo = self._height_hi = float(fixed)
            self._corpus_max_speed = None
            self._corpus_max_yaw = None
            return

        num = self.config.height_fit_samples
        motion_lib = self.env.motion_lib
        motion_ids = torch.randint(
            0, motion_lib.num_motions(), (num,), device=self.env.device
        )
        lengths = motion_lib.get_motion_length(motion_ids)
        times = torch.rand(num, device=self.env.device) * lengths
        state = motion_lib.get_motion_state(motion_ids, times)
        z = state.rigid_body_pos[:, 0, 2]
        speed = state.rigid_body_vel[:, 0, :2].norm(dim=-1)

        var = speed.var()
        slope = (
            ((speed - speed.mean()) * (z - z.mean())).mean() / var
            if float(var) > 1e-8
            else torch.zeros((), device=z.device)
        )
        self._height_b = float(slope)
        self._height_a = float(z.mean() - slope * speed.mean())
        self._height_lo = float(z.quantile(0.05))
        self._height_hi = float(z.quantile(0.95))
        # Top sustained speed this corpus demonstrates, for tasks that need to
        # turn a distance into a deadline. p99 rather than the outright max:
        # the max is a single-frame spike (4.10 m/s on the go2) while p99
        # (2.44) sits at the fastest sustained gait (fastest clip mean 2.60).
        self._corpus_max_speed = float(speed.quantile(0.99))
        # Same statistic for the turn, for the same reason: the deadline
        # budgets |bearing| / top_yaw, and the corpus max is a spike (6.93
        # rad/s on the go2) while p99 (3.51) is the fastest pivot it sustains.
        # Left unset, ball_chase falls back to 1.0 rad/s and budgets a 180 deg
        # turn at 3.1 s -- the dog then turns slowly, exactly on schedule.
        yaw_rate = state.rigid_body_ang_vel[:, 0, 2].abs()
        self._corpus_max_yaw = float(yaw_rate.quantile(0.99))
        print(
            f"[mm-steering] commanded root height from corpus: "
            f"{self._height_a:.4f} + {self._height_b:.4f}*speed, clamped to "
            f"[{self._height_lo:.4f}, {self._height_hi:.4f}] "
            f"(robot default_root_height is "
            f"{self.env.robot_config.default_root_height:.4f}); "
            f"corpus p99 speed {self._corpus_max_speed:.2f} m/s, "
            f"p99 yaw rate {self._corpus_max_yaw:.2f} rad/s",
            flush=True,
        )

    def _target_heights(self, speed_cmd: Tensor, current: Tensor) -> Tensor:
        """Commanded base-link height above terrain, per env.

        Fits on first use: the env builds observations once during setup,
        before any reset, so populate_context can land here first.

        Args:
            speed_cmd: Commanded planar speed [envs, 1].
            current: The robot's CURRENT height above terrain [envs, 1].
        """
        if self._height_a is None:
            self._measure_height_fit()
        if self.config.height_mode == "current":
            # z carries no instruction: the target sits exactly where the
            # robot already is, so rel_pos.z is 0 and the command is purely
            # planar. The mask itself cannot express this -- one bit covers
            # the whole translation vector, so dropping z drops x and y too.
            height = current
        else:
            height = self._height_a + self._height_b * speed_cmd
        # Clamped even in "current" mode, so a robot that sinks is pulled back
        # up rather than dragging its own target into the floor.
        return height.clamp(self._height_lo, self._height_hi)

    def reset(self, env_ids: Tensor):
        """Reset without resampling times or masks -- both are fixed here."""
        MimicControl.reset(self, env_ids)
        if self._height_a is None:
            self._measure_height_fit()
        if len(env_ids) == 0:
            return
        self.target_times[env_ids] = (
            self.env.motion_manager.motion_times[env_ids].unsqueeze(-1)
            + self._lead_times()[env_ids]
        )
        self._initialized = True

    def step(self):
        """Advance the fixed ladder and keep the motion clock in range.

        The clip is never the objective here, but MimicControl.populate_context
        still queries the motion library at motion_times, and the motion
        manager keeps advancing them past the end of the clip once
        check_resets_and_terminations() stops asking for resets. Wrapping keeps
        every lookup in bounds without teleporting the robot.
        """
        MimicControl.step(self)
        if not self._initialized:
            return

        motion_manager = self.env.motion_manager
        lengths = self.env.motion_lib.motion_lengths[motion_manager.motion_ids]
        motion_manager.motion_times.remainder_(lengths.clamp_min(self.env.dt))
        self.target_times[:] = (
            motion_manager.motion_times.unsqueeze(-1) + self._lead_times()
        )

        if self.config.report_every_steps > 0:
            self._accumulate_tracking()

    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        """Never reset: the command is open-ended, the clip is not the task."""
        zeros = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        return zeros, zeros.clone()

    # ------------------------------------------------------------------
    # Tracking readout
    # ------------------------------------------------------------------

    def _accumulate_tracking(self):
        """Average |command - achieved| per channel, in the heading frame.

        The command is a velocity, so this is the only thing that says whether
        the target roll-out is being followed: the marker spheres show where
        the robot was TOLD to be, not whether it got there.
        """
        root_state = self.env.simulator.get_root_state()
        heading_inv = rotations.calc_heading_quat_inv(root_state.root_rot, True)
        local_vel = rotations.quat_rotate(heading_inv, root_state.root_vel, True)
        achieved = torch.stack(
            [local_vel[:, 0], root_state.root_ang_vel[:, 2], local_vel[:, 1]],
            dim=-1,
        )
        cmd = self._command()
        self._report_abs_err += (achieved - cmd).abs().mean(dim=0)
        self._report_abs_cmd += cmd.abs().mean(dim=0)
        self._report_steps += 1

        if self._report_steps < self.config.report_every_steps:
            return
        err = (self._report_abs_err / self._report_steps).tolist()
        mag = (self._report_abs_cmd / self._report_steps).tolist()
        print(
            "[mm-steering] mean |cmd-achieved| over "
            f"{self._report_steps} steps x {self.env.num_envs} envs -- "
            f"forward {err[0]:.3f} m/s (|cmd| {mag[0]:.3f}), "
            f"yaw {err[1]:.3f} rad/s (|cmd| {mag[1]:.3f}), "
            f"lateral {err[2]:.3f} m/s (|cmd| {mag[2]:.3f})",
            flush=True,
        )
        self._report_steps = 0
        self._report_abs_err.zero_()
        self._report_abs_cmd.zero_()

    # ------------------------------------------------------------------
    # Command roll-out
    # ------------------------------------------------------------------

    def _lead_times(self) -> Tensor:
        """Lead times of the conditioned slots, [envs, steps].

        Fixed ladder here. Per-env rather than a bare [steps] vector so a
        subclass can set the deadline from the task -- e.g. scaling it with
        the distance left to run, instead of demanding the same 1 s of a goal
        1 m away and one 8 m away.
        """
        return self._offsets.unsqueeze(0).expand(self.env.num_envs, -1)

    def _command(self) -> Tensor:
        """The sibling steering component's published [forward, yaw, lateral].

        command(), not the raw _target: when that component shapes commands
        into the achievable region, the targets must be built from the shaped
        value or the robot is handed one command and scored on another.
        """
        component = self.env.control_manager.components[self.config.command_component]
        return component.command()

    def _rollout(
        self, root_pos: Tensor, root_rot: Tensor, lead: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Integrate the command from the live root pose.

        With a constant body-frame velocity (v_f, v_s) and yaw rate w, heading
        is h(t) = h0 + w*t and the world velocity is
            xdot = v_f*cos(h) - v_s*sin(h)
            ydot = v_f*sin(h) + v_s*cos(h)
        which integrates in closed form; the w -> 0 limit is the straight line.

        Returns:
            Tuple of (target_xy [envs, steps, 2], target_heading [envs, steps]).
        """
        cmd = self._command()
        fwd = cmd[:, 0:1]
        turn = cmd[:, 1:2]
        side = cmd[:, 2:3]

        h0 = rotations.calc_heading(root_rot, True).unsqueeze(-1)  # [envs, 1]
        t = lead  # [envs, steps]
        h = h0 + turn * t  # [envs, steps]

        straight = turn.abs() < _STRAIGHT_EPS
        turn_safe = torch.where(straight, torch.ones_like(turn), turn)
        arc_a = (torch.sin(h) - torch.sin(h0)) / turn_safe
        arc_b = (torch.cos(h) - torch.cos(h0)) / turn_safe
        # w -> 0 limits of the two integrals.
        lin_a = torch.cos(h0) * t
        lin_b = -torch.sin(h0) * t
        a = torch.where(straight, lin_a, arc_a)
        b = torch.where(straight, lin_b, arc_b)

        x = root_pos[:, 0:1] + fwd * a + side * b
        y = root_pos[:, 1:2] - fwd * b + side * a
        return torch.stack([x, y], dim=-1), h

    def populate_context(self, ctx: EnvContext) -> None:
        """Publish the commanded base-link targets as the sparse conditioning."""
        # ctx.mimic still comes from the motion library: the tracking rewards
        # and evaluators read it, and nothing in the MaskedMimic prior does.
        MimicControl.populate_context(self, ctx)

        num_envs = self.env.num_envs
        num_steps = self.config.num_masked_future_steps
        device = self.env.device

        root_state = self.env.simulator.get_root_state()
        lead = self._lead_times()
        target_xy, target_heading = self._rollout(
            root_state.root_pos, root_state.root_rot, lead
        )

        ground = self.env.terrain.get_ground_heights(
            target_xy.reshape(-1, 2)
        ).view(num_envs, num_steps)
        cmd = self._command()
        speed_cmd = cmd[:, [0, 2]].norm(dim=-1, keepdim=True)
        current_height = (
            root_state.root_pos[:, 2:3]
            - self.env.terrain.get_ground_heights(
                root_state.root_pos[:, :2]
            ).view(num_envs, 1)
        )
        height = self._target_heights(speed_cmd, current_height)  # [envs, 1]
        target_pos = torch.cat(
            [target_xy, (ground + height).unsqueeze(-1)], dim=-1
        )  # [envs, steps, 3]

        up = torch.zeros(num_envs * num_steps, 3, device=device, dtype=torch.float)
        up[:, 2] = 1.0
        if self.config.preserve_tilt:
            # Turn the body's CURRENT orientation about world up to the
            # commanded heading. The target then carries the roll and pitch
            # the dog actually has, the way a clip's rigid_body_rot does,
            # instead of also demanding it be perfectly level.
            root_rot = root_state.root_rot
            heading_now = rotations.calc_heading(root_rot, True).unsqueeze(-1)
            delta = (target_heading - heading_now).reshape(-1)
            spin = rotations.quat_from_angle_axis(delta, up, True)
            current = root_rot.unsqueeze(1).expand(-1, num_steps, -1).reshape(-1, 4)
            target_rot = rotations.quat_mul(spin, current, True).view(
                num_envs, num_steps, 4
            )
        else:
            target_rot = rotations.quat_from_angle_axis(
                target_heading.reshape(-1), up, True
            ).view(num_envs, num_steps, 4)

        # Masked-out bodies are zeroed downstream; they only need to be finite.
        ref_pos = target_pos.unsqueeze(2).repeat(1, 1, self._num_bodies, 1)
        ref_rot = self._identity_quat.view(1, 1, 1, 4).repeat(
            num_envs, num_steps, self._num_bodies, 1
        )
        ref_pos[:, :, self._root_body_id, :] = target_pos
        ref_rot[:, :, self._root_body_id, :] = target_rot

        self._marker_target_pos = target_pos

        ctx.masked_mimic = MaskedMimicContext(
            mimic=ctx.mimic,
            ref_pos=ref_pos,
            ref_rot=ref_rot,
            target_times=self.target_times,
            time_offsets=lead,
            target_poses_masks=self.masked_mimic_target_poses_masks,
            target_bodies_masks=self.masked_mimic_target_bodies_masks,
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, VisualizationMarkerConfig]:
        """One sphere per conditioned lead time, tracing the commanded arc.

        Replaces the parent's per-body blue/yellow/red spheres: only the base
        link is conditioned, so there is one target per future step and they
        all sit in the same 0.2-1.0 s window. The velocity readouts the user
        steers by (green direction arrow, spinning turn dial) come from the
        sibling steering component.
        """
        if headless:
            return {}
        return {
            "mm_steering_targets": VisualizationMarkerConfig(
                type="sphere",
                color=(1.0, 0.25, 0.25),
                markers=[
                    MarkerConfig(size="small")
                    for _ in range(self.config.num_masked_future_steps)
                ],
            ),
        }

    def get_markers_state(self) -> Dict[str, MarkerState]:
        if self.env.simulator.headless or not self._initialized:
            return {}
        target_pos = getattr(self, "_marker_target_pos", None)
        if target_pos is None:
            return {}
        return {
            "mm_steering_targets": MarkerState(
                translation=target_pos.view(self.env.num_envs, -1, 3),
                orientation=self._identity_quat.view(1, 1, 4).repeat(
                    self.env.num_envs, self.config.num_masked_future_steps, 1
                ),
            ),
        }
