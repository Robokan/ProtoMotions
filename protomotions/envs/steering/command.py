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
from typing import TYPE_CHECKING, Tuple

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


from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)
from protomotions.utils import rotations


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
    friction_mu: float = 0.6
    rate_frac_min: float = 0.1
    rate_frac_max: float = 1.0
    # Skill buttons ("press B to sit"). The state is published for the agent's
    # latent-bank override and appended to the task obs so a learned HLC can
    # also see which button is held. 0 = no buttons, obs width unchanged.
    num_buttons: int = 0
    difficulty_epochs: int = 1
    # "gamepad" = a /dev/input/js0 reader drives the CURRENTLY SELECTED
    # robot's commands at inference (camera-followed env, cycle with =/-);
    # all other envs keep the random-walk generator. Auto-detected at viewer
    # launch by the steering experiment's inference hook; force with
    # --command-source steering_cmd=gamepad|random. Never set in training.
    # "external" hands the command to another component (e.g. a ball-chase
    # pursuit controller) via set_command(): the random walk is suppressed,
    # but the first-order ramp still runs, so a controller can set a step
    # target and get natural acceleration out of it for free.
    command_source: str = None
    # Project every command into the region the robot can actually attempt
    # before the policy (or the reward) sees it -- see
    # shape_command_to_achievable. Matters most for teleop: a gamepad stick
    # can be pushed into corners of the box the corpus never demonstrates
    # (sprint-and-strafe is 19 s of the whole go2 corpus), and full deflection
    # should mean "as much as is possible here", not a number with nothing
    # behind it. OFF by default so the ASE HLC keeps the exact reward-side-only
    # behaviour it was trained with.
    shape_commands: bool = False


# Original _adjust_by_rate hard-codes dt = 0.1 (not the sim step).
_RATE_DT = 0.1


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
        # [num_envs, num_buttons], 1.0 while held. Driven by teleop (a
        # gamepad drives env 0) or scripted tests; zeros during training
        # unless something sets it.
        self.button_state = torch.zeros(
            num_envs, config.num_buttons, device=device, dtype=torch.float
        )
        # Spinning-phase accumulator for the turn marker (rad). The original
        # IsaacLabASE game task animates circle_arrows at the commanded yaw
        # rate; the marker's spin SPEED is the turn command readout.
        self._turn_anim = torch.zeros(num_envs, device=device, dtype=torch.float)
        self._gamepad = None

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

        # ONLY forward random-walks; turn and side are drawn ABSOLUTELY and
        # afresh every interval. The original assigns them outright --
        #     desired_forward_vel = add_with_bounce_min_max(desired_forward_vel,
        #                                                   delta, min, max)
        #     desired_side_vel    = desired_side_vel      # plain '='
        #     desired_turn_vel    = desired_turn_vel      # plain '='
        # (amp_game_controller_task._reset_task) -- and scales turn by
        # turn_vel_range (= turn_vel_max), NOT by the full span.
        #
        # Walking the turn channel with a span-scaled delta (the old port)
        # gave it lag-1 autocorrelation ~0.55 and an almost uniform marginal:
        # near-max yaw was commanded ~25% of the time instead of ~1%, and a
        # given env could sit pinned at one turn direction across several
        # resample windows (Eric, 2026-09-13: the command "is only doing it a
        # little, not for the whole range"). Beta(4,4)'s whole purpose -- the
        # original's comment calls it "how much it is centered around 0" --
        # only survives on an ABSOLUTE draw.
        d = self._difficulty
        beta = torch.distributions.Beta(
            torch.tensor(4.0, device=device), torch.tensor(4.0, device=device)
        )
        fwd_delta = (2.0 * torch.rand(n, device=device) - 1.0) * span[0] * d
        turn_abs = (
            (2.0 * beta.sample((n,)) - 1.0) * self.config.turn_vel_max * d
        )
        side_abs = (
            (2.0 * torch.rand(n, device=device) - 1.0)
            * self.config.side_vel_max
            * d
        )

        fwd_walk = self._reflect(
            self._desired[env_ids, 0] + fwd_delta, self._lo[0], self._hi[0]
        )
        self._desired[env_ids] = torch.stack(
            [fwd_walk, turn_abs, side_abs], dim=-1
        )

        # Dimensionless per-step fraction of the REMAINING gap, matching the
        # original's rate ~ U(0.1, 1.0) used as (target-current) * rate * 0.1.
        self._rates[env_ids] = self.config.rate_frac_min + (
            self.config.rate_frac_max - self.config.rate_frac_min
        ) * torch.rand(n, 3, device=device)

        change_steps = torch.randint(
            low=self.config.heading_change_steps_min,
            high=self.config.heading_change_steps_max,
            size=(n,),
            device=device,
            dtype=torch.int64,
        )
        progress = self.env.progress_buf[env_ids]
        # A fresh episode, i.e. NOT the mid-episode resample path that reuses
        # this method. reset_buf/terminate_buf mark the envs whose episode was
        # just ended -- but the very first reset of a run has neither set
        # (they are freshly allocated zeros, and env.reset() only clears them
        # AFTER control_manager.reset), so progress_buf == 0 is what catches
        # it. Without that clause the opening episode skips the spawn-velocity
        # seeding below and every robot starts from a zero command no matter
        # what pose RSI dropped it in; in a viewer with terminations off that
        # first episode is the ONLY episode, so nothing is ever seeded
        # (Eric, 2026-09-19: "why is it always starting with everything at
        # 0"). A resample cannot collide with this: _change_steps is set to
        # progress + at least heading_change_steps_min.
        is_env_reset = (
            self.env.reset_buf[env_ids]
            | self.env.terminate_buf[env_ids]
            | (progress == 0)
        )
        progress = torch.where(is_env_reset, torch.zeros_like(progress), progress)
        self._change_steps[env_ids] = progress + change_steps
        # Fresh episodes: seed BOTH the target and the desired command from
        # the velocity the robot actually spawned with (RSI drops it at a
        # random mocap frame, so it may already be mid-trot). Original
        # IsaacLabASE comment, ported verbatim in spirit: "When we reset we
        # put the robot in a random place in the motion capture. We want to
        # make sure we set the desired velocities to match what is in the
        # motion capture. That way it will learn what poses match what
        # velocities. This is important when doing gait transitions."
        # Commanding zero to a spawned-running robot punishes it for the pose
        # it was placed in. reset() runs after the simulator state write
        # (env.py: simulator.reset_envs -> control_manager.reset), so the
        # spawn state is readable here.
        fresh = env_ids[is_env_reset]
        if len(fresh) > 0:
            root = self.env.simulator.get_root_state(fresh)
            heading = rotations.calc_heading(root.root_rot, True)
            cos_h, sin_h = torch.cos(heading), torch.sin(heading)
            vx, vy = root.root_vel[:, 0], root.root_vel[:, 1]
            seeded = torch.stack(
                [
                    cos_h * vx + sin_h * vy,       # forward (heading frame)
                    root.root_ang_vel[:, 2],       # yaw rate
                    -sin_h * vx + cos_h * vy,      # lateral (heading frame)
                ],
                dim=-1,
            )
            # Out-of-range mocap velocities: desired is clipped so the robot
            # ramps back into range (original clip-the-desired behavior, with
            # clip_initial_targets_also semantics for the live target too).
            # The Go2 chain sets clip_initial_targets_also = False, so ONLY
            # the desired command is clipped: the live target keeps the raw
            # mocap spawn velocity and ramps back into range.
            self._target[fresh] = seeded
            self._desired[fresh] = torch.clamp(seeded, self._lo, self._hi)
        if self.button_state.shape[1]:
            self.button_state[env_ids[is_env_reset]] = 0.0

    def set_command(self, cmd: Tensor, env_ids: Tensor = None) -> None:
        """Drive the command from outside (command_source="external").

        Sets the DESIRED command; the existing first-order lag ramps the live
        target toward it, so an external controller inherits the same
        acceleration profile the random walk gets instead of stepping the
        command discontinuously.
        """
        if env_ids is None:
            self._desired[:] = torch.clamp(cmd, self._lo, self._hi)
        else:
            self._desired[env_ids] = torch.clamp(cmd, self._lo, self._hi)

    def step(self):
        external = getattr(self.config, "command_source", None) == "external"
        if not external:
            resample_mask = self.env.progress_buf >= self._change_steps
            env_ids = resample_mask.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset(env_ids)

        # First-order lag toward the command, exactly as the original's
        # _adjust_by_rate: current += (target - current) * rate * dt, dt=0.1
        # hard-coded and unrelated to the sim step. The old port slewed at a
        # constant speed instead, which arrived (and overshot the original's
        # attenuation) 2-5x faster. Difficulty is already baked into
        # _desired at draw time, as the original does.
        self._target += (self._desired - self._target) * self._rates * _RATE_DT

        if getattr(self.config, "command_source", None) == "gamepad":
            if self._gamepad is None:
                from protomotions.envs.steering.gamepad import GamepadReader

                self._gamepad = GamepadReader(
                    num_buttons=max(self.config.num_buttons, 1)
                )
            channels, buttons, pad_active = self._gamepad.state()
            pad = torch.tensor(
                channels, device=self.env.device, dtype=torch.float
            )
            # The pad drives ONLY the currently selected robot (the one the
            # viewer camera follows; cycle with =/-). Everyone else keeps
            # their random-walk commands, exactly like the original task.
            sel = 0
            cam = getattr(self.env.simulator, "_camera_target", None)
            if isinstance(cam, dict):
                sel = int(cam.get("env", 0))
            # Only override while the pad is actually being used (per Eric):
            # an idle controller releases the selected robot back to the
            # random generator, so it wanders like every other env until a
            # stick moves again.
            if pad_active:
                scale = torch.stack([self._hi[0], self._hi[1], self._hi[2]])
                self._target[sel] = torch.clamp(pad * scale, self._lo, self._hi)
                if self.config.num_buttons > 0:
                    self.button_state[sel] = torch.tensor(
                        buttons[: self.config.num_buttons],
                        device=self.env.device,
                        dtype=torch.float,
                    )

        # Advance the turn marker's spin phase at the commanded yaw rate,
        # wrapped to +/-2pi exactly as the original does.
        self._turn_anim += self._target[:, 1] * self.env.dt
        two_pi = 2.0 * torch.pi
        self._turn_anim = torch.where(
            self._turn_anim > two_pi, self._turn_anim - two_pi, self._turn_anim
        )
        self._turn_anim = torch.where(
            self._turn_anim < -two_pi, self._turn_anim + two_pi, self._turn_anim
        )

    def command(self) -> Tensor:
        """The command as published: [forward, yaw, lateral], per env.

        This -- not the raw _target -- is what observations, rewards, markers
        and any downstream consumer should read, so that shaping is applied
        exactly once and everyone sees the same thing.
        """
        if not self.config.shape_commands:
            return self._target
        fwd, side = shape_command_to_achievable(
            self._target[:, 0],
            self._target[:, 1],
            self._target[:, 2],
            forward_vel_max=self.config.forward_vel_max,
            side_vel_max=self.config.side_vel_max,
            friction_mu=self.config.friction_mu,
        )
        return torch.stack([fwd, self._target[:, 1], side], dim=-1)

    def set_buttons(self, state: Tensor, env_ids: Tensor = None) -> None:
        """Teleop hook: set held buttons (1.0 = held) for some/all envs."""
        if env_ids is None:
            self.button_state[:] = state
        else:
            self.button_state[env_ids] = state

    def create_visualization_markers(self, headless: bool):
        """The IsaacLabASE game-controller indicators, original assets:
        a green direction arrow offset by the commanded velocity vector and
        the circle_arrows turn dial spinning at the commanded yaw rate.
        Inference-only by construction -- headless returns nothing, and
        training always runs headless."""
        if headless:
            return {}
        return {
            "steering_dir": VisualizationMarkerConfig(
                type="usd",
                usd_path="usd/markers/direction_marker_green.usd",
                markers=[MarkerConfig(size="regular")],
            ),
            "steering_turn": VisualizationMarkerConfig(
                type="usd",
                usd_path="usd/markers/circle_arrows.usd",
                markers=[MarkerConfig(size="regular")],
            ),
        }

    def get_markers_state(self):
        """Original marker math (amp_game_controller_task._update_markers):
        direction marker at root_xy + heading_frame_velocity * 0.5, oriented
        to the robot's heading; turn dial at the root, spun by the
        accumulated phase, flipped about x for negative yaw so the arrows
        visually reverse. Buttons held -> markers sink to z=-1 (hidden)."""
        if self.env.simulator.headless:
            return {}
        root_state = self.env.simulator.get_root_state()
        root_pos = root_state.root_pos
        heading = rotations.calc_heading(root_state.root_rot, True)
        facing = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        side = torch.stack(
            [torch.cos(heading + torch.pi / 2), torch.sin(heading + torch.pi / 2)],
            dim=-1,
        )
        up = torch.zeros_like(root_pos)
        up[..., 2] = 1.0
        heading_q = rotations.quat_from_angle_axis(heading, up, True)

        ground = self.env.terrain.get_ground_heights(root_pos[..., :2]).view(-1)

        # Direction marker: offset by half the commanded velocity vector.
        # The SHAPED command, so the arrow shows what is actually being asked
        # rather than the raw stick.
        cmd = self.command()
        fwd = cmd[:, 0].unsqueeze(-1) * 0.5
        lat = cmd[:, 2].unsqueeze(-1) * 0.5
        dir_pos = root_pos.clone()
        dir_pos[..., 0:2] = root_pos[..., 0:2] + facing * fwd + side * lat
        dir_pos[..., 2] = ground + 0.02

        # Turn dial: at the root, spun by the SIGNED accumulated phase.
        # The original (amp_game_controller_task._update_markers) computes a
        # flip-about-x for negative yaw but leaves it COMMENTED OUT and
        # assigns the plain signed rotation:
        #     turn_marker_rot = quat_from_angle_axis(angular_rotation_position,
        #                                            heading_axis)
        # The phase already carries the command's sign, so applying the flip
        # as well double-negates it -- R_x(pi) . R_z(t) == R_z(-t) . R_x(pi) --
        # and every dial spins the SAME way regardless of turn direction
        # (Eric, 2026-09-13: "the rings only turn to the left").
        turn_q = rotations.quat_from_angle_axis(self._turn_anim, up, True)
        turn_pos = root_pos.clone()
        turn_pos[..., 2] = ground + 0.01

        # Hide both while any skill button is held (original behavior).
        if self.button_state.shape[1]:
            held = self.button_state.any(dim=-1)
            dir_pos[held, 2] = -1.0
            turn_pos[held, 2] = -1.0

        n = self.env.num_envs
        return {
            "steering_dir": MarkerState(
                translation=dir_pos.view(n, -1, 3),
                orientation=heading_q.view(n, -1, 4),
            ),
            "steering_turn": MarkerState(
                translation=turn_pos.view(n, -1, 3),
                orientation=turn_q.view(n, -1, 4),
            ),
        }

    def populate_context(self, ctx) -> None:
        cmd = self.command()
        ctx.steering_cmd = SteeringCommandContext(
            fwd_cmd=cmd[:, 0],
            turn_cmd=cmd[:, 1],
            side_cmd=cmd[:, 2],
            buttons=self.button_state,
        )


# =============================================================================
# Observation kernel
# =============================================================================


def compute_steering_command_obs(
    root_rot: Tensor,
    root_vel: Tensor,
    root_ang_vel: Tensor,
    fwd_cmd: Tensor,
    turn_cmd: Tensor,
    side_cmd: Tensor,
    buttons: Tensor,
    w_last: bool = True,
) -> Tensor:
    """Command + gait proprioception observation, 12 dims (+1 per button):
    [fwd_cmd, turn_cmd, side_cmd, projected_gravity(3), root_ang_vel(3, WORLD
    frame, as the original), heading-frame local linear velocity(3)].
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
            root_ang_vel,
            local_vel,
            buttons,
        ],
        dim=-1,
    )


# =============================================================================
# Reward kernel
# =============================================================================


def shape_command_to_achievable(
    fwd_cmd: Tensor,
    turn_cmd: Tensor,
    side_cmd: Tensor,
    forward_vel_max: float,
    side_vel_max: float,
    friction_mu: float = 0.6,
) -> Tuple[Tensor, Tensor]:
    """Project a raw velocity command into the region the robot can attempt.

    Two couplings a static per-channel box cannot express:

    * calculate_safe_velocity: cap |forward| at sqrt(mu*g*R) with R=|v|/|w|,
      keeping the sign and leaving an already-safe command untouched. The old
      port used mu*g/|w| with torch.minimum, which is a strictly lower cap
      (2.94 vs 3.43 m/s at fwd=4, turn=2) and never capped backward commands.
    * calculate_appropriate_side_velocity: a magnitude CEILING, not a scale.
      A lateral command under the ceiling passes through UNCHANGED; only
      over-ceiling ones are clipped to it. The old port multiplied every
      lateral command by the factor, shrinking sub-ceiling targets up to 3x
      and handing a non-strafing robot 7-18% more r_side than it had earned.

    Used in two places and it matters that it is the same code: scoring the
    command (compute_steering_command_reward) and, when the control component
    sets shape_commands, the command actually handed to the policy -- so a
    gamepad pushed into a corner of the box asks for something demonstrated
    instead of a combination the corpus never shows (sprint-and-strafe is 19 s
    of the whole go2 corpus).

    Returns:
        Tuple of (forward, lateral) commands, both projected. The yaw command
        is returned unchanged by construction and is not part of the tuple.
    """
    radius = fwd_cmd.abs() / turn_cmd.abs().clamp_min(1e-6)
    safe_vel = torch.sqrt(friction_mu * _GRAVITY_MPS2 * radius) * torch.sign(
        fwd_cmd
    )
    fwd_tgt = torch.where(safe_vel.abs() >= fwd_cmd.abs(), fwd_cmd, safe_vel)

    mx = forward_vel_max * 1.1
    max_side = (1.0 - (fwd_tgt / mx).abs()) * side_vel_max
    side_tgt = torch.where(
        side_cmd.abs() > max_side, torch.sign(side_cmd) * max_side, side_cmd
    )
    return fwd_tgt, side_tgt


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
    pre_shaped: bool = False,
) -> Tensor:
    """Mean of three exponential velocity-tracking terms in the heading frame.

    Targets are reshaped (not the robot penalized) before scoring:
    - safe velocity: forward target capped by the centripetal friction limit
      mu*g/|yaw_rate| during sharp turns;
    - appropriate side velocity: lateral target CLIPPED to a ceiling that
      falls linearly to ~9% of side_vel_max as the forward target approaches
      forward_vel_max (commands below the ceiling are left alone).
    """
    heading_inv = rotations.calc_heading_quat_inv(root_rot, w_last)
    local_vel = rotations.quat_rotate(heading_inv, root_vel, w_last)
    cur_fwd = local_vel[:, 0]
    cur_side = local_vel[:, 1]
    cur_turn = root_ang_vel[:, 2]

    if pre_shaped:
        # Already projected at the command source: projecting twice is NOT a
        # no-op. The safe-velocity cap depends on the forward command through
        # R = |v|/|w|, so re-applying it to an already-capped value shrinks it
        # again, and iterating converges on mu*g/|w| -- precisely the old,
        # too-conservative formula this port replaced.
        fwd_tgt, side_tgt = fwd_cmd, side_cmd
    else:
        fwd_tgt, side_tgt = shape_command_to_achievable(
            fwd_cmd,
            turn_cmd,
            side_cmd,
            forward_vel_max=forward_vel_max,
            side_vel_max=side_vel_max,
            friction_mu=friction_mu,
        )

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
            # WORLD frame, the same field the reward scores against. The
            # original feeds self.directional_root_ang_vel (= body_ang_vel_w)
            # to BOTH obs (:793) and reward (:829); the old port fed the
            # body-frame variant to the obs only. Both are angular velocity --
            # only the frame differed -- and on this corpus the z components
            # correlate 0.995, so this was a faithfulness fix, not a big one.
            "root_ang_vel": EnvContext.current.root_ang_vel,
            "fwd_cmd": EnvContext.steering_cmd.fwd_cmd,
            "turn_cmd": EnvContext.steering_cmd.turn_cmd,
            "side_cmd": EnvContext.steering_cmd.side_cmd,
            "buttons": EnvContext.steering_cmd.buttons,
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
    pre_shaped: bool = False,
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
            "pre_shaped": pre_shaped,
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
