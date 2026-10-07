# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Record what the robot saw and what it was told to do, as a LeRobot dataset.

The chase is a demonstrator: it produces (front camera, proprioception) ->
(where to put the torso, by when) at whatever rate you run envs. This writes
those pairs out in the format the imitation-learning tooling already reads, so
the next step is a training run rather than a parser.

Three choices that matter more than the file format:

* **The action is egocentric.** The controller thinks in world coordinates --
  it is handed the ball's position by the simulator. A camera cannot see world
  coordinates, so an action expressed in them is unlearnable: the same picture
  would carry a different label depending on where the dog happens to be
  standing. What is recorded is the target in the robot's own heading frame,
  which is exactly what a policy looking through that camera can predict, and
  the control component converts it back.

* **The action carries the heading.** Chasing, the commanded heading is just
  atan2(dy, dx) of the target -- but searching, answering a "can you see a
  ball?", or standing after a "no", the target is the robot's own position
  and the heading is the whole instruction (the sweep, the turn to face the
  ball). So: dx, dy, how long it has, and which way to face.

* **Each throw is a question, and every frame an answer.** The episode's
  prompt is "chase the red ball" or "can you see a ball?" (task_index), and
  response_index names what the dog would say right then -- "let me look",
  "yes, I see a ball", "no, I don't see a ball" (meta/responses.jsonl) --
  so a language-emitting policy has its text targets without re-recording.

* **Privileged columns are labels, never inputs.** ``ball_visible`` and
  ``ball_xy`` are written because they make good auxiliary supervision and an
  honest evaluation ("did it know where the ball was, or did it get lucky"),
  but nothing the student is fed at inference may come from them.

One timing fact worth knowing: the frame is rendered during the physics
step, BEFORE the control components update, so every image is one control
step (20 ms) older than the labels beside it. For continuous motion that is
a few centimetres, inside the measured alignment error. For a re-throw it is
a different ball entirely -- the image still shows the old one, huge and
underfoot, while the label describes a new one metres away. Those frames are
dropped (see _thrown), which is 0.4% of them.

Run it headless. With a viewer open, every visualization marker renders into
the camera too, including the conditioned-target ladder -- which is the label
drawn on top of the image (see VisualizationMarkerConfig.camera_visible).

What lands here is a STAGING layout, not the dataset you train on: one
parquet and one mp4 per episode, with meta/{info,episodes,episodes_stats,
tasks} -- LeRobot v2.1 shaped, so it is self-describing and their
``convert_dataset_v21_to_v30.py`` can also read it.

The dataset lerobot actually reads is **v3.0**, which it is not backward
compatible with ("not backward compatible with v2.1", and it raises rather
than guess). v3.0 concatenates many episodes into shared parquet and mp4
files addressed by row and timestamp ranges, and hand-emitting that is a good
way to produce something that loads happily and serves the wrong frames. So
lerobot writes it, from these files, via ``scripts/chase_to_lerobot.py`` --
run in the lerobot venv, since lerobot cannot be installed beside Isaac Sim.
Recording stays here, where it is cheap and has no dependencies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.utils import rotations

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


_CODEBASE_VERSION = "v2.1"
_CHUNK = 0


# The conditioning a policy has to produce to drive the go2 through
# MaskedMimic: for every conditionable body, where it should be and how it
# should be turned, each switched on or off, and the time to be there. The
# chase prior is shown ONE future slot (MaskedMimicGoalControlConfig.
# visible_targets), so the final slot is the whole conditioning. A chase is
# "trunk here, facing there"; a sit or beg adds every leg body, positioned and
# oriented (measured: with the feet' positions alone a beg would not come back
# down -- see MaskedMimicGoalControlConfig.pose_bodies).
MASKED_TARGET_FIELDS = ("x", "y", "z", "fwd_x", "fwd_y", "fwd_z", "up_x", "up_y", "up_z")


def masked_target_names(goal) -> List[str]:
    """Column names of masked_targets for this robot, in order."""
    names = list(goal.env.robot_config.kinematic_info.body_names)
    bodies = [names[b] for b in goal.conditionable_body_ids.tolist()]
    cols = [f"{body}_{f}" for body in bodies for f in MASKED_TARGET_FIELDS]
    cols += [f"{body}_{bit}" for body in bodies for bit in ("on", "rot_on")]
    return cols + ["seconds"]


def masked_targets(mm, goal, root, terrain) -> Tensor:
    """[N, 9B + 2B + 1] the final MaskedMimic target, in the robot's own frame.

    Per conditionable body: position, x forward and y left of the robot in its
    heading frame and z the height above the ground under the target; then
    orientation, its forward and up vectors in that frame (quat_to_tan_norm).
    Then which of those the prior is actually told (pos, rot per body), and
    the lead time of the final slot. A target it is not told is zero.
    """
    n = root.root_pos.shape[0]
    cond = goal.conditionable_body_ids.tolist()
    steps = mm.ref_pos.shape[1]
    slot = steps - 1
    inv = rotations.calc_heading_quat_inv(root.root_rot, True)
    masks = mm.target_bodies_masks.view(n, steps, len(cond), 2)[:, slot]
    world = mm.ref_pos[:, slot][:, cond]                              # [N, B, 3]
    b = world.shape[1]
    rel = world - root.root_pos.unsqueeze(1)
    ground = terrain.get_ground_heights(world[..., :2].reshape(-1, 2)).view(n, b)
    inv_b = inv.unsqueeze(1).expand(n, b, 4).reshape(-1, 4)
    local = rotations.quat_rotate(inv_b, rel.reshape(-1, 3), True).view(n, b, 3)
    local[..., 2] = world[..., 2] - ground
    rot = rotations.quat_mul(inv_b, mm.ref_rot[:, slot][:, cond].reshape(-1, 4), True)
    rot = rotations.quat_to_tan_norm(rot, True).view(n, b, 6)
    on, rot_on = masks[..., 0], masks[..., 1]
    poses = torch.cat(
        [local * on.unsqueeze(-1).float(), rot * rot_on.unsqueeze(-1).float()], dim=-1
    )
    seconds = mm.time_offsets[:, slot] if mm.time_offsets.dim() == 2 else mm.time_offsets
    return torch.cat(
        [poses.reshape(n, -1), torch.stack([on, rot_on], dim=-1).reshape(n, -1).float(),
         seconds.reshape(n, 1).float()],
        dim=-1,
    )


def language_texts(goal, source, prompts, responses, colors, n, device) -> List[tuple]:
    """(prompt, response) text per env for this frame, as the recorder writes them.

    goal is the chase demonstrator (its look mode and answer), source its
    ball command source (what was asked this throw). Shared with the
    viewer caption, so what is shown on screen is what gets recorded.
    """
    zeros = torch.zeros(n, dtype=torch.long, device=device)

    def flag(name):
        value = getattr(source, name, None)
        return (value if value is not None else zeros.bool()).tolist()

    look = (goal._look_mode() if hasattr(goal, "_look_mode") else zeros.bool()).tolist()
    any_mode = flag("any_mode")
    color = getattr(source, "color", zeros).tolist()
    throw = getattr(source, "throw_id", zeros).tolist()
    answer = getattr(goal, "_answer", zeros).tolist()
    pose = getattr(source, "pose", zeros - 1).tolist()
    pose_names = list(getattr(getattr(source, "config", None), "poses", []) or [])
    out = []
    for e in range(n):
        kind = ("look" if look[e] else "get") + ("_any" if any_mode[e] else "_color")
        if pose[e] >= 0:
            kind = f"pose_{pose_names[pose[e]]}"
        name = colors[color[e]]
        phrasings = prompts[kind]
        # One phrasing per throw: stable across the episode, varied
        # across throws and envs.
        prompt = phrasings[(throw[e] * 7 + e) % len(phrasings)]
        said = responses[kind][answer[e]]
        out.append((prompt.format(color=name), said.format(color=name)))
    return out

@dataclass
class LeRobotRecorderConfig(ControlComponentConfig):
    """Configuration for the dataset recorder."""

    _target_: str = "protomotions.envs.control.lerobot_recorder.LeRobotRecorder"

    root: str = "output/datasets/go2_chase"
    # The rate the STUDENT will run at. Frames are sampled every
    # round(1 / (fps * dt)) control steps, so the recorded spacing is the
    # spacing it will see at inference.
    fps: float = 10.0
    # One episode is this many recorded frames. The chase never terminates,
    # so episodes are a bookkeeping unit rather than a task boundary; a reset
    # cuts one short because the state jumps.
    episode_steps: int = 200
    max_episodes: int = 40
    # End the process once max_episodes are written. The chase never ends
    # by itself, so without this a finished recording keeps simulating --
    # and holding its GPU -- until someone notices.
    exit_when_done: bool = True
    # Which control component holds the demonstrator, and which holds the ball.
    goal_component: str = "masked_mimic"
    ball_component: str = "ball"
    camera: str = "front_camera"
    # Name of a CameraEye component, or None. Set, the recorded image is the
    # eye's crop of the full-resolution frame, observation.state ends with
    # the gaze that crop was taken with, and the action ends with the gaze
    # for the next frame -- see camera_eye.
    eye_component: Optional[str] = None
    # Prompts, per kind of throw: a GET or a LOOK question, naming the
    # wanted ball's colour or not ("get the ball" takes the first ball seen;
    # see BallChaseCommandSourceConfig). Each throw uses one phrasing, picked
    # by throw, so the student learns the request and not one sentence.
    # {color} is the wanted ball's colour. meta/tasks.jsonl lists every
    # expansion; task_index points into it.
    prompts: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "get_color": ["get the {color} ball", "go get the {color} ball",
                          "fetch the {color} ball"],
            "get_any": ["get the ball", "go get a ball", "fetch a ball"],
            "look_color": ["is there a {color} ball?", "can you see a {color} ball?",
                           "do you see a {color} ball?"],
            "look_any": ["is there a ball?", "can you see a ball?",
                         "do you see a ball?"],
            # Pose commands (BallChaseCommandSourceConfig.pose_frac).
            "pose_sit": ["sit", "sit down", "sit!"],
            "pose_beg": ["beg", "sit up and beg", "beg for it"],
            "pose_lie": ["lie down", "down", "lay down"],
        }
    )
    # What the dog would say, per frame: [none yet, yes, no] per kind
    # (ball_chase.ANSWER_*). {color} is the wanted ball -- for a colourless
    # request, the one it picked. meta/responses.jsonl lists every
    # expansion; response_index points into it.
    responses: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "get_color": ["looking for the {color} ball", "getting the {color} ball",
                          "I can't find a {color} ball"],
            "get_any": ["looking for a ball", "getting the {color} ball",
                        "I can't find a ball"],
            "look_color": ["let me look", "yes, I see a {color} ball",
                           "no, I don't see a {color} ball"],
            "look_any": ["let me look", "yes, I see a {color} ball",
                         "no, I don't see a ball"],
            # [getting into it, done, --]
            "pose_sit": ["sitting down", "I'm sitting", "I'm sitting"],
            "pose_beg": ["okay, I'll beg", "I'm begging", "I'm begging"],
            "pose_lie": ["lying down", "I'm lying down", "I'm lying down"],
        }
    )
    # Written even while it is the only task: a constant prompt costs one
    # column and is what makes the dataset usable later, when there are
    # several objects and the prompt has to pick one.
    robot_type: str = "go2"
    video_codec: str = "libx264"
    # Parallel envs share one world, and where they spawn is random: measured,
    # four of them sit 53-61 m apart at the default spacing and 16-26 m apart
    # at a LARGER one, because a wider scatter radius means more chances to
    # land near someone. They drift too -- the chase never resets, so each dog
    # random-walks behind its own ball. No setting guarantees isolation, so
    # this does: a frame whose env has a neighbour closer than this is not
    # recorded.
    #
    # Where to put it follows from what a neighbour looks like. At 1.87 px/deg
    # (120 deg over 224 px) an object of size s at distance d is 107*s/d px,
    # so the neighbour's BALL -- the thing that could be mistaken for ours --
    # is 26/d px and the neighbour's DOG is 43/d px. At 30 m that is a
    # 0.9 px ball and a 1.4 px dog: nothing a convolution can find. Below
    # about 13 m the ball becomes a real 2 px blob and the frame is a lie.
    # 30 m has margin and, at the separations actually observed, costs
    # nothing. 0 disables the check.
    min_neighbour_m: float = 30.0
    state_names: List[str] = field(default_factory=list)


class LeRobotRecorder(ControlComponent):
    """Write (image, state) -> (egocentric target) pairs as a LeRobot dataset."""

    config: LeRobotRecorderConfig

    def __init__(self, config: LeRobotRecorderConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self._every = max(int(round(1.0 / (config.fps * env.dt))), 1)
        self._steps = 0
        self._episode_index = 0
        self._frame_total = 0
        self._done = False
        self._episodes_meta: List[dict] = []
        self._episodes_stats: List[dict] = []
        # Per-env buffers: an episode is a run of consecutive samples from ONE
        # env, so envs are recorded in parallel and flushed independently.
        self._frames: Dict[int, List[np.ndarray]] = {}
        self._rows: Dict[int, List[dict]] = {}
        self._reset_pending = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self._root = os.path.abspath(config.root)
        self._announced = False
        # Filled on the first step: control components are constructed one
        # by one, so the ball does not exist yet while this runs.
        # A catch between two sampled frames still ends the pursuit; the
        # flag carries that across to the next sample.
        self._cut_pending = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self._last_plan: Optional[Tensor] = None
        self._dropped = 0
        self._crowded_out = 0

        effective_fps = 1.0 / (self._every * env.dt)
        if abs(effective_fps - config.fps) > 1e-6:
            print(
                f"[recorder] {config.fps} Hz is not a divisor of the "
                f"{1.0 / env.dt:.0f} Hz control rate; recording at "
                f"{effective_fps:.2f} Hz instead (every {self._every} steps)",
                flush=True,
            )
        self._fps = effective_fps

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, env_ids: Tensor) -> None:
        """A reset teleports the robot: whatever was being recorded is over.

        Keeping the frames either side of it in one episode would splice two
        unrelated trajectories together and teach the student a transition
        that cannot happen.
        """
        if len(env_ids) > 0:
            self._reset_pending[env_ids] = True

    def populate_context(self, ctx: EnvContext) -> None:
        """Recording only: publishes nothing."""

    def _plan_id(self) -> Tensor:
        """The ball's plan counter; bumps on every throw and course change."""
        ball = self.env.control_manager.components.get(self.config.ball_component)
        source = getattr(ball, "command_source", None)
        plan = getattr(source, "plan_id", None)
        if plan is None:
            return torch.zeros(
                self.env.num_envs, dtype=torch.long, device=self.env.device
            )
        return plan

    def _crowded(self) -> Tensor:
        """Envs with another robot close enough to be in frame.

        Another env's dog is an object the labels say nothing about, and its
        ball is a second red blob directly contradicting them. Distance is a
        conservative test -- it also drops frames where the neighbour is
        behind -- but it is a test that cannot miss.
        """
        limit = self.config.min_neighbour_m
        if limit <= 0 or self.env.num_envs < 2:
            return torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self.env.device
            )
        from protomotions.envs.control.env_separation import (  # noqa: PLC0415
            nearest_neighbour,
        )

        xy = self.env.simulator.get_root_state().root_pos[:, :2]
        return nearest_neighbour(xy) < limit

    def _thrown(self) -> Tensor:
        """Envs whose ball moved discontinuously since the last look.

        The image in hand was rendered before this step's control update, so
        on the step a ball is caught and re-thrown the picture shows the old
        ball -- filling the frame, about to be caught -- and the label points
        at the new one. Unlabelable: drop it rather than teach it. Measured
        at 0.4% of frames, and every one of them was a catch.
        """
        plan = self._plan_id()
        if self._last_plan is None:
            self._last_plan = plan.clone()
            return torch.zeros_like(plan, dtype=torch.bool)
        moved = plan != self._last_plan
        self._last_plan = plan.clone()
        return moved

    def step(self) -> None:
        if self._done:
            return
        for env_id in self._reset_pending.nonzero(as_tuple=False).flatten().tolist():
            self._flush(env_id, reason="reset")
        self._reset_pending[:] = False

        # Checked every control step, not only the recorded ones. Two
        # separate consequences, and conflating them was a bug: a throw ENDS
        # THE EPISODE whenever it happens, but only makes the frame
        # unusable if it happens on the step that frame was rendered for.
        thrown = self._thrown()
        self._cut_pending |= thrown
        crowded = self._crowded()
        self._steps += 1
        if self._steps % self._every != 0:
            return

        images = self.env.simulator.get_camera_images()
        frame_batch = images.get(self.config.camera)
        if frame_batch is None:
            if not self._announced:
                print(
                    f"[recorder] camera '{self.config.camera}' is not "
                    "producing images -- nothing will be recorded. Is "
                    "--camera set?",
                    flush=True,
                )
                self._announced = True
            return

        if not self._announced:
            self._report_foot_sensors()
            self._announced = True
        state = self._state()
        action, extras = self._action()
        eye = self._eye()
        if eye is not None:
            frame_batch = eye.crop(frame_batch)
        frames = frame_batch.detach().cpu().numpy()
        if frames.shape[-1] == 4:
            frames = frames[..., :3]

        for env_id in range(self.env.num_envs):
            # An episode is ONE PURSUIT: throw to catch. It ends at the
            # catch because past that point the future stops being
            # predictable from the present -- the next ball has not been
            # thrown yet and is in no pixel of this frame. A chunk spanning
            # the boundary would be asking a policy to invent one, which it
            # will duly learn to do. Ending here also keeps the timeline
            # honest when a frame is skipped: writing the next frame as
            # though it followed immediately would put a 200 ms step inside
            # a 10 Hz episode, which lerobot checks.
            if bool(self._cut_pending[env_id]):
                self._flush(env_id, reason="catch")
                self._cut_pending[env_id] = False
            if bool(thrown[env_id]):
                # The throw landed on the step this frame was rendered for,
                # so the picture is of the old ball and the labels describe
                # the new one. Unlabelable.
                self._dropped += 1
                continue
            if bool(crowded[env_id]):
                self._crowded_out += 1
                self._flush(env_id, reason="neighbour")
                continue
            rows = self._rows.setdefault(env_id, [])
            self._frames.setdefault(env_id, []).append(
                np.ascontiguousarray(frames[env_id])
            )
            rows.append(
                {
                    "observation.state": state[env_id],
                    "action": action[env_id],
                    "ball_visible": bool(extras["visible"][env_id]),
                    "ball_xy": extras["ball_xy"][env_id],
                    "task_index": int(extras["task_index"][env_id]),
                    "response_index": int(extras["response_index"][env_id]),
                    **({
                        "input_prev_response_index": int(extras["input_prev_response_index"][env_id]),
                        "input_prev_heading": float(extras["input_prev_heading"][env_id]),
                    } if "input_prev_heading" in extras else {}),
                }
            )
            if len(rows) >= self.config.episode_steps:
                self._flush(env_id, reason="full")
                if self._done:
                    return

    # ------------------------------------------------------------------
    # What gets written
    # ------------------------------------------------------------------

    def _state(self) -> np.ndarray:
        """Proprioception -- and only things a real go2 could measure.

        No ball, no world position, no heading: an IMU (gravity direction and
        body rates), the body-frame velocity a state estimator provides, and
        the joints. Put anything else here and the dataset trains a policy
        that cannot be deployed.
        """
        sim = self.env.simulator
        root = sim.get_root_state()
        rot = root.root_rot
        gravity = torch.zeros_like(root.root_pos)
        gravity[:, 2] = -1.0
        parts = [
            rotations.quat_rotate_inverse(rot, gravity, True),
            rotations.quat_rotate_inverse(rot, root.root_vel, True),
            rotations.quat_rotate_inverse(rot, root.root_ang_vel, True),
        ]
        dof = sim.get_dof_state()
        parts.append(dof.dof_pos)
        parts.append(dof.dof_vel)
        # Load under each foot. A go2 has force sensors there, so this is
        # proprioception and not a simulator privilege -- and it is what the
        # demonstrator reads to decide which way to sweep when it loses
        # sight of the ball (see ball_chase.search_follows_lean). Leaving it
        # out would make that decision unlearnable: the student would see
        # two identical empty frames labelled turn-left and turn-right.
        load = self._foot_load()
        if load is not None:
            parts.append(load)
        # Where the eye was pointed for this frame: without it a zoomed-in
        # near ball and a far ball in a wide view are the same picture.
        eye = self._eye()
        if eye is not None:
            parts.append(eye.gaze())
        return torch.cat(parts, dim=-1).detach().cpu().numpy().astype(np.float32)

    def _eye(self):
        if not self.config.eye_component:
            return None
        return self.env.control_manager.components.get(self.config.eye_component)

    def _report_foot_sensors(self) -> None:
        """Say once whether the foot loads are real, because zeros are not.

        A dead column is worse than a missing one: the width looks right,
        the demonstrator's lean rule silently collapses to a constant, and
        nothing complains until the trained policy only ever turns one way.
        """
        sim = self.env.simulator
        sensors = getattr(sim, "_contact_sensor_map", {})
        load = self._foot_load()
        print(
            f"[recorder] contact sensors: {len(sensors)} "
            f"({sorted(sensors)[:6]}), foot load "
            f"{'unreadable' if load is None else load[0].tolist()}",
            flush=True,
        )

    def _foot_load(self) -> Optional[Tensor]:
        """Contact force magnitude under each contact body, or None.

        The populated field is rigid_body_contact_forces -- RobotState also
        carries a rigid_body_contacts that this path leaves as None, and
        reading that one drops the foot loads out of the state vector
        without a word.
        """
        ids = getattr(self.env, "contact_body_ids", None)
        if ids is None or len(ids) == 0:
            return None
        contacts = self.env.simulator.get_bodies_contact_buf()
        forces = getattr(contacts, "rigid_body_contact_forces", None)
        if forces is None:
            forces = getattr(contacts, "rigid_body_contacts", None)
        if forces is None:
            return None
        return forces[:, ids].norm(dim=-1)

    def _action(self):
        """The demonstrator's target, in the robot's own heading frame."""
        goal = self.env.control_manager.components[self.config.goal_component]
        root = self.env.simulator.get_root_state()
        heading = rotations.calc_heading(root.root_rot, True)
        cos, sin = torch.cos(-heading), torch.sin(-heading)

        def to_body(world_xy):
            d = world_xy - root.root_pos[:, :2]
            return torch.stack(
                [cos * d[:, 0] - sin * d[:, 1], sin * d[:, 0] + cos * d[:, 1]],
                dim=-1,
            )

        # The action IS the MaskedMimic conditioning -- the final target the
        # demonstrator handed the prior this step (see masked_targets). The
        # DAgger driver keeps the referee's in _label_mm; otherwise it is the
        # demonstrator's own.
        mm = getattr(goal, "_label_mm", None)
        if mm is None:
            mm = getattr(goal, "_last_mm", None)
        action = masked_targets(mm, goal, root, self.env.terrain)
        eye = self._eye()
        if eye is not None:
            gaze = eye.rule_gaze() if hasattr(eye, "rule_gaze") else eye.next_gaze()
            action = torch.cat([action, gaze], dim=-1)

        ball_xy = to_body(goal._goal_xy())
        visible = (
            getattr(goal, "_ball_in_view", goal._ball_seen)
            if goal._unprivileged()
            else torch.ones_like(goal._deadline, dtype=torch.bool)
        )
        task_index, response_index = self._language(goal)
        extras = {
            "visible": visible.detach().cpu().numpy(),
            "ball_xy": ball_xy.detach().cpu().numpy().astype(np.float32),
            "task_index": task_index,
            "response_index": response_index,
        }
        # A VLA driving (DAgger): the inputs it had are its OWN previous
        # answer and heading, not the demonstrator's -- record those.
        inputs = getattr(goal, "_vla_inputs_now", None)
        if inputs is not None:
            table = {t: i for i, t in enumerate(self._responses())}
            prev_text, prev_heading = inputs
            extras["input_prev_response_index"] = np.array(
                [table.get(t, -1) for t in prev_text], dtype=np.int64
            )
            extras["input_prev_heading"] = prev_heading.detach().cpu().numpy().astype(np.float32)
        return action.detach().cpu().numpy().astype(np.float32), extras

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _flush(self, env_id: int, reason: str) -> None:
        rows = self._rows.get(env_id) or []
        frames = self._frames.get(env_id) or []
        self._rows[env_id] = []
        self._frames[env_id] = []
        # A stub is worse than nothing: too short to slice an action chunk out
        # of, and it still costs an episode index. Cutting at every catch
        # means most episodes are one pursuit long, a few seconds -- which is
        # the natural unit anyway.
        if len(rows) < max(int(self._fps), 2):
            return
        if self._done:
            return

        index = self._episode_index
        self._write_video(index, frames)
        self._write_parquet(index, rows)
        self._episodes_meta.append(
            {
                "episode_index": index,
                # One throw, one question: the episode's prompt.
                "tasks": [self._tasks()[rows[0]["task_index"]]],
                "length": len(rows),
            }
        )
        self._episodes_stats.append(
            {"episode_index": index, "stats": self._stats(rows, frames)}
        )
        self._frame_total += len(rows)
        self._episode_index += 1
        print(
            f"[recorder] episode {index}: {len(rows)} frames from env "
            f"{env_id} ({reason}), {self._frame_total} total",
            flush=True,
        )
        self._write_meta()
        if self._episode_index >= self.config.max_episodes:
            self._done = True
            print(
                f"[recorder] done: {self._episode_index} episodes, "
                f"{self._frame_total} frames at {self._fps:.1f} Hz in "
                f"{self._root} ({self._dropped} dropped at re-throws, "
                f"{self._crowded_out} with a robot too close)",
                flush=True,
            )
            if self.config.exit_when_done:
                import sys  # noqa: PLC0415

                print("[recorder] exiting (exit_when_done)", flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                # Every file is closed; skip the simulator's slow teardown.
                os._exit(0)

    def _video_path(self, index: int) -> str:
        return os.path.join(
            self._root,
            "videos",
            f"chunk-{_CHUNK:03d}",
            f"observation.images.{self.config.camera}",
            f"episode_{index:06d}.mp4",
        )

    def _write_video(self, index: int, frames: List[np.ndarray]) -> None:
        import imageio.v2 as imageio  # noqa: PLC0415

        path = self._video_path(index)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        writer = imageio.get_writer(
            path,
            fps=self._fps,
            codec=self.config.video_codec,
            pixelformat="yuv420p",
            macro_block_size=1,
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    def _write_parquet(self, index: int, rows: List[dict]) -> None:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        n = len(rows)
        base = self._frame_total
        table = pa.table(
            {
                "observation.state": [r["observation.state"] for r in rows],
                "action": [r["action"] for r in rows],
                "ball_visible": pa.array(
                    [r["ball_visible"] for r in rows], type=pa.bool_()
                ),
                "ball_xy": [r["ball_xy"] for r in rows],
                "timestamp": pa.array(
                    [i / self._fps for i in range(n)], type=pa.float32()
                ),
                "frame_index": pa.array(list(range(n)), type=pa.int64()),
                "episode_index": pa.array([index] * n, type=pa.int64()),
                "index": pa.array(list(range(base, base + n)), type=pa.int64()),
                "task_index": pa.array(
                    [r["task_index"] for r in rows], type=pa.int64()
                ),
                "response_index": pa.array(
                    [r["response_index"] for r in rows], type=pa.int64()
                ),
                **({
                    "input_prev_response_index": pa.array(
                        [r["input_prev_response_index"] for r in rows], type=pa.int64()
                    ),
                    "input_prev_heading": pa.array(
                        [r["input_prev_heading"] for r in rows], type=pa.float32()
                    ),
                } if "input_prev_heading" in rows[0] else {}),
            }
        )
        path = os.path.join(
            self._root, "data", f"chunk-{_CHUNK:03d}", f"episode_{index:06d}.parquet"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(table, path)

    def _stats(self, rows: List[dict], frames: List[np.ndarray]) -> dict:
        """Per-episode statistics, in the shapes LeRobot expects.

        Images are reduced per channel and rescaled to [0, 1]; vectors are
        reduced per element.
        """

        def vec(name):
            a = np.stack([r[name] for r in rows]).astype(np.float64)
            return {
                "min": a.min(0).tolist(),
                "max": a.max(0).tolist(),
                "mean": a.mean(0).tolist(),
                "std": a.std(0).tolist(),
                "count": [len(a)],
            }

        pixels = np.stack(frames).astype(np.float64) / 255.0  # [N, H, W, C]
        per_channel = pixels.transpose(3, 0, 1, 2).reshape(pixels.shape[3], -1)
        image_stats = {
            key: value.reshape(-1, 1, 1).tolist()
            for key, value in (
                ("min", per_channel.min(1)),
                ("max", per_channel.max(1)),
                ("mean", per_channel.mean(1)),
                ("std", per_channel.std(1)),
            )
        }
        image_stats["count"] = [len(frames)]
        return {
            f"observation.images.{self.config.camera}": image_stats,
            "observation.state": vec("observation.state"),
            "action": vec("action"),
            "ball_xy": vec("ball_xy"),
        }

    def _features(self, state_dim: int) -> dict:
        cam = self.config.camera
        height, width = self._frame_shape
        names = self.config.state_names or self._state_names(state_dim)
        return {
            f"observation.images.{cam}": {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": height,
                    "video.width": width,
                    "video.channels": 3,
                    "video.codec": self.config.video_codec,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": self._fps,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": names,
            },
            "action": {
                "dtype": "float32",
                "shape": [len(self._action_names())],
                # Where to put the torso and by when, in the robot's own
                # frame: x forward, y left, seconds. Then, with the eye on,
                # where to look for the next frame.
                "names": self._action_names(),
            },
            "ball_visible": {"dtype": "bool", "shape": [1], "names": None},
            # What it would say: an index into meta/responses.jsonl.
            "response_index": {"dtype": "int64", "shape": [1], "names": None},
            **({
                # DAgger: what the driving VLA itself last said (-1: nothing
                # yet) and last commanded -- its inputs, not the labels.
                "input_prev_response_index": {"dtype": "int64", "shape": [1], "names": None},
                "input_prev_heading": {"dtype": "float32", "shape": [1], "names": None},
            } if getattr(self.env.control_manager.components.get(self.config.goal_component),
                         "_vla_inputs_now", None) is not None else {}),
            "ball_xy": {
                "dtype": "float32",
                "shape": [2],
                "names": ["ball_x", "ball_y"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }

    def _colors(self) -> List[str]:
        ball = self.env.control_manager.components.get(self.config.ball_component)
        source = getattr(ball, "command_source", None)
        return list(getattr(getattr(source, "config", None), "colors", None) or ["red"])

    @staticmethod
    def _expand(table: Dict[str, List[str]], colors: List[str]) -> List[str]:
        """Every template filled with every colour, deduplicated, in order."""
        out: List[str] = []
        for kind in table:
            for text in table.get(kind, []):
                for c in colors:
                    t = text.format(color=c)
                    if t not in out:
                        out.append(t)
        return out

    def _tasks(self) -> List[str]:
        """Prompts by task_index."""
        return self._expand(self.config.prompts, self._colors())

    def _responses(self) -> List[str]:
        """Answer texts by response_index."""
        return self._expand(self.config.responses, self._colors())

    def _language(self, goal):
        """(task_index, response_index) per env for this frame."""
        ball = self.env.control_manager.components.get(self.config.ball_component)
        texts = language_texts(
            goal, getattr(ball, "command_source", None), self.config.prompts,
            self.config.responses, self._colors(), self.env.num_envs, self.env.device,
        )
        tasks = {t: i for i, t in enumerate(self._tasks())}
        responses = {t: i for i, t in enumerate(self._responses())}
        task_index = np.array([tasks[p] for p, _ in texts], dtype=np.int64)
        response_index = np.array([responses[r] for _, r in texts], dtype=np.int64)
        return task_index, response_index

    def _action_names(self) -> List[str]:
        goal = self.env.control_manager.components[self.config.goal_component]
        names = masked_target_names(goal)
        if self.config.eye_component:
            names += ["gaze_u", "gaze_v", "gaze_zoom"]
        return names

    def _state_names(self, state_dim: int) -> List[str]:
        """Label the state vector, so the columns are readable a year later."""
        names = [f"gravity_{a}" for a in "xyz"]
        names += [f"lin_vel_{a}" for a in "xyz"]
        names += [f"ang_vel_{a}" for a in "xyz"]
        dofs = list(self.env.robot_config.kinematic_info.dof_names)
        names += [f"{d}.pos" for d in dofs] + [f"{d}.vel" for d in dofs]
        names += [f"{b}.load" for b in (self.env.robot_config.contact_bodies or [])]
        if self.config.eye_component:
            names += ["gaze_u", "gaze_v", "gaze_zoom"]
        if len(names) != state_dim:
            return [f"s{i}" for i in range(state_dim)]
        return names

    def _write_meta(self) -> None:
        meta = os.path.join(self._root, "meta")
        os.makedirs(meta, exist_ok=True)
        state_dim = len(self._episodes_stats[0]["stats"]["observation.state"]["mean"])
        info = {
            "codebase_version": _CODEBASE_VERSION,
            "robot_type": self.config.robot_type,
            "total_episodes": self._episode_index,
            "total_frames": self._frame_total,
            "total_tasks": len(self._tasks()),
            "total_videos": self._episode_index,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self._fps,
            "splits": {"train": f"0:{self._episode_index}"},
            "data_path": (
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
            ),
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            ),
            "features": self._features(state_dim),
        }
        with open(os.path.join(meta, "info.json"), "w") as f:
            json.dump(info, f, indent=4)
        with open(os.path.join(meta, "tasks.jsonl"), "w") as f:
            for i, task in enumerate(self._tasks()):
                f.write(json.dumps({"task_index": i, "task": task}) + "\n")
        with open(os.path.join(meta, "responses.jsonl"), "w") as f:
            for i, text in enumerate(self._responses()):
                f.write(json.dumps({"response_index": i, "response": text}) + "\n")
        with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
            for row in self._episodes_meta:
                f.write(json.dumps(row) + "\n")
        with open(os.path.join(meta, "episodes_stats.jsonl"), "w") as f:
            for row in self._episodes_stats:
                f.write(json.dumps(row) + "\n")

    @property
    def _frame_shape(self):
        eye = self._eye()
        if eye is not None:
            return (eye.config.out_res, eye.config.out_res)
        cameras = getattr(self.env.simulator.config, "onboard_cameras", {}) or {}
        cam = cameras.get(self.config.camera)
        if cam is None:
            return (0, 0)
        return (cam.height, cam.width)
