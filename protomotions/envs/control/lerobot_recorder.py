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

* **The action does not include the heading.** The demonstrator always faces
  what it is running at, so the commanded heading is atan2(dy, dx) of the
  position it was already given -- recording it would be asking the student to
  learn a copy of its own output. Three numbers: dx, dy, and how long it has.

* **Privileged columns are labels, never inputs.** ``ball_visible`` and
  ``ball_xy`` are written because they make good auxiliary supervision and an
  honest evaluation ("did it know where the ball was, or did it get lucky"),
  but nothing the student is fed at inference may come from them.

Run it headless. With a viewer open, every visualization marker renders into
the camera too, including the conditioned-target ladder -- which is the label
drawn on top of the image (see VisualizationMarkerConfig.camera_visible).

Format is LeRobot v2.1: one parquet and one mp4 per episode, with
meta/{info,episodes,episodes_stats,tasks}. The installed lerobot writes v3.0
and ships ``convert_dataset_v21_to_v30.py`` to bring this forward; v2.1 is
also what the Isaac-GR00T fine-tuning tutorial expects.
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
    # Which control component holds the demonstrator, and which holds the ball.
    goal_component: str = "masked_mimic"
    ball_component: str = "ball"
    camera: str = "front_camera"
    task: str = "chase the red ball"
    # Written even while it is the only task: a constant prompt costs one
    # column and is what makes the dataset usable later, when there are
    # several objects and the prompt has to pick one.
    robot_type: str = "go2"
    video_codec: str = "libx264"
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

    def step(self) -> None:
        if self._done:
            return
        for env_id in self._reset_pending.nonzero(as_tuple=False).flatten().tolist():
            self._flush(env_id, reason="reset")
        self._reset_pending[:] = False

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

        state = self._state()
        action, extras = self._action()
        frames = frame_batch.detach().cpu().numpy()
        if frames.shape[-1] == 4:
            frames = frames[..., :3]

        for env_id in range(self.env.num_envs):
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
        return torch.cat(parts, dim=-1).detach().cpu().numpy().astype(np.float32)

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

        target = to_body(goal._target_xy())
        remaining = (goal._deadline - goal._now()).clamp(
            goal.config.min_horizon_sec, goal.config.max_horizon_sec
        )
        action = torch.cat([target, remaining.unsqueeze(-1)], dim=-1)

        ball_xy = to_body(goal._goal_xy())
        visible = (
            goal._ball_seen
            if goal._unprivileged()
            else torch.ones_like(goal._deadline, dtype=torch.bool)
        )
        extras = {
            "visible": visible.detach().cpu().numpy(),
            "ball_xy": ball_xy.detach().cpu().numpy().astype(np.float32),
        }
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
        # of, and it still costs an episode index.
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
                "tasks": [self.config.task],
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
                f"{self._root}",
                flush=True,
            )

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
                "task_index": pa.array([0] * n, type=pa.int64()),
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
                "shape": [3],
                # Where to put the torso and by when, in the robot's own
                # frame: x forward, y left, seconds.
                "names": ["target_x", "target_y", "seconds_remaining"],
            },
            "ball_visible": {"dtype": "bool", "shape": [1], "names": None},
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

    def _state_names(self, state_dim: int) -> List[str]:
        """Label the state vector, so the columns are readable a year later."""
        names = [f"gravity_{a}" for a in "xyz"]
        names += [f"lin_vel_{a}" for a in "xyz"]
        names += [f"ang_vel_{a}" for a in "xyz"]
        dofs = list(self.env.robot_config.kinematic_info.dof_names)
        names += [f"{d}.pos" for d in dofs] + [f"{d}.vel" for d in dofs]
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
            "total_tasks": 1,
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
            f.write(json.dumps({"task_index": 0, "task": self.config.task}) + "\n")
        with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
            for row in self._episodes_meta:
                f.write(json.dumps(row) + "\n")
        with open(os.path.join(meta, "episodes_stats.jsonl"), "w") as f:
            for row in self._episodes_stats:
                f.write(json.dumps(row) + "\n")

    @property
    def _frame_shape(self):
        cameras = getattr(self.env.simulator.config, "onboard_cameras", {}) or {}
        cam = cameras.get(self.config.camera)
        if cam is None:
            return (0, 0)
        return (cam.height, cam.width)
