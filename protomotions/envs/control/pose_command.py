# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Hold a mocap pose: "be in THIS pose, here, by then", straight from a clip.

MaskedMimic's native request is a set of body poses at future times. A pose
command takes one frame of a clip in the training corpus -- sitting, sitting
with a paw up -- re-anchors it at the robot's current spot and heading, and
conditions EVERY tracked body on it (position and rotation) with a deadline.
No new clips, no retraining: if the corpus has the frame, the prior knows the
pose.

A standalone test runs keyframes in sequence -- e.g. sit, then sit with the
right paw up (reaching a paw-up from a sit is a smaller ask than from a
stand) -- holds each, releases, and repeats, printing how close it got.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor

from protomotions.envs.control.ball_chase import (
    MaskedMimicGoalControl,
    MaskedMimicGoalControlConfig,
)
from protomotions.utils import rotations


def _up(n: int, device) -> Tensor:
    up = torch.zeros(n, 3, device=device)
    up[:, 2] = 1.0
    return up


@dataclass
class PoseTestControlConfig(MaskedMimicGoalControlConfig):
    """Reach and hold a sequence of mocap keyframes, then rest, forever."""

    _target_: str = "protomotions.envs.control.pose_command.PoseTestControl"

    # "clip:seconds" -- a clip name as it appears in the motion file list
    # (without directory or ".motion") and a time in it. Held in order.
    keyframes: List[str] = field(
        default_factory=lambda: ["25_clip_1:8.5", "25_clip_1:0.62"]
    )
    reach_sec: float = 1.5
    hold_sec: float = 3.0
    rest_sec: float = 2.0
    # Condition only these bodies (plus the root) -- e.g. the four feet, to
    # say "this paw up, that one down" without dictating every joint. None:
    # every tracked body.
    bodies: Optional[List[str]] = None
    # Condition every body's ROTATION too, or only the root's. Positions for
    # the bodies plus the root's orientation is the smaller ask -- and the
    # smaller action for a policy that has to produce it.
    all_rotations: bool = True


class PoseTestControl(MaskedMimicGoalControl):
    config: PoseTestControlConfig

    def __init__(self, config: PoseTestControlConfig, env):
        super().__init__(config, env)
        n, dev = env.num_envs, env.device
        self._poses: Optional[list] = None
        self._anchor_xy = torch.zeros(n, 2, device=dev)
        self._anchor_heading = torch.zeros(n, device=dev)
        self._cycle_start = None
        self._reported = set()
        self._err_sum = None
        self._err_n = 0
        self._base_mask = None

    # ------------------------------------------------------------------
    # Keyframes
    # ------------------------------------------------------------------

    def _stages(self) -> list:
        """(spec, reach, hold) per stage. "a>b" is a transition a, then b.

        Going straight to a pose far from where the dog is (standing to
        lying) can be a jump the prior will not make; the clip's own
        in-between frame first shows it the way. The reach time is split
        across the chain and only the last one is held.
        """
        stages = []
        for spec in self.config.keyframes:
            chain = [c.strip() for c in spec.split(">") if c.strip()]
            for i, c in enumerate(chain):
                last = i == len(chain) - 1
                stages.append((c, self.config.reach_sec / len(chain),
                               self.config.hold_sec if last else 0.0))
        return stages

    def _load_poses(self) -> list:
        """Each keyframe as body poses relative to its own root and heading."""
        lib = self.env.motion_lib
        files = [str(f) for f in lib.motion_files]
        dev = self.env.device
        poses = []
        self._stage_times = [(r, h) for _, r, h in self._stages()]
        for spec, _, _ in self._stages():
            clip, _, t = spec.rpartition(":")
            matches = [i for i, f in enumerate(files) if f.split("/")[-1] == f"{clip}.motion"]
            if not matches:
                raise ValueError(f"[pose-test] no clip '{clip}' in the motion library")
            mid = torch.tensor([matches[0]], device=dev)
            state = lib.get_motion_state(mid, torch.tensor([float(t)], device=dev))
            pos, rot = state.rigid_body_pos[0], state.rigid_body_rot[0]
            heading = rotations.calc_heading(rot[:1], True)
            inv = rotations.quat_from_angle_axis(-heading, _up(1, dev), True)
            b = pos.shape[0]
            rel = pos.clone()
            rel[:, :2] -= pos[0, :2]
            rel_pos = rotations.quat_rotate(inv.expand(b, 4), rel, True)
            rel_rot = rotations.quat_mul(inv.expand(b, 4), rot, True)
            fwd = rotations.quat_rotate(rot[:1], torch.tensor([[1.0, 0.0, 0.0]], device=dev), True)[0]
            pitch = math.degrees(math.atan2(float(fwd[2]), float(torch.linalg.norm(fwd[:2]))))
            poses.append({"spec": spec, "pos": rel_pos, "rot": rel_rot, "pitch": pitch})
            print(f"[pose-test] keyframe {spec}: trunk {float(pos[0, 2]):.2f} m, "
                  f"nose-up {pitch:+.0f} deg", flush=True)
        return poses

    # ------------------------------------------------------------------
    # Schedule: [reach, hold] per keyframe, then rest
    # ------------------------------------------------------------------

    def _phase(self):
        """(stage or None when resting, stage deadline, cycle, time into stage)."""
        now = float(self._now()[0])
        if self._cycle_start is None:
            self._cycle_start = now
        times = getattr(self, "_stage_times", None) or [
            (r, h) for _, r, h in self._stages()
        ]
        cycle_len = sum(r + h for r, h in times) + self.config.rest_sec
        t = now - self._cycle_start
        cycle = int(t // cycle_len)
        start = self._cycle_start + cycle * cycle_len
        t -= cycle * cycle_len
        for stage, (reach, hold) in enumerate(times):
            if t < reach + hold:
                return stage, start + reach, cycle, t
            t -= reach + hold
            start += reach + hold
        return None, now, cycle, t

    def step(self) -> None:
        if self._poses is None:
            self._poses = self._load_poses()
        stage, _, cycle, into = self._phase()
        root = self.env.simulator.get_root_state()
        if stage == 0 and into < self.env.dt * 1.5:
            # A new cycle: the pose goes where the dog is now, facing its way.
            self._anchor_xy = root.root_pos[:, :2].clone()
            self._anchor_heading = rotations.calc_heading(root.root_rot, True)
        super().step()
        self._score(stage, cycle, into)

    # The torso target is the anchor: stay here, face this way.
    def _target_xy(self) -> Tensor:
        return self._anchor_xy

    def _holding(self) -> Tensor:
        return torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)

    def _command_heading(self, root_pos: Tensor) -> Tensor:
        return self._anchor_heading

    def _lead_times(self) -> Tensor:
        stage, deadline, _, _ = self._phase()
        remaining = torch.full(
            (self.env.num_envs,), deadline - float(self._now()[0]), device=self.env.device
        ).clamp(self.config.min_horizon_sec, self.config.max_horizon_sec)
        return remaining.unsqueeze(-1) * self._fractions().unsqueeze(0)

    def _pose_mask(self) -> Tensor:
        if getattr(self, "_pose_mask_cache", None) is not None:
            return self._pose_mask_cache
        n = self.env.num_envs
        steps = self.config.num_masked_future_steps
        k = self.num_conditionable_bodies
        mask = torch.zeros(n, steps, k, 2, dtype=torch.bool, device=self.env.device)
        if self.config.bodies is None:
            mask[:] = True
        else:
            names = list(self.env.robot_config.kinematic_info.body_names)
            wanted = {names.index(b) for b in self.config.bodies} | {self._root_body_id}
            cond = self.conditionable_body_ids.tolist()
            for body in wanted:
                if body in cond:
                    mask[:, :, cond.index(body), :] = True

        if not self.config.all_rotations:
            # Positions for the chosen bodies, orientation for the root only.
            root = self.conditionable_body_ids.tolist().index(self._root_body_id)
            mask[:, :, :, 1] = False
            mask[:, :, root, 1] = True
        if __import__("os").environ.get("PROTOMOTIONS_POSE_NO_ROOT_POS"):
            # Orientation of the trunk only, not where it is.
            root = self.conditionable_body_ids.tolist().index(self._root_body_id)
            mask[:, :, root, 0] = False
        self._pose_mask_cache = mask.view(n, -1)
        return self._pose_mask_cache

    def _world_pose(self, pose):
        """The keyframe placed at each env's anchor: [N, B, 3], [N, B, 4]."""
        n, dev = self.env.num_envs, self.env.device
        b = pose["pos"].shape[0]
        spin = rotations.quat_from_angle_axis(self._anchor_heading, _up(n, dev), True)
        spin_b = spin.unsqueeze(1).expand(n, b, 4).reshape(-1, 4)
        pos = rotations.quat_rotate(spin_b, pose["pos"].unsqueeze(0).expand(n, b, 3).reshape(-1, 3), True)
        pos = pos.view(n, b, 3)
        pos[..., :2] += self._anchor_xy.unsqueeze(1)
        ground = self.env.terrain.get_ground_heights(self._anchor_xy).view(n, 1)
        pos[..., 2] += ground
        rot = rotations.quat_mul(spin_b, pose["rot"].unsqueeze(0).expand(n, b, 4).reshape(-1, 4), True)
        return pos, rot.view(n, b, 4)

    def populate_context(self, ctx) -> None:
        # The env builds its first observations before the first step.
        if self._poses is None:
            self._poses = self._load_poses()
        stage, _, _, _ = self._phase()
        if self._base_mask is None:
            self._base_mask = self.masked_mimic_target_bodies_masks.clone()
        if stage is None:
            self.masked_mimic_target_bodies_masks[:] = self._base_mask
            super().populate_context(ctx)
            return
        # The chosen bodies (default: all), position and rotation, at every
        # future step.
        self.masked_mimic_target_bodies_masks[:] = self._pose_mask()
        super().populate_context(ctx)
        pos, rot = self._world_pose(self._poses[stage])
        steps = self.config.num_masked_future_steps
        ctx.masked_mimic.ref_pos = pos.unsqueeze(1).expand(-1, steps, -1, -1).contiguous()
        ctx.masked_mimic.ref_rot = rot.unsqueeze(1).expand(-1, steps, -1, -1).contiguous()

    # ------------------------------------------------------------------

    def _score(self, stage, cycle, into) -> None:
        """At the end of each hold: how far every body is from the keyframe."""
        if stage is None:
            return
        reach, hold = self._stage_times[stage]
        if hold <= 0 or into < reach:
            return
        bodies = self.env.simulator.get_bodies_state()
        target, _ = self._world_pose(self._poses[stage])
        err = torch.linalg.norm(bodies.rigid_body_pos - target, dim=-1)  # [N, B]
        self._err_sum = err.mean(0) if self._err_sum is None else self._err_sum + err.mean(0)
        self._err_n += 1
        stage_end = reach + hold
        key = (cycle, stage)
        if into < stage_end - self.env.dt * 1.5 or key in self._reported:
            return
        self._reported.add(key)
        dump = __import__("os").environ.get("PROTOMOTIONS_POSE_DUMP")
        if dump:
            # Achieved vs target body positions (env 0), for a side-by-side look.
            import json
            with open(dump, "a") as f:
                f.write(json.dumps({
                    "cycle": cycle, "spec": self._poses[stage]["spec"],
                    "achieved": bodies.rigid_body_pos[0].tolist(),
                    "target": target[0].tolist(),
                    "root_xy": self._anchor_xy[0].tolist(),
                    "heading": float(self._anchor_heading[0]),
                }) + "\n")
        mean = self._err_sum / max(self._err_n, 1)
        self._err_sum, self._err_n = None, 0
        names = list(self.env.robot_config.kinematic_info.body_names)
        root = self.env.simulator.get_root_state()
        fwd = rotations.quat_rotate(root.root_rot[:1], torch.tensor([[1.0, 0.0, 0.0]], device=self.env.device), True)[0]
        pitch = math.degrees(math.atan2(float(fwd[2]), float(torch.linalg.norm(fwd[:2]))))
        left = rotations.quat_rotate(root.root_rot[:1], torch.tensor([[0.0, 1.0, 0.0]], device=self.env.device), True)[0]
        roll = math.degrees(math.asin(max(-1.0, min(1.0, float(left[2])))))
        z = bodies.rigid_body_pos[0, :, 2]
        feet = "  ".join(f"{n[:2]} {float(z[names.index(n)]):.2f}"
                         for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot") if n in names)
        worst = sorted(zip(mean.tolist(), names), reverse=True)[:3]
        print(
            f"[pose-test] cycle {cycle} {self._poses[stage]['spec']}: mean body error "
            f"{float(mean.mean()) * 100:.1f} cm (worst: "
            + ", ".join(f"{n} {e * 100:.0f} cm" for e, n in worst)
            + f"); trunk {float(root.root_pos[0, 2]):.2f} m, roll {roll:+.0f}, nose-up {pitch:+.0f} deg "
            f"(keyframe {self._poses[stage]['pitch']:+.0f}); feet z {feet}",
            flush=True,
        )
