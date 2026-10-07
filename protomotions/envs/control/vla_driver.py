# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Let a trained VLA drive the chase, and score it against the demonstrator.

The demonstrator (MaskedMimicGoalControl) still runs underneath -- it is the
referee: it knows where the ball is, whether it is in the eye's window, what
the right answer is right now, and what conditioning IT would send (the
DAgger label, _label_mm). What it no longer does is DRIVE. At the student's
rate (10 Hz) each dog's eye crop, request and self-knowledge go to the VLA
server (scripts/vla_server.py), and its reply IS the dog's MaskedMimic
conditioning (lerobot_recorder.masked_targets, inverted) and the eye's next
gaze:

    per body   position, orientation  -> ref_pos, ref_rot (robot frame -> world)
    per body   on / rot_on            -> target_bodies_masks
    seconds    lead time              -> the targets' deadline, counting down
    gaze       u, v, zoom             -> CameraEye.set_external

Between replies the world targets stand still and the deadline counts down,
as the demonstrator's do. What goes over is only what the robot would have:
the 224x224 crop, the request and how long ago it was made, the VLA's OWN
previous answer and heading, how far the body has turned since the throw (gyro), which side the weight is on
(foot loads), and where the eye is pointed. Nothing privileged.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor

from protomotions.envs.context_views import MaskedMimicContext
from protomotions.envs.control.ball_chase import (
    ANSWER_NO,
    ANSWER_NONE,
    ANSWER_YES,
    MaskedMimicGoalControl,
    MaskedMimicGoalControlConfig,
)
from protomotions.utils import rotations

AUTHKEY = b"protomotions-vla"
NOTHING_SAID = "nothing yet"


def answer_kind(text: str) -> int:
    """ANSWER_* that a sentence commits to (as the training script grades it)."""
    t = (text or "").lower()
    if t.startswith("no") or "can't find" in t:
        return ANSWER_NO
    # "getting the green ball", "I'm sitting": found it / done it.
    if t.startswith("yes") or t.startswith("getting the") or t.startswith("i'm"):
        return ANSWER_YES
    return ANSWER_NONE



def targets_to_world(act: Tensor, root_pos: Tensor, root_rot: Tensor, terrain):
    """Invert lerobot_recorder.masked_targets: robot-frame targets -> world.

    act is [N, 9B + 2B + 1] over the conditionable bodies (position x y z,
    orientation fwd xyz up xyz; then on, rot_on per body; then seconds).
    Returns world positions [N, B, 3], orientations [N, B, 4] (xyzw), the
    bits [N, B, 2] as bools, and seconds [N].
    """
    n = act.shape[0]
    b = (act.shape[1] - 1) // 11
    poses = act[:, : 9 * b].view(n, b, 9)
    bits = act[:, 9 * b : 11 * b].view(n, b, 2) > 0.5
    spin = rotations.calc_heading_quat(root_rot, True)
    spin_b = spin.unsqueeze(1).expand(n, b, 4).reshape(-1, 4)
    flat = poses[..., :3].clone()
    flat[..., 2] = 0.0
    pos = rotations.quat_rotate(spin_b, flat.reshape(-1, 3), True).view(n, b, 3)
    pos[..., :2] += root_pos[:, :2].unsqueeze(1)
    ground = terrain.get_ground_heights(pos[..., :2].reshape(-1, 2)).view(n, b)
    pos[..., 2] = ground + poses[..., 2]
    local = rotations.tan_norm_to_quat(poses[..., 3:9].reshape(-1, 6), True)
    rot = rotations.quat_mul(spin_b, local, True).view(n, b, 4)
    # An orientation it does not ask for is only a placeholder (zeros are
    # not even a rotation): identity, masked out.
    ident = torch.zeros_like(rot)
    ident[..., 3] = 1.0
    rot = torch.where(bits[..., 1:2] & torch.isfinite(rot).all(-1, keepdim=True), rot, ident)
    return pos, rot, bits, act[:, 11 * b]

@dataclass
class VlaGoalControlConfig(MaskedMimicGoalControlConfig):
    """The chase, driven by a VLA server instead of the demonstrator."""

    _target_: str = "protomotions.envs.control.vla_driver.VlaGoalControl"

    server: str = "localhost:6010"
    # One request for every throw ("is there a green ball?", "sit"), or None
    # for the recorder's mix of requests, one phrasing per throw.
    prompt: Optional[str] = None
    # The VLA runs at this rate (what it was trained on).
    vla_hz_rate: float = 10.0
    # Generate the answer text every Nth call; actions come every call.
    answer_every: int = 3
    camera: str = "front_camera"
    report_every: int = 250


class VlaGoalControl(MaskedMimicGoalControl):
    config: VlaGoalControlConfig

    def __init__(self, config: VlaGoalControlConfig, env):
        super().__init__(config, env)
        n, dev = env.num_envs, env.device
        # The VLA's last conditioning, in the world: [N, B, 3], [N, B, 4],
        # [N, B, 2] over the conditionable bodies, and when it is due.
        self._vla_pos: Optional[Tensor] = None
        self._vla_rot: Optional[Tensor] = None
        self._vla_bits: Optional[Tensor] = None
        self._vla_deadline = torch.zeros(n, device=dev)
        self._vla_prev: List[str] = [NOTHING_SAID] * n
        self._vla_turned = torch.zeros(n, device=dev)
        # When this throw's request was made: the model is told how long ago.
        self._vla_asked_at = torch.zeros(n, device=dev)
        # Its own last commanded heading (egocentric), fed back like the answer.
        self._vla_prev_heading = torch.zeros(n, device=dev)
        self._vla_skip = False
        self._vla_throw = torch.full((n,), -1, dtype=torch.long, device=dev)
        self._vla_every = max(int(round(1.0 / (config.vla_hz_rate * env.dt))), 1)
        self._vla_ticks = 0
        self._vla_calls = 0
        self._vla_started = False
        self._conn = None
        self._agree = self._judged = 0
        self._act_ms: List[float] = []
        self._ans_ms: List[float] = []
        self._rtt_ms: List[float] = []

    # ------------------------------------------------------------------
    # The MaskedMimic conditioning -- the VLA's reply
    # ------------------------------------------------------------------

    def populate_context(self, ctx) -> None:
        """The demonstrator's conditioning is the label; the VLA's is sent."""
        super().populate_context(ctx)
        label = ctx.masked_mimic
        # A snapshot: the demonstrator rewrites its mask in place next step.
        self._label_mm = MaskedMimicContext(
            mimic=label.mimic, ref_pos=label.ref_pos.clone(), ref_rot=label.ref_rot.clone(),
            target_times=label.target_times, time_offsets=label.time_offsets.clone(),
            target_poses_masks=label.target_poses_masks,
            target_bodies_masks=label.target_bodies_masks.clone(),
        )
        if self._vla_pos is None:
            return  # nothing asked yet: the demonstrator's first step stands
        n, steps = label.ref_pos.shape[0], label.ref_pos.shape[1]
        cond = self.conditionable_body_ids
        ref_pos, ref_rot = label.ref_pos.clone(), label.ref_rot.clone()
        ref_pos[:, :, cond] = self._vla_pos.unsqueeze(1).expand(-1, steps, -1, -1)
        ref_rot[:, :, cond] = self._vla_rot.unsqueeze(1).expand(-1, steps, -1, -1)
        masks = self._vla_bits.unsqueeze(1).expand(-1, steps, -1, -1).reshape(n, -1)
        remaining = (self._vla_deadline - self._now()).clamp(
            self.config.min_horizon_sec, self.config.max_horizon_sec
        )
        ctx.masked_mimic = MaskedMimicContext(
            mimic=label.mimic, ref_pos=ref_pos, ref_rot=ref_rot,
            target_times=label.target_times,
            time_offsets=remaining.unsqueeze(-1) * self._fractions().unsqueeze(0),
            target_poses_masks=label.target_poses_masks,
            target_bodies_masks=masks.contiguous(),
        )
        # The viewer's target spheres show where the VLA sends the trunk.
        self._marker_target_pos = ref_pos[:, :, self._root_body_id]

    def _apply(self, act: Tensor) -> None:
        """A reply ([N, 9B + 2B + 1], robot frame) as the world targets to send."""
        root = self.env.simulator.get_root_state()
        pos, rot, bits, seconds = targets_to_world(
            act, root.root_pos, root.root_rot, self.env.terrain
        )
        self._vla_pos, self._vla_rot, self._vla_bits = pos, rot, bits
        self._vla_deadline = self._now() + seconds.clamp_min(self.config.min_horizon_sec)
        # base_link is the first conditionable body: its forward vector.
        fx, fy = act[:, 3], act[:, 4]
        self._vla_prev_heading = torch.where(
            bits[:, 0, 1], torch.atan2(fy, fx), torch.zeros_like(fx)
        )

    # ------------------------------------------------------------------

    def _connect(self):
        from multiprocessing.connection import Client

        host, _, port = self.config.server.rpartition(":")
        print(f"[vla] connecting to {host}:{port} ...", flush=True)
        self._conn = Client((host or "localhost", int(port)), authkey=AUTHKEY)
        print("[vla] connected", flush=True)

    def _start(self) -> None:
        """First step: pin the typed request onto the throws, then re-throw."""
        self._vla_started = True
        source = self._ball_source()
        prompt = self.config.prompt
        if prompt:
            from protomotions.envs.control.lerobot_recorder import LeRobotRecorderConfig

            phrasings = LeRobotRecorderConfig().prompts
            poses = list(getattr(source.config, "poses", []))
            said = prompt.strip().lower()
            pose = next((i for i, name in enumerate(poses)
                         if said in phrasings.get(f"pose_{name}", [])), -1)
            source.force_pose = pose
            if pose >= 0:
                print(f"[vla] every throw: {prompt!r} (pose: {poses[pose]})", flush=True)
            else:
                colors = list(getattr(source.config, "colors", ["red"]))
                named = [i for i, c in enumerate(colors) if c in said]
                source.force_look = said.endswith("?")
                source.force_color = named[0] if named else -1
                kind = "question" if source.force_look else "fetch"
                what = colors[named[0]] if named else "any colour"
                print(f"[vla] every throw: {prompt!r} ({kind}, {what})", flush=True)
        source._set_random_target(torch.arange(self.env.num_envs, device=self.env.device))

    def _prompts(self) -> List[str]:
        """What each dog is asked this throw, worded as the recordings word it."""
        n = self.env.num_envs
        if self.config.prompt:
            return [self.config.prompt] * n
        from protomotions.envs.control.lerobot_recorder import (
            LeRobotRecorderConfig, language_texts,
        )

        table = LeRobotRecorderConfig()
        source = self._ball_source()
        colors = list(getattr(source.config, "colors", None) or ["red"])
        texts = language_texts(self, source, table.prompts, table.responses, colors, n, self.env.device)
        return [prompt for prompt, _ in texts]

    def step(self) -> None:
        if not self._vla_started:
            self._start()
        source = self._ball_source()
        # A new throw: the dog has said nothing and turned nowhere yet.
        fresh = (source.throw_id != self._vla_throw).nonzero(as_tuple=False).flatten().tolist()
        root = self.env.simulator.get_root_state()
        for e in fresh:
            self._vla_prev[e] = NOTHING_SAID
        if fresh:
            ids = torch.tensor(fresh, device=self.env.device)
            self._vla_turned[ids] = 0.0
            self._vla_asked_at[ids] = self._now()[ids]
            self._vla_throw[ids] = source.throw_id[ids]
            self._vla_prev_heading[ids] = 0.0
            # The frame in hand was rendered before the throw: it shows the
            # old ball. Ask on the next one.
            self._vla_skip = True
        yaw_rate = rotations.quat_rotate_inverse(root.root_rot, root.root_ang_vel, True)[:, 2]
        self._vla_turned += yaw_rate.abs() * self.env.dt
        # What the VLA would be handed for this frame -- its own last answer
        # and heading. The recorder writes these as the INPUTS of a DAgger row.
        self._vla_inputs_now = (list(self._vla_prev), self._vla_prev_heading.clone())

        if self._vla_skip:
            self._vla_skip = False
        else:
            if self._vla_ticks % self._vla_every == 0:
                self._ask()
            self._vla_ticks += 1
        super().step()
        self._score()

    def _ask(self) -> None:
        eye = (
            self.env.control_manager.components[self.config.eye_component]
            if self.config.eye_component else None
        )
        frames = self.env.simulator.get_camera_images().get(self.config.camera)
        if frames is None or eye is None:
            return
        if self._conn is None:
            self._connect()
        crops = eye.crop(frames)[..., :3].detach().cpu().numpy()
        gaze = eye.gaze().tolist()
        lean = self._lean_sign().tolist()
        prompts = self._prompts()
        elapsed = (self._now() - self._vla_asked_at).tolist()
        want_answer = self._vla_calls % max(self.config.answer_every, 1) == 0
        request = {
            "want_answer": want_answer,
            "envs": [
                {
                    "image": crops[e].tobytes(), "shape": crops[e].shape,
                    "prompt": prompts[e], "prev": self._vla_prev[e],
                    "turned": float(self._vla_turned[e]),
                    "elapsed": float(elapsed[e]),
                    "prev_heading": float(self._vla_prev_heading[e]),
                    "lean": "left" if lean[e] > 0 else "right",
                    "gaze": gaze[e],
                }
                for e in range(self.env.num_envs)
            ],
        }
        t0 = time.time()
        self._conn.send(request)
        reply = self._conn.recv()
        self._rtt_ms.append((time.time() - t0) * 1000.0)
        self._act_ms.append(reply["action_ms"])
        if want_answer:
            self._ans_ms.append(reply["answer_ms"])
        self._vla_calls += 1

        act = torch.tensor(reply["actions"], device=self.env.device, dtype=torch.float32)
        from protomotions.envs.control.lerobot_recorder import masked_target_names

        expected = len(masked_target_names(self)) + 3
        if act.shape[-1] != expected:
            raise RuntimeError(
                f"[vla] the model answers {act.shape[-1]} numbers, this robot's conditioning "
                f"plus gaze is {expected}: a model trained on an older recording"
            )
        self._apply(act[:, :-3])
        eye.set_external(act[:, -3:])
        if os.environ.get("PROTOMOTIONS_VLA_DEBUG"):
            root = self.env.simulator.get_root_state()
            heading = rotations.calc_heading(root.root_rot, True)
            c, s_ = torch.cos(heading[0]), torch.sin(heading[0])
            ball = self._ball_source().control._tar_pos[0, :2] - root.root_pos[0, :2]
            bx, by = float(c * ball[0] + s_ * ball[1]), float(-s_ * ball[0] + c * ball[1])
            a = act[0].tolist()
            b = len(self.conditionable_body_ids)
            legs_on = int(sum(x > 0.5 for x in a[9 * b + 2 : 11 * b : 2]))
            print(
                f"[vla-debug] prompt={prompts[0]!r} prev={self._vla_prev[0]!r} "
                f"turned={math.degrees(float(self._vla_turned[0])):.0f} | trunk x {a[0]:+.2f} y {a[1]:+.2f} "
                f"z {a[2]:.2f} h {math.degrees(float(self._vla_prev_heading[0])):+.0f} "
                f"t {a[11 * b]:.2f} legs on {legs_on} gaze {a[-3]:+.2f} {a[-2]:+.2f} {a[-1]:.1f} "
                f"| ball body ({bx:+.2f}, {by:+.2f}) in_view={bool(self._ball_in_view[0])} "
                f"truth_answer={int(self._answer[0])} | label {self._debug_label()}",
                flush=True,
            )

        for e, said in enumerate(reply["answers"]):
            if said is None:
                continue
            if said != self._vla_prev[e]:
                print(f"[vla] dog {e}: {prompts[e]!r} -> {said!r}", flush=True)
            self._vla_prev[e] = said

    def _debug_label(self) -> str:
        """The demonstrator's conditioning for dog 0, as the VLA debug line shows its own."""
        from protomotions.envs.control.lerobot_recorder import masked_targets

        if getattr(self, "_label_mm", None) is None:
            return "-"
        root = self.env.simulator.get_root_state()
        a = masked_targets(self._label_mm, self, root, self.env.terrain)[0].tolist()
        b = len(self.conditionable_body_ids)
        legs_on = int(sum(x > 0.5 for x in a[9 * b + 2 : 11 * b : 2]))
        return (f"trunk x {a[0]:+.2f} y {a[1]:+.2f} z {a[2]:.2f} "
                f"h {math.degrees(math.atan2(a[4], a[3])):+.0f} t {a[11 * b]:.2f} legs on {legs_on}")

    def _score(self) -> None:
        """Does what the VLA last said match what the demonstrator knows?"""
        said = torch.tensor([answer_kind(p) for p in self._vla_prev], device=self.env.device)
        self._agree += int((said == self._answer).sum())
        self._judged += self.env.num_envs
        if self._vla_ticks % self.config.report_every != 0 or not self._rtt_ms:
            return
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")  # noqa: E731
        print(
            f"[vla] answers agree with the demonstrator {100.0 * self._agree / max(self._judged, 1):.0f}% "
            f"of steps; model {med(self._act_ms):.0f} ms actions, {med(self._ans_ms):.0f} ms answer, "
            f"{med(self._rtt_ms):.0f} ms round trip",
            flush=True,
        )
        self._agree = self._judged = 0
        self._act_ms, self._ans_ms, self._rtt_ms = [], [], []
