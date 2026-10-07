# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Named poses taken straight from mocap frames, for MaskedMimic to hold.

A pose is one or more frames of clips in the training corpus. Each is stored
relative to its own root and heading, then re-anchored wherever the robot is
and conditioned on every tracked body -- "be in this pose, here, by then".
A pose that is a chain is reached through its earlier frames first (a beg
goes via the sit), which is how the corpus itself gets there.

Measured on the go2 corpus (see the --pose-test runs of 2026-09-28): the sit
lands within 3-6 cm per body, and the beg -- both front paws up, balanced on
the hind legs -- is reached from the sit. A clean single-paw-up and
lying on the side are not reachable from this corpus.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from protomotions.utils import rotations

# A pose is a sequence of steps, each re-anchored where the dog is when it
# begins (except a hold, which stays put):
#   key(clip, t[, reach]) -- that frame, reached in `reach` seconds
#                            (default pose_reach_sec);
#   play(clip, t0, t1)    -- the clip itself from t0 to t1, in real time, as
#                            the sparse future frames MaskedMimic was trained
#                            to follow. Backwards if t1 < t0: MaskedMimic is
#                            conditioned on poses at times, never on
#                            velocities, so a reversed clip is as valid.
# The last step is held; its release (POSE_RELEASE) follows the hold.
#
# Transitions are PLAYED, not jumped to. From a sit, "trunk higher by time T"
# is ambiguous -- the corpus both stands up and begs from there -- and the
# prior picked either (measured 2026-09-30: of begs built from keyframes,
# half stood up instead; a sit told to stand up begged). What tells them
# apart is the motion on the way: getting up keeps the paws down while the
# hind end rises, a beg lifts the paws. Playing the corpus's own transition
# says which.


def key(clip: str, seconds: float, reach: float = None) -> dict:
    return {"kind": "key", "clip": clip, "t": float(seconds), "reach": reach}


def play(clip: str, start: float, end: float) -> dict:
    """The clip from start to end in real time; backwards if end < start."""
    return {"kind": "play", "clip": clip, "t0": float(start), "t1": float(end)}


# The still end of 13_clip_1: from a stand the sit is a clean fold, while a
# dog that is already low slides into lying instead.
STAND = key("13_clip_1", 3.0)
# 25_clip_1 7.4-8.3 s: standing (0.33 m, nose +13) folding into the sit it
# then holds (0.30 m, +19, paws down).
SIT_DOWN = play("25_clip_1", 7.4, 8.3)
# 30_clip_2 3.1-3.9 s: sitting (0.30 m, +16) up onto four feet, front paws
# down. Getting up from a sit is rare in the corpus (11 get-ups in the whole
# library), and the prior only does the common style: this one and
# 33_clip_4 6.05-6.8 stood 59 of 59 dogs up; 47_clip_5's and 18_clip_5's
# left most of them sitting (measured 2026-09-30).
GET_UP = play("30_clip_2", 3.1, 3.9)
GO2_POSES: Dict[str, List[dict]] = {
    "stand": [STAND],
    "sit": [STAND, SIT_DOWN],
    # Both front paws up from the sit, the way every beg in the corpus goes
    # (the 18_clip_* and 33_clip_* begs: sit, rise in 0.3-0.5 s, hold ~1 s,
    # back down to the sit). 33_clip_3 2.5-3.4 s is a sit (0.30 m, +16)
    # rising into a steady beg (0.36 m, +37, paws 34 / 36 cm) that lasts to
    # the clip's end. (25_clip_1:0.38, used first, is a 0.22 s flash of the
    # paws -- held for 2 s the dog either never rose or rose and fell.)
    "beg": [STAND, SIT_DOWN, play("33_clip_3", 2.5, 3.4)],
    "lie": [STAND, key("7_clip_3_lay_down", 0.9), key("7_clip_3_lay_down", 2.5)],
}

# Seconds a pose is held once reached, where the corpus says less than the
# chase's default (MaskedMimicGoalControlConfig.pose_hold_sec): no beg in the
# corpus lasts past ~1.3 s, so a longer one is a balance the prior never saw.
POSE_HOLD_SEC: Dict[str, float] = {"beg": 1.0}

# How a pose is left, after its hold and before the next command: every pose
# ends STANDING. The chase's own stand-up only says "trunk to standing
# height" and keeps the trunk's current tilt (preserve_tilt), so a begging
# dog stayed up and a sitting one rose into a beg, then fell forward when the
# next keyframe pulled its paws down (nose -33 to -56 deg, measured). Every
# corpus beg ends by dropping back into the sit, and the beg's own rise played
# backwards does exactly that from exactly the held beg. The corpus's drops
# (18_clip_3 / _5 / _8) start from other, higher begs (+44 to +68 deg): the
# jump in the targets made a held beg rear to +70 and crash instead.
POSE_RELEASE: Dict[str, List[dict]] = {
    "sit": [GET_UP],
    "beg": [play("33_clip_3", 3.4, 2.5), GET_UP],
    "lie": [STAND],
}


def _up(n: int, device) -> Tensor:
    up = torch.zeros(n, 3, device=device)
    up[:, 2] = 1.0
    return up


def _clip_id(motion_lib, clip: str) -> int:
    files = [str(f) for f in motion_lib.motion_files]
    matches = [i for i, f in enumerate(files) if f.split("/")[-1] == f"{clip}.motion"]
    if not matches:
        raise ValueError(f"no clip '{clip}' in the motion library")
    return matches[0]


def _relative(pos: Tensor, rot: Tensor, origin_xy: Tensor, origin_rot: Tensor):
    """Body poses [M, B, *] relative to an origin root position and heading."""
    m, b = pos.shape[0], pos.shape[1]
    heading = rotations.calc_heading(origin_rot.view(1, 4), True)
    inv = rotations.quat_from_angle_axis(-heading, _up(1, pos.device), True)
    rel = pos.clone()
    rel[..., :2] -= origin_xy.view(1, 1, 2)
    inv_b = inv.expand(m * b, 4)
    return (
        rotations.quat_rotate(inv_b, rel.reshape(-1, 3), True).view(m, b, 3),
        rotations.quat_mul(inv_b, rot.reshape(-1, 4), True).view(m, b, 4),
    )


def load_keyframe(motion_lib, clip: str, seconds: float, device) -> dict:
    """One corpus frame as body poses relative to its root position and heading."""
    state = motion_lib.get_motion_state(
        torch.tensor([_clip_id(motion_lib, clip)], device=device),
        torch.tensor([float(seconds)], device=device),
    )
    pos, rot = state.rigid_body_pos[0], state.rigid_body_rot[0]
    rel_pos, rel_rot = _relative(pos[None], rot[None], pos[0, :2], rot[0])
    fwd = rotations.quat_rotate(rot[:1], torch.tensor([[1.0, 0.0, 0.0]], device=device), True)[0]
    return {
        "spec": f"{clip}:{seconds:g}",
        "pos": rel_pos[0],
        "rot": rel_rot[0],
        "pitch": math.degrees(math.atan2(float(fwd[2]), float(torch.linalg.norm(fwd[:2])))),
    }


def load_step(motion_lib, step: dict, device) -> dict:
    """A pose step made ready to play: its clip, and the frame it ends on."""
    out = dict(step)
    if step["kind"] == "key":
        out["frame"] = load_keyframe(motion_lib, step["clip"], step["t"], device)
        return out
    mid = _clip_id(motion_lib, step["clip"])
    start = motion_lib.get_motion_state(
        torch.tensor([mid], device=device), torch.tensor([step["t0"]], device=device)
    )
    out["mid"] = mid
    out["origin_xy"] = start.rigid_body_pos[0, 0, :2].clone()
    out["origin_rot"] = start.rigid_body_rot[0, 0].clone()
    out["frame"] = load_keyframe(motion_lib, step["clip"], step["t1"], device)
    out["frame"]["spec"] = f"{step['clip']}:{step['t0']:g}-{step['t1']:g}"
    return out


def play_frames(motion_lib, step: dict, clip_times: Tensor) -> Tuple[Tensor, Tensor]:
    """A played step's frames at these clip times, relative to where it starts."""
    m = clip_times.numel()
    state = motion_lib.get_motion_state(
        torch.full((m,), step["mid"], dtype=torch.long, device=clip_times.device),
        clip_times.reshape(-1),
    )
    return _relative(state.rigid_body_pos, state.rigid_body_rot, step["origin_xy"], step["origin_rot"])


def place_bodies(pos: Tensor, rot: Tensor, anchor_xy: Tensor, anchor_heading: Tensor, terrain):
    """Relative body poses [N, B, *] put at each env's anchor, facing its heading."""
    n, b, dev = pos.shape[0], pos.shape[1], pos.device
    spin = rotations.quat_from_angle_axis(anchor_heading, _up(n, dev), True)
    spin_b = spin.unsqueeze(1).expand(n, b, 4).reshape(-1, 4)
    out = rotations.quat_rotate(spin_b, pos.reshape(-1, 3), True).view(n, b, 3)
    out[..., :2] += anchor_xy.unsqueeze(1)
    out[..., 2] += terrain.get_ground_heights(anchor_xy).view(n, 1)
    return out, rotations.quat_mul(spin_b, rot.reshape(-1, 4), True).view(n, b, 4)


def place(keyframe: dict, anchor_xy: Tensor, anchor_heading: Tensor, terrain) -> Tuple[Tensor, Tensor]:
    """A keyframe put at each env's anchor, facing its heading: [N, B, 3], [N, B, 4]."""
    n, b = anchor_xy.shape[0], keyframe["pos"].shape[0]
    return place_bodies(
        keyframe["pos"].unsqueeze(0).expand(n, b, 3), keyframe["rot"].unsqueeze(0).expand(n, b, 4),
        anchor_xy, anchor_heading, terrain,
    )
