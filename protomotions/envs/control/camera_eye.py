# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""A movable eye inside the front camera: pick where to look, and how closely.

The go2's camera streams 1280x720, the student is shown 224 pixels. Shrinking
the whole frame to that throws away most of what the sensor resolves: at
1.87 px/deg a ball at 4 m is a 5-7 px smudge, and past that nothing. So the
sensor is rendered at full resolution and the student is handed a CROP of it
-- a window with the frame's own 16:9 shape, 1/zoom of its size, centred
wherever the eye is pointed -- squashed to out_res x out_res.

Keeping the window 16:9 at every zoom means the squash is the same at every
zoom: the image of a ball changes size and position as the eye moves, never
shape. Zoom 1 is the whole frame, so an eye that is not trying still sees
everything the camera does.

The eye is part of the demonstrator. What it looks at decides what the dog
KNOWS -- with the eye on, the ball is known only while it is inside the
crop, not merely inside the camera -- and where it looks next is part of the
action, so the student learns to aim its own eye:

* observation.state gets the gaze the frame was taken with (u, v, zoom);
* action gets the gaze for the NEXT frame.

The demonstrator's rule is deliberately simple, so it is learnable from the
picture alone: while the ball is in the crop, centre on it and zoom in
(rate-limited) until it spans ball_frac of the window; when it is not, zoom
out one level per update, widening around where it was looking, until it is
back or the eye is at zoom 1 -- and only then does the dog start a search.
Nothing here uses where the ball is outside the crop -- the student will
not have that.

Coordinates are normalised to the full frame: u in [-1, 1] left to right,
v in [-1, 1] top to bottom. A window of zoom z has half-extent 1/z in both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)
from protomotions.utils import rotations

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


def project_to_camera(
    points_w: Tensor,
    root_pos: Tensor,
    root_rot: Tensor,
    lens_forward_m: float,
    hfov_deg: float,
    aspect: float,
):
    """World points -> normalised image coordinates of a body-mounted camera.

    The camera is level, looking along the body's +X from lens_forward_m
    ahead of the root -- go2_front_camera -- and rides the FULL body pose, so
    pitch and roll move the ball in the image exactly as they do on the
    robot. Pinhole, like the rendered camera: horizontal FOV hfov_deg, and
    the vertical one follows from aspect (width / height).

    Returns (u, v, depth, tan_half_h, tan_half_v): u right, v down, both 1.0
    at the frame edge; depth along the optical axis (<= 0 is behind the lens).
    """
    offset = torch.zeros_like(root_pos)
    offset[:, 0] = lens_forward_m
    lens = root_pos + rotations.quat_rotate(root_rot, offset, True)
    d = rotations.quat_rotate_inverse(root_rot, points_w - lens, True)
    tan_h = math.tan(math.radians(hfov_deg) * 0.5)
    tan_v = tan_h / aspect
    depth = d[:, 0]
    safe = depth.clamp_min(1e-6)
    u = -d[:, 1] / (safe * tan_h)
    v = -d[:, 2] / (safe * tan_v)
    return u, v, depth, tan_h, tan_v


def window_contains(
    u: Tensor,
    v: Tensor,
    depth: Tensor,
    radius_m: float,
    tan_h: float,
    tan_v: float,
    center_u: Tensor,
    center_v: Tensor,
    zoom: Tensor,
) -> Tensor:
    """Is a ball of radius_m at (u, v, depth) recognisably inside the window?

    "Recognisably" = its centre is within half its own radius of the edge,
    so at least a clear cap of it is on screen. Testing the bare centre says
    "gone" while a third of the ball is still in frame, which reads -- to
    anyone watching the camera -- as the dog searching for a ball it can see.
    """
    safe = depth.clamp_min(1e-6)
    r_u = radius_m / (safe * tan_h)
    r_v = radius_m / (safe * tan_v)
    half = 1.0 / zoom
    return (
        (depth > radius_m)
        & ((u - center_u).abs() <= half + 0.5 * r_u)
        & ((v - center_v).abs() <= half + 0.5 * r_v)
    )


def crop_frames(
    frames: Tensor, center_u: Tensor, center_v: Tensor, zoom: Tensor, out_res: int
) -> Tensor:
    """[N, H, W, C] uint8 frames -> [N, out_res, out_res, C] crops.

    Area-correct (antialiased) resampling: at zoom 1 a 1280-wide frame is
    shrunk 5.7x, and plain bilinear at that ratio aliases a small ball into
    flicker -- exactly the detail the eye exists to keep.
    """
    n, h, w, c = frames.shape
    out = torch.empty(
        (n, out_res, out_res, c), dtype=frames.dtype, device=frames.device
    )
    cu = center_u.detach().cpu().tolist()
    cv = center_v.detach().cpu().tolist()
    zz = zoom.detach().cpu().tolist()
    for i in range(n):
        half = 1.0 / zz[i]
        x0 = int(round((cu[i] - half + 1.0) * 0.5 * w))
        y0 = int(round((cv[i] - half + 1.0) * 0.5 * h))
        cw = max(int(round(w * half)), 1)
        ch = max(int(round(h * half)), 1)
        x0 = min(max(x0, 0), w - cw)
        y0 = min(max(y0, 0), h - ch)
        patch = frames[i, y0 : y0 + ch, x0 : x0 + cw].permute(2, 0, 1)
        patch = patch.unsqueeze(0).float()
        resized = F.interpolate(
            patch, size=(out_res, out_res), mode="bilinear",
            align_corners=False, antialias=True,
        )
        out[i] = resized[0].permute(1, 2, 0).round().clamp(0, 255).to(frames.dtype)
    return out


@dataclass
class CameraEyeConfig(ControlComponentConfig):
    """Configuration for the movable eye."""

    _target_: str = "protomotions.envs.control.camera_eye.CameraEye"

    camera: str = "front_camera"
    # Which control component is the demonstrator: it owns the ball, the
    # lens geometry and the sight test this eye gates.
    goal_component: str = "masked_mimic"
    # What the student is shown: out_res x out_res.
    out_res: int = 224
    # Deepest zoom. At 4 a 1280x720 sensor gives a 320x180 window -- still
    # downsampled to 224, so no zoom level is inventing pixels. 5.7 is 1:1.
    max_zoom: float = 4.0
    # Zoom until the ball spans this fraction of the window's width. Leaves
    # room for the ball to move between frames without leaving the window.
    ball_frac: float = 0.15
    # Most the zoom may grow per eye update (x).
    zoom_in_per_update: float = 1.5
    # One level out per eye update (/) while the ball is not in the window,
    # widening around where it was: 4 -> 2 -> 1. The dog keeps chasing its
    # last sighting meanwhile, and only searches once the eye is all the way
    # out and still has nothing (ball_chase._update_belief).
    zoom_out_per_update: float = 2.0
    # The eye moves at the rate the student will see frames (same as the
    # recorder), so each recorded frame has one gaze and one next gaze.
    hz: float = 10.0
    # Viewer only: outline the window with dots this far in front of the
    # lens. Drawn in the world, on the camera's own frustum, so looking
    # through the head camera the outline IS the crop -- at any viewport
    # size -- and from outside it shows where the dog is looking.
    outline_distance_m: float = 0.5
    outline_dots_per_side: int = 12


class CameraEye(ControlComponent):
    """Where the student's 224x224 is taken from within the full frame."""

    config: CameraEyeConfig

    def __init__(self, config: CameraEyeConfig, env: "BaseEnv"):
        super().__init__(config, env)
        n, dev = env.num_envs, env.device
        self._every = max(int(round(1.0 / (config.hz * env.dt))), 1)
        self._steps = 0
        self.center_u = torch.zeros(n, device=dev)
        self.center_v = torch.zeros(n, device=dev)
        self.zoom = torch.ones(n, device=dev)
        self._next: Optional[tuple] = None
        self._next_step = -1

    def reset(self, env_ids: Tensor) -> None:
        self.center_u[env_ids] = 0.0
        self.center_v[env_ids] = 0.0
        self.zoom[env_ids] = 1.0

    def populate_context(self, ctx: EnvContext) -> None:
        """The gaze is read directly by the recorder and the sight test."""

    def _goal(self):
        return self.env.control_manager.components[self.config.goal_component]

    def contains(self, u: Tensor, v: Tensor, depth: Tensor, radius_m, tan_h, tan_v):
        """Sight test for the demonstrator: is it inside the CURRENT window."""
        return window_contains(
            u, v, depth, radius_m, tan_h, tan_v,
            self.center_u, self.center_v, self.zoom,
        )

    def gaze(self) -> Tensor:
        """[N, 3] (u, v, zoom) the current frame is being taken with."""
        return torch.stack([self.center_u, self.center_v, self.zoom], dim=-1)

    def next_gaze(self) -> Tensor:
        """[N, 3] the gaze the demonstrator chooses for the next frame.

        Computed once per step and cached, so the recorder's label and the
        gaze actually committed at the end of the step are the same numbers.
        """
        if self._next is not None and self._next_step == self._steps:
            return torch.stack(self._next, dim=-1)
        goal = self._goal()
        u, v, depth, tan_h, tan_v = goal._ball_image()
        radius = goal.config.ball_radius_m
        # In the window NOW -- tested here rather than read off the
        # demonstrator's belief, which a privileged chase pins to True even
        # with the ball behind the lens.
        seen = self.contains(u, v, depth, radius, tan_h, tan_v)
        # Zoom that makes the ball span ball_frac of the window width: the
        # ball is 2*r_u of the 2-unit frame, the window is 2/zoom.
        r_u = radius / (depth.clamp_min(1e-3) * tan_h)
        want = (self.config.ball_frac / r_u.clamp_min(1e-6)).clamp(
            1.0, self.config.max_zoom
        )
        z_in = torch.minimum(want, self.zoom * self.config.zoom_in_per_update)
        z_out = (self.zoom / self.config.zoom_out_per_update).clamp_min(1.0)
        z = torch.where(seen, z_in, z_out)
        # Centre on the ball; lost, widen around where the window already
        # is. Either way slide the window back inside the frame.
        half = 1.0 / z
        cu = torch.where(seen, u, self.center_u)
        cv = torch.where(seen, v, self.center_v)
        cu = torch.maximum(torch.minimum(cu, 1.0 - half), half - 1.0)
        cv = torch.maximum(torch.minimum(cv, 1.0 - half), half - 1.0)
        self._next = (cu, cv, z)
        self._next_step = self._steps
        return torch.stack(self._next, dim=-1)

    def crop(self, frames: Tensor) -> Tensor:
        """The student's view of a batch of full-resolution frames."""
        return crop_frames(
            frames, self.center_u, self.center_v, self.zoom, self.config.out_res
        )

    def is_tick(self) -> bool:
        """True on the steps a frame is taken (and the eye then moves)."""
        return (self._steps + 1) % self._every == 0

    def step(self) -> None:
        # Stepped LAST: the recorder has already written this frame with the
        # current gaze and next_gaze() as its label; now the eye moves.
        if self.is_tick():
            cu, cv, z = self.next_gaze().unbind(-1)
            self.center_u, self.center_v, self.zoom = cu.clone(), cv.clone(), z.clone()
        self._steps += 1

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, VisualizationMarkerConfig]:
        if headless:
            return {}
        return {
            "eye_window": VisualizationMarkerConfig(
                type="sphere",
                color=(1.0, 0.85, 0.0),
                markers=[
                    MarkerConfig(size="tiny")
                    for _ in range(4 * self.config.outline_dots_per_side)
                ],
            )
        }

    def get_markers_state(self) -> Dict[str, MarkerState]:
        if not self.env.simulator.show_markers:
            return {}
        goal = self._goal()
        n, k = self.env.num_envs, self.config.outline_dots_per_side
        dev = self.env.device
        half = 1.0 / self.zoom
        u0, u1 = self.center_u - half, self.center_u + half
        v0, v1 = self.center_v - half, self.center_v + half
        t = torch.linspace(0.0, 1.0, k + 1, device=dev)[:-1].unsqueeze(0)
        lerp = lambda a, b: a.unsqueeze(-1) + (b - a).unsqueeze(-1) * t  # noqa: E731
        us = torch.cat([lerp(u0, u1), u1.unsqueeze(-1).expand(n, k),
                        lerp(u1, u0), u0.unsqueeze(-1).expand(n, k)], dim=-1)
        vs = torch.cat([v0.unsqueeze(-1).expand(n, k), lerp(v0, v1),
                        v1.unsqueeze(-1).expand(n, k), lerp(v1, v0)], dim=-1)
        # Back-project through the same pinhole the sight test uses.
        tan_h = math.tan(math.radians(goal.config.fov_deg) * 0.5)
        tan_v = tan_h / goal.config.sight_aspect
        d = self.config.outline_distance_m
        local = torch.stack(
            [torch.full_like(us, d), -us * d * tan_h, -vs * d * tan_v], dim=-1
        )
        local[..., 0] += goal.config.sight_forward_m
        root = self.env.simulator.get_root_state()
        rot = root.root_rot.unsqueeze(1).expand(n, 4 * k, 4).reshape(-1, 4)
        world = rotations.quat_rotate(rot, local.reshape(-1, 3), True).view(n, 4 * k, 3)
        world = world + root.root_pos.unsqueeze(1)
        quat = torch.zeros(n, 4 * k, 4, device=dev)
        quat[..., 3] = 1.0
        return {"eye_window": MarkerState(translation=world, orientation=quat)}

    def report(self) -> Dict[str, float]:
        return {
            "zoom_mean": float(self.zoom.mean()),
            "zoom_max": float(self.zoom.max()),
        }
