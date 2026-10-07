# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Show where an onboard camera is: a dot at the lens, dots along its axis.

Viewer only, and annotation (never in camera frames): for checking by eye
that the lens sits where the real one does and looks where it should. The
camera must ride the robot's root body (go2_front_camera rides base_link).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

import torch
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


@dataclass
class CameraMarkerConfig(ControlComponentConfig):
    """Configuration for the camera marker."""

    _target_: str = "protomotions.envs.control.camera_marker.CameraMarker"

    camera: str = "front_camera"
    # Axis dots from the lens out to this far ahead (m), this many of them.
    axis_length_m: float = 0.6
    axis_dots: int = 12


class CameraMarker(ControlComponent):
    """Draw the lens and its optical axis in the viewer."""

    config: CameraMarkerConfig

    def __init__(self, config: CameraMarkerConfig, env: "BaseEnv"):
        super().__init__(config, env)
        cams = getattr(env.simulator.config, "onboard_cameras", {}) or {}
        self._cam = cams.get(config.camera)
        if self._cam is None:
            print(f"[camera-marker] no onboard camera '{config.camera}' -- nothing to show.", flush=True)

    def reset(self, env_ids: Tensor) -> None:
        pass

    def step(self) -> None:
        pass

    def populate_context(self, ctx: EnvContext) -> None:
        """Visualization only."""

    def create_visualization_markers(self, headless: bool) -> Dict[str, VisualizationMarkerConfig]:
        if headless or self._cam is None:
            return {}
        return {
            "camera_lens": VisualizationMarkerConfig(
                type="sphere", color=(1.0, 0.0, 1.0), markers=[MarkerConfig(scale=0.02)]
            ),
            "camera_axis": VisualizationMarkerConfig(
                type="sphere",
                color=(0.0, 0.9, 1.0),
                markers=[MarkerConfig(size="tiny") for _ in range(self.config.axis_dots)],
            ),
        }

    def get_markers_state(self) -> Dict[str, MarkerState]:
        if self._cam is None or not self.env.simulator.show_markers:
            return {}
        n, dev = self.env.num_envs, self.env.device
        root = self.env.simulator.get_root_state()
        pitch = math.radians(getattr(self._cam, "pitch_deg", 0.0))
        yaw = math.radians(getattr(self._cam, "yaw_deg", 0.0))
        # Optical axis in the body frame: +X, pitched down and yawed left.
        axis = torch.tensor(
            [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)],
            device=dev,
        )
        lens_b = torch.tensor(list(self._cam.pos), device=dev, dtype=torch.float32)
        k = self.config.axis_dots
        along = torch.linspace(self.config.axis_length_m / k, self.config.axis_length_m, k, device=dev)
        pts_b = torch.cat([lens_b.unsqueeze(0), lens_b + along.unsqueeze(-1) * axis], dim=0)
        rot = root.root_rot.unsqueeze(1).expand(n, k + 1, 4).reshape(-1, 4)
        pts = rotations.quat_rotate(rot, pts_b.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3), True)
        pts = pts.view(n, k + 1, 3) + root.root_pos.unsqueeze(1)
        quat = torch.zeros(n, k + 1, 4, device=dev)
        quat[..., 3] = 1.0
        return {
            "camera_lens": MarkerState(
                translation=pts[:, :1].contiguous(), orientation=quat[:, :1].contiguous()
            ),
            "camera_axis": MarkerState(
                translation=pts[:, 1:].contiguous(), orientation=quat[:, 1:].contiguous()
            ),
        }
