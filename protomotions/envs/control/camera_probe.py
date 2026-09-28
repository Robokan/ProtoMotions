# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Write onboard camera frames to disk. Measurement only, drives nothing.

An onboard camera is easy to configure and hard to believe: the prim is
parented somewhere inside the articulation, the offset is in a convention that
is not the robot's, and every one of those mistakes produces an image rather
than an error -- of the floor, of the inside of the robot's own chassis, of
the world rotated ninety degrees. The only check that means anything is
looking at the picture.

Drop it into control_components next to the task and it saves the first few
frames, then stops. It is the eyes-on step before a recorder exists.
"""

import os
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class CameraProbeConfig(ControlComponentConfig):
    """Configuration for the onboard-camera probe."""

    _target_: str = "protomotions.envs.control.camera_probe.CameraProbe"

    every_steps: int = 50
    env_id: int = 0
    # Stop after this many saves per camera; a chase runs for hours and the
    # point is a handful of frames to look at, not a dataset.
    max_frames: int = 12
    # Name of a CameraEye component, or None. Set, each save is two PNGs:
    # the eye's crop (what the student is shown) and the full frame with
    # the crop window outlined.
    eye_component: Optional[str] = None
    out_dir: str = "output/camera"
    data_type: str = "rgb"


class CameraProbe(ControlComponent):
    """Save every Nth frame of each onboard camera as a PNG."""

    config: CameraProbeConfig

    def __init__(self, config: CameraProbeConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self._steps = 0
        self._saved = 0
        self._announced = False

    def reset(self, env_ids: Tensor):
        pass

    def populate_context(self, ctx: EnvContext) -> None:
        """Measurement only: publishes nothing."""

    def step(self):
        self._steps += 1
        if self.config.every_steps <= 0 or self._saved >= self.config.max_frames:
            return
        if self._steps % self.config.every_steps != 0:
            return

        images = self.env.simulator.get_camera_images(self.config.data_type)
        if not images:
            if not self._announced:
                print(
                    "[camera-probe] no onboard cameras are producing "
                    f"'{self.config.data_type}' -- is one configured, and was "
                    "the app launched with cameras enabled?",
                    flush=True,
                )
                self._announced = True
            return

        import imageio.v2 as imageio  # noqa: PLC0415

        os.makedirs(self.config.out_dir, exist_ok=True)
        eye = None
        if self.config.eye_component:
            eye = self.env.control_manager.components.get(self.config.eye_component)
        for name, batch in images.items():
            frame = batch[self.config.env_id].detach().cpu().numpy()
            if frame.ndim == 3 and frame.shape[-1] == 4:
                frame = frame[..., :3]   # RGBA -> RGB, nothing is transparent
            path = os.path.join(
                self.config.out_dir, f"{name}_{self._steps:06d}.png"
            )
            if eye is not None and name == eye.config.camera:
                crop = eye.crop(batch)[self.config.env_id].detach().cpu().numpy()
                imageio.imwrite(
                    os.path.join(
                        self.config.out_dir, f"{name}_eye_{self._steps:06d}.png"
                    ),
                    crop[..., :3],
                )
                frame = _outline(frame, *eye.gaze()[self.config.env_id].tolist())
            imageio.imwrite(path, frame)
            if not self._announced:
                print(
                    f"[camera-probe] {name}: {tuple(batch.shape)} {batch.dtype} "
                    f"-> {path}",
                    flush=True,
                )
        self._announced = True
        self._saved += 1


def _outline(frame, center_u: float, center_v: float, zoom: float):
    """The frame with the eye's window drawn on it in yellow."""
    import numpy as np  # noqa: PLC0415

    out = np.ascontiguousarray(frame).copy()
    h, w = out.shape[:2]
    half = 1.0 / zoom
    x0 = int(round((center_u - half + 1.0) * 0.5 * w))
    x1 = int(round((center_u + half + 1.0) * 0.5 * w)) - 1
    y0 = int(round((center_v - half + 1.0) * 0.5 * h))
    y1 = int(round((center_v + half + 1.0) * 0.5 * h)) - 1
    x0, x1 = max(x0, 0), min(x1, w - 1)
    y0, y1 = max(y0, 0), min(y1, h - 1)
    t = max(h // 180, 1)
    yellow = np.array([255, 220, 0], dtype=out.dtype)
    out[y0 : y0 + t, x0 : x1 + 1] = yellow
    out[y1 - t + 1 : y1 + 1, x0 : x1 + 1] = yellow
    out[y0 : y1 + 1, x0 : x0 + t] = yellow
    out[y0 : y1 + 1, x1 - t + 1 : x1 + 1] = yellow
    return out
