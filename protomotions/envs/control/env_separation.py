# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Report how far apart the parallel robots are. Measurement only.

Parallel envs share one world and one sky. That is invisible while the only
consumer of the scene is the physics, and it stops being invisible the moment
a robot carries a camera: another env's dog walking through the frame is an
object the labels say nothing about, and a second red ball is an outright
contradiction. Whether that happens is a question about distances, not about
pixels -- pixels can show you a problem but never prove its absence.

So this measures the distance from each robot to the nearest other one, and
says how often that falls inside a range where the neighbour is more than a
speck. The go2 is about 0.4 m tall; at a 120 deg field of view over 224 px
(1.87 px/deg) it is:

    10 m -> 4.3 px      30 m -> 1.4 px      50 m -> 0.9 px
    20 m -> 2.1 px      40 m -> 1.1 px     100 m -> 0.4 px

Under a pixel it cannot be resolved at all, so 50 m is the point where a
neighbour stops existing as far as the camera is concerned. The default
warning threshold is deliberately more conservative than that.

Note that separation is not a constant: the chase never resets, so each dog
random-walks as it is led around by its own ball. Two that start far apart
can drift together over a long recording, which is exactly why this reports
the minimum seen rather than the distance at spawn.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


def nearest_neighbour(root_xy: Tensor) -> Tensor:
    """Distance from each robot to the nearest other one.

    Infinite when there is only one env, which is the honest answer: nothing
    can wander into frame.
    """
    if root_xy.shape[0] < 2:
        return torch.full(
            (root_xy.shape[0],), float("inf"), device=root_xy.device
        )
    gaps = torch.cdist(root_xy.unsqueeze(0), root_xy.unsqueeze(0)).squeeze(0)
    gaps.fill_diagonal_(float("inf"))
    return gaps.min(dim=-1).values


@dataclass
class EnvSeparationProbeConfig(ControlComponentConfig):
    """Configuration for the env-separation probe."""

    _target_: str = (
        "protomotions.envs.control.env_separation.EnvSeparationProbe"
    )

    report_every_steps: int = 500
    # Below this, a neighbouring robot is a visible object rather than a
    # speck, and any camera recording is contaminated.
    warn_below_m: float = 30.0


class EnvSeparationProbe(ControlComponent):
    """Track the closest approach between any two parallel robots."""

    config: EnvSeparationProbeConfig

    def __init__(self, config: EnvSeparationProbeConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self._steps = 0
        self._closest_ever = float("inf")
        self._breaches = 0
        self._samples = 0
        self._warned = False

    def reset(self, env_ids: Tensor):
        pass

    def populate_context(self, ctx: EnvContext) -> None:
        """Measurement only: publishes nothing."""

    def _nearest(self) -> Tensor:
        return nearest_neighbour(
            self.env.simulator.get_root_state().root_pos[:, :2]
        )

    def step(self):
        if self.env.num_envs < 2 or self.config.report_every_steps <= 0:
            return
        nearest = self._nearest()
        closest = float(nearest.min())
        self._closest_ever = min(self._closest_ever, closest)
        self._breaches += int((nearest < self.config.warn_below_m).sum())
        self._samples += self.env.num_envs
        self._steps += 1

        if closest < self.config.warn_below_m and not self._warned:
            print(
                f"[env-separation] WARNING: two robots are {closest:.1f} m "
                f"apart, inside {self.config.warn_below_m:.0f} m -- each is "
                "in the other's camera. Any recording from here is "
                "contaminated. Raise --env-spacing or drop to one env.",
                flush=True,
            )
            self._warned = True

        if self._steps < self.config.report_every_steps:
            return
        print(
            f"[env-separation] nearest neighbour now {closest:.0f} m, "
            f"closest ever {self._closest_ever:.0f} m, "
            f"{100.0 * self._breaches / max(self._samples, 1):.1f}% of samples "
            f"within {self.config.warn_below_m:.0f} m",
            flush=True,
        )
        self._steps = 0
