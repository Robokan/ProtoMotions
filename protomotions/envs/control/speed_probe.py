# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Report the achieved root speed distribution. Measurement only.

Drop this into any experiment's control_components to get the same numbers out
of different tasks, so "the policy is slow" can be checked against "the policy
is slow AT THIS TASK". It drives nothing and publishes no context.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class RootSpeedProbeConfig(ControlComponentConfig):
    """Configuration for the root-speed probe."""

    _target_: str = "protomotions.envs.control.speed_probe.RootSpeedProbe"

    report_every_steps: int = 250
    label: str = "speed"


class RootSpeedProbe(ControlComponent):
    """Accumulate planar root speed and print its percentiles."""

    config: RootSpeedProbeConfig

    def __init__(self, config: RootSpeedProbeConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self._samples = []

    def reset(self, env_ids: Tensor):
        pass

    def step(self):
        if self.config.report_every_steps <= 0:
            return
        root = self.env.simulator.get_root_state()
        self._samples.append(root.root_vel[:, :2].norm(dim=-1).flatten())
        if len(self._samples) < self.config.report_every_steps:
            return
        s = torch.cat(self._samples)
        q = lambda p: float(s.quantile(p))
        print(
            f"[{self.config.label}] achieved root speed over {len(s)} samples: "
            f"p50 {q(0.5):.2f}  p75 {q(0.75):.2f}  p90 {q(0.9):.2f}  "
            f"p95 {q(0.95):.2f}  p99 {q(0.99):.2f}  max {float(s.max()):.2f} m/s",
            flush=True,
        )
        self._samples = []

    def populate_context(self, ctx: EnvContext) -> None:
        """Measurement only: publishes nothing."""
