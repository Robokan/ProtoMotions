# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from dataclasses import dataclass, field
from typing import Any
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig
from protomotions.simulator.isaacgym.config import IsaacGymPhysXParams
import torch


@dataclass
class ProtoMotionsIsaacLabMarkers:
    """Configuration for a single marker instance."""

    marker: Any
    scale: torch.Tensor


@dataclass
class IsaacLabPhysXParams(IsaacGymPhysXParams):
    """PhysX physics engine parameters."""

    # Reduced GPU buffer sizes for multi-GPU training with large motion datasets
    # Original values used ~10GB+ of GPU memory for buffers alone
    gpu_found_lost_pairs_capacity: int = 2**18       # 256K (was 2M)
    gpu_max_rigid_contact_count: int = 2**19         # 512K (was 8M)
    gpu_found_lost_aggregate_pairs_capacity: int = 2**20  # 1M (was 32M)


@dataclass
class IsaacLabSimParams(SimParams):
    """PhysX-specific simulation parameters used by IsaacGym and IsaacLab."""

    physx: IsaacLabPhysXParams = field(default_factory=IsaacLabPhysXParams)


@dataclass
class IsaacLabSimulatorConfig(SimulatorConfig):
    """Configuration specific to IsaacLab simulator."""

    _target_: str = "protomotions.simulator.isaaclab.simulator.IsaacLabSimulator"
    w_last: bool = False
    sim: IsaacLabSimParams = field(default_factory=IsaacLabSimParams)
