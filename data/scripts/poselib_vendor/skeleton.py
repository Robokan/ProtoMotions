# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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
"""Lightweight skeleton tree + FK and poselib-format NPY writer (wxyz)."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from .quat import quat_conjugate, quat_mul, quat_normalize, quat_rotate


@dataclass
class Skeleton:
    node_names: List[str]
    parent_indices: torch.Tensor  # (J,) long, -1 for root
    local_translation: torch.Tensor  # (J, 3) offsets relative to parent

    def index(self, name: str) -> int:
        return self.node_names.index(name)

    @classmethod
    def from_mjcf(cls, mjcf_path: str) -> "Skeleton":
        """Build a skeleton from an MJCF body tree (document order, like
        protomotions.components.pose_lib.extract_kinematic_info)."""
        root = ET.parse(mjcf_path).getroot()
        names, parents, offsets = [], [], []

        def walk(body, parent_idx):
            idx = len(names)
            names.append(body.get("name"))
            parents.append(parent_idx)
            pos = [float(x) for x in (body.get("pos") or "0 0 0").split()]
            offsets.append(pos)
            assert body.get("quat") is None, "body quat offsets unsupported"
            for child in body.findall("body"):
                walk(child, idx)

        worldbody = root.find("worldbody")
        body_elems = worldbody.findall("body")
        assert len(body_elems) == 1, "expected a single root body"
        walk(body_elems[0], -1)

        return cls(
            node_names=names,
            parent_indices=torch.tensor(parents, dtype=torch.long),
            local_translation=torch.tensor(offsets, dtype=torch.float32),
        )


def fk_global(skeleton: Skeleton, local_rot: torch.Tensor, root_trans: torch.Tensor):
    """Forward kinematics.

    Args:
        local_rot: (..., J, 4) wxyz local rotations (root entry = global root rot)
        root_trans: (..., 3) root translation
    Returns:
        global_rot (..., J, 4), global_pos (..., J, 3)
    """
    J = len(skeleton.node_names)
    global_rot = torch.zeros_like(local_rot)
    global_pos = torch.zeros(local_rot.shape[:-2] + (J, 3), dtype=local_rot.dtype)

    for i in range(J):
        p = int(skeleton.parent_indices[i])
        if p < 0:
            global_rot[..., i, :] = local_rot[..., i, :]
            global_pos[..., i, :] = root_trans
        else:
            global_rot[..., i, :] = quat_mul(
                global_rot[..., p, :], local_rot[..., i, :]
            )
            offset = skeleton.local_translation[i].expand_as(global_pos[..., i, :])
            global_pos[..., i, :] = global_pos[..., p, :] + quat_rotate(
                global_rot[..., p, :], offset
            )
    return quat_normalize(global_rot), global_pos


def global_to_local(skeleton: Skeleton, global_rot: torch.Tensor) -> torch.Tensor:
    """Inverse of fk_global for rotations: (..., J, 4) global -> local wxyz."""
    local_rot = global_rot.clone()
    for i in range(len(skeleton.node_names)):
        p = int(skeleton.parent_indices[i])
        if p >= 0:
            local_rot[..., i, :] = quat_mul(
                quat_conjugate(global_rot[..., p, :]), global_rot[..., i, :]
            )
    return quat_normalize(local_rot)


def save_skeleton_motion_npy(
    path: str,
    skeleton: Skeleton,
    local_rot: torch.Tensor,
    root_trans: torch.Tensor,
    fps: float,
):
    """Write a poselib-SkeletonMotion-compatible NPY dict (wxyz local quats),
    matching what data/scripts/convert_quadruped_poselib_to_proto.py loads."""

    def tensor_to_dict(x):
        arr = x.detach().cpu().numpy()
        return {"arr": arr, "context": {"dtype": arr.dtype.name}}

    data = {
        "rotation": tensor_to_dict(local_rot),
        "root_translation": tensor_to_dict(root_trans),
        "fps": float(fps),
        "is_local": True,
        "wxyz": True,
        "skeleton_tree": {
            "node_names": list(skeleton.node_names),
            "parent_indices": tensor_to_dict(skeleton.parent_indices),
            "local_translation": tensor_to_dict(skeleton.local_translation),
        },
        "__name__": "SkeletonMotion",
    }
    np.save(path, data, allow_pickle=True)
