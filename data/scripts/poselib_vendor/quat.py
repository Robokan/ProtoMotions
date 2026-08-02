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
"""Batched quaternion operations in WXYZ convention (matches the poselib NPY
format consumed by data/scripts/convert_quadruped_poselib_to_proto.py)."""

import torch


def quat_identity(shape):
    q = torch.zeros(list(shape) + [4], dtype=torch.float32)
    q[..., 0] = 1.0
    return q


def quat_normalize(q):
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-9)


def quat_mul(a, b):
    """Hamilton product, wxyz. Broadcasts over leading dims."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_conjugate(q):
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_rotate(q, v):
    """Rotate vectors v (..., 3) by quaternions q (..., 4) wxyz."""
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def quat_from_angle_axis(angle, axis):
    """angle (...,), axis (..., 3) -> wxyz quat. Axis is normalized."""
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-9)
    half = angle.unsqueeze(-1) * 0.5
    return torch.cat([torch.cos(half), torch.sin(half) * axis], dim=-1)


def quat_slerp(q0, q1, t):
    """Spherical lerp between q0 and q1 (..., 4) with scalar fraction t."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    small = sin_theta.abs() < 1e-6
    w0 = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta)
    w1 = torch.where(small, torch.full_like(sin_theta, t), torch.sin(t * theta) / sin_theta)
    return quat_normalize(w0 * q0 + w1 * q1)


def yaw_quat(q):
    """Extract the heading (yaw about world Z) component of q (..., 4) wxyz."""
    # forward = q * x_axis
    fwd = quat_rotate(q, torch.tensor([1.0, 0.0, 0.0]).expand(q.shape[:-1] + (3,)))
    yaw = torch.atan2(fwd[..., 1], fwd[..., 0])
    axis = torch.zeros(q.shape[:-1] + (3,))
    axis[..., 2] = 1.0
    return quat_from_angle_axis(yaw, axis)
