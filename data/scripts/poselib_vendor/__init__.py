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
"""Minimal vendored poselib utilities (wxyz quaternion convention).

The BVH parsing logic is adapted from the project owner's modified NVIDIA
poselib (BSD-3-Clause, Copyright (c) 2018-2022 NVIDIA Corporation,
skeleton3d.py parse_bvh_file / euler_to_quaternion) which is the trusted
path the existing Go2 motion NPYs came through. Quaternion math and FK are
reimplemented here so that no IsaacLab/pxr/fbx dependencies are needed.
"""

from .quat import (  # noqa: F401
    quat_identity,
    quat_normalize,
    quat_mul,
    quat_conjugate,
    quat_rotate,
    quat_from_angle_axis,
    quat_slerp,
    yaw_quat,
)
from .bvh import parse_bvh_file  # noqa: F401
from .skeleton import (  # noqa: F401
    Skeleton,
    fk_global,
    global_to_local,
    save_skeleton_motion_npy,
)
