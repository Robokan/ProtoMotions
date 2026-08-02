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
"""BVH parser producing wxyz local quaternions.

Adapted from the project owner's modified NVIDIA poselib
(BSD-3-Clause, Copyright (c) 2018-2022 NVIDIA Corporation;
skeleton3d.py: parse_bvh_file / euler_to_quaternion). This is the trusted
parsing path that produced the Go2 training NPYs.
"""

import math

import torch


def euler_to_quaternion(rotation_values, rotation_axes):
    """Compose intrinsic Euler rotations (radians, channel order) into a wxyz quat."""
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    for angle, axis in zip(rotation_values, rotation_axes):
        half = angle / 2.0
        s, c = math.sin(half), math.cos(half)
        if axis == "x":
            q_axis = torch.tensor([c, s, 0.0, 0.0], dtype=torch.float32)
        elif axis == "y":
            q_axis = torch.tensor([c, 0.0, s, 0.0], dtype=torch.float32)
        elif axis == "z":
            q_axis = torch.tensor([c, 0.0, 0.0, s], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown rotation axis {axis}")
        # Hamilton product q = q * q_axis
        w0, x0, y0, z0 = q.tolist()
        w1, x1, y1, z1 = q_axis.tolist()
        q = torch.tensor(
            [
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ],
            dtype=torch.float32,
        )
    return q


def parse_bvh_file(bvh_file_path):
    """Parse a BVH file.

    Returns:
        joint_names: list[str] (End Sites named '<parent>_end')
        joint_parents: LongTensor (J,)
        root_translations: FloatTensor (N, 3) in BVH units (cm for this dataset)
        local_offsets: FloatTensor (J, 3) in BVH units
        local_rotations: FloatTensor (N, J, 4) wxyz
        fps: float
    """
    with open(bvh_file_path, "r") as f:
        lines = f.readlines()

    joint_names = []
    joint_parents = []
    local_offsets = []
    joint_channels = []

    stack = []
    parent_index = -1
    index = 0
    parsing_motion = False
    frame_time = 1.0 / 60.0
    frames = []

    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        if tokens[0] == "HIERARCHY":
            continue
        elif tokens[0] in ("ROOT", "JOINT"):
            joint_names.append(tokens[1])
            joint_parents.append(parent_index)
            parent_index = index
            index += 1
            stack.append(parent_index)
            joint_channels.append([])
            local_offsets.append(None)
        elif tokens[0] == "End" and tokens[1] == "Site":
            joint_names.append(joint_names[parent_index] + "_end")
            joint_parents.append(parent_index)
            parent_index = index
            index += 1
            stack.append(parent_index)
            joint_channels.append([])
            local_offsets.append(None)
        elif tokens[0] == "{":
            continue
        elif tokens[0] == "}":
            stack.pop()
            parent_index = stack[-1] if stack else -1
        elif tokens[0] == "OFFSET":
            local_offsets[parent_index] = [float(t) for t in tokens[1:4]]
        elif tokens[0] == "CHANNELS":
            num_channels = int(tokens[1])
            joint_channels[parent_index] = tokens[2 : 2 + num_channels]
        elif tokens[0] == "MOTION":
            parsing_motion = True
        elif parsing_motion:
            if tokens[0] == "Frames:":
                continue
            elif tokens[0] == "Frame" and tokens[1] == "Time:":
                frame_time = float(tokens[2])
            else:
                frames.append([float(v) for v in tokens])

    fps = 1.0 / frame_time
    num_joints = len(joint_names)
    num_frames = len(frames)

    local_rotations = torch.zeros((num_frames, num_joints, 4), dtype=torch.float32)
    local_rotations[..., 0] = 1.0
    root_translations = torch.zeros((num_frames, 3), dtype=torch.float32)

    for frame_index, frame_data in enumerate(frames):
        channel_pointer = 0
        for joint_index in range(num_joints):
            channels = joint_channels[joint_index]
            if not channels:
                continue  # End Sites have no channels

            rotation_values = []
            rotation_axes = []
            position_values = []
            position_channels = []
            for ch in channels:
                value = frame_data[channel_pointer]
                if "position" in ch.lower():
                    position_values.append(value)
                    position_channels.append(ch)
                elif "rotation" in ch.lower():
                    rotation_values.append(math.radians(value))
                    rotation_axes.append(ch[0].lower())
                channel_pointer += 1

            if rotation_values:
                local_rotations[frame_index, joint_index] = euler_to_quaternion(
                    rotation_values, rotation_axes
                )

            if joint_index == 0 and position_values:
                pos = torch.zeros(3, dtype=torch.float32)
                for val, ch in zip(position_values, position_channels):
                    pos[{"Xposition": 0, "Yposition": 1, "Zposition": 2}[ch]] = val
                root_translations[frame_index] = pos

    return (
        joint_names,
        torch.tensor(joint_parents, dtype=torch.long),
        root_translations,
        torch.tensor(local_offsets, dtype=torch.float32),
        local_rotations,
        fps,
    )
