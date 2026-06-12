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
"""Convert Go2 poselib NPY motion clips to ProtoMotions .pt format.

Input: poselib SkeletonMotion NPY files (local wxyz quaternions, ~60 fps)
Output: packed ProtoMotions MotionLib .pt file ready for training

Usage:
    cd ~/sparkpack/ProtoMotions
    python data/scripts/convert_quadruped_poselib_to_proto.py \\
        --yaml-file /path/to/full_set.yaml \\
        --motion-dir /path/to/go2/npy_clips/ \\
        --output data/motions/go2/go2_full.pt
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
import typer
from tqdm import tqdm

from protomotions.components.pose_lib import (
    extract_kinematic_info,
    fk_from_transforms_with_velocities,
    extract_qpos_from_transforms,
    compute_cartesian_velocity,
)
from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.utils.rotations import quaternion_to_matrix

app = typer.Typer(pretty_exceptions_enable=False)

# Defaults target Go2; override via CLI options for other robots (e.g. anymal_d).
MJCF_PATH = "protomotions/data/assets/mjcf/go2.xml"
# Poselib uses 'trunk' for the root body; MJCF uses 'base_link' — same body, different name.
POSELIB_ROOT_NAME = "trunk"
MJCF_ROOT_NAME = "base_link"


def load_poselib_npy(npy_path: str):
    """Load a poselib SkeletonMotion NPY file.

    Returns:
        rotation: (N, num_bodies, 4) local wxyz quaternions
        root_translation: (N, 3) root positions
        fps: float
        node_names: list[str]
    """
    data = np.load(npy_path, allow_pickle=True).item()

    rotation = data["rotation"]["arr"]          # (N, B, 4) wxyz local quats
    root_translation = data["root_translation"]["arr"]  # (N, 3)
    fps = float(data["fps"])
    node_names = list(data["skeleton_tree"]["node_names"])

    assert data.get("is_local", True), "Expected local rotations in poselib file"
    assert data.get("wxyz", True), "Expected wxyz quaternion convention"

    return rotation, root_translation, fps, node_names


def verify_skeleton_order(
    node_names: list,
    kinematic_body_names: list,
    poselib_root_name: str = POSELIB_ROOT_NAME,
    mjcf_root_name: str = MJCF_ROOT_NAME,
):
    """Verify poselib body order matches MJCF body order (modulo root name)."""
    poselib_names = [mjcf_root_name if n == poselib_root_name else n for n in node_names]
    if poselib_names != kinematic_body_names:
        raise ValueError(
            f"Skeleton mismatch!\nPoselib: {poselib_names}\nMJCF:    {kinematic_body_names}"
        )


def convert_clip(
    rotation_np: np.ndarray,
    root_translation_np: np.ndarray,
    fps: float,
    kinematic_info,
    device: torch.device,
    dtype: torch.dtype,
    output_fps: int,
) -> Optional[object]:
    """Convert one poselib clip to a ProtoMotions RobotState motion.

    Args:
        rotation_np: (N, B, 4) local wxyz quaternions
        root_translation_np: (N, 3) root positions
        fps: source fps
        kinematic_info: extracted from MJCF
        device/dtype: torch settings
        output_fps: target fps (will downsample if source fps > output_fps)

    Returns:
        RobotState motion, or None if clip is too short.
    """
    factor = max(1, round(fps / output_fps))

    rotation_np = rotation_np[::factor]        # (M, B, 4)
    root_translation_np = root_translation_np[::factor]  # (M, 3)

    N = rotation_np.shape[0]
    if N < 4:
        return None  # too short for velocity estimation

    # Convert to torch
    rot_quats = torch.from_numpy(rotation_np).to(device, dtype)    # (N, B, 4) wxyz
    root_pos = torch.from_numpy(root_translation_np).to(device, dtype)  # (N, 3)

    # Convert wxyz quats → rotation matrices (N, B, 3, 3)
    joint_rot_mats = quaternion_to_matrix(rot_quats, w_last=False)  # (N, B, 3, 3)

    # Forward kinematics → global body positions/rotations/velocities
    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=joint_rot_mats,
        fps=output_fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )

    # Extract joint angles from rotation matrices (inverse FK)
    qpos = extract_qpos_from_transforms(kinematic_info, root_pos, joint_rot_mats)
    motion.dof_pos = qpos[:, 7:]  # strip root pos + root quat

    # Compute DOF velocities via finite differences on joint angles
    joint_angles = qpos[:, 7:]
    dof_vel = compute_cartesian_velocity(
        batched_robot_pos=joint_angles.unsqueeze(1),
        fps=output_fps,
    )
    motion.dof_vel = dof_vel.squeeze(1)

    # Fix height so feet don't clip below ground
    translation_vecs = motion.fix_height_per_frame(height_offset=0.02)
    if motion.rigid_body_vel is not None:
        vel_delta = torch.zeros(
            translation_vecs.shape[0], 1, 3, device=device, dtype=dtype
        )
        vel_delta[:-1] = (
            (translation_vecs[1:] - translation_vecs[:-1]).unsqueeze(1) / motion.motion_dt
        )
        motion.rigid_body_vel = motion.rigid_body_vel + vel_delta
    motion.fix_height(height_offset=0.04)

    # Zero contacts (contact detection not available for these clips)
    motion.rigid_body_contacts = torch.zeros(
        N, kinematic_info.num_bodies, device=device, dtype=torch.bool
    )

    # Disable local rot interpolation in MotionLib
    motion.local_rigid_body_rot = None

    return motion


@app.command()
def main(
    yaml_file: Path = typer.Option(
        ..., help="YAML file listing motion clips (poselib format with 'motions' key)"
    ),
    motion_dir: Optional[Path] = typer.Option(
        None,
        help="Directory containing NPY files. Defaults to the YAML file's directory.",
    ),
    output: Path = typer.Option(
        ..., help="Output .pt path for the packed MotionLib file"
    ),
    output_fps: int = typer.Option(60, help="Target fps (source is ~60fps)"),
    intermediate_dir: Optional[Path] = typer.Option(
        None,
        help="Directory to save per-clip .motion files. Defaults to <output_dir>/clips/",
    ),
    force_remake: bool = typer.Option(False, help="Re-convert clips even if .motion already exists"),
    mjcf_path: str = typer.Option(
        MJCF_PATH, help="MJCF file defining the robot kinematics"
    ),
    poselib_root_name: str = typer.Option(
        POSELIB_ROOT_NAME, help="Root body name in the poselib skeleton"
    ),
    mjcf_root_name: str = typer.Option(
        MJCF_ROOT_NAME, help="Root body name in the MJCF"
    ),
):
    device = torch.device("cpu")
    dtype = torch.float32

    # Resolve directories
    yaml_dir = Path(yaml_file).parent
    if motion_dir is None:
        motion_dir = yaml_dir

    output = Path(output)
    if intermediate_dir is None:
        intermediate_dir = output.parent / "clips"
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(output.parent, exist_ok=True)

    # Load kinematic info from the robot MJCF
    print(f"Loading MJCF from {mjcf_path}")
    kinematic_info = extract_kinematic_info(mjcf_path)
    print(f"Bodies: {kinematic_info.body_names}")
    print(f"DOFs:   {kinematic_info.dof_names}")

    # Load YAML clip list
    with open(yaml_file) as f:
        yaml_data = yaml.safe_load(f)
    entries = yaml_data["motions"]
    print(f"Found {len(entries)} clips in YAML")

    # Per-clip conversion
    converted_clips = []   # list of (motion_file_path, weight)
    skeleton_verified = False

    for entry in tqdm(entries, desc="Converting clips"):
        npy_filename = entry["file"]
        weight = float(entry.get("weight", 1.0))
        npy_path = motion_dir / npy_filename

        motion_filename = npy_filename.replace(".npy", ".motion").replace("/", "_")
        motion_path = intermediate_dir / motion_filename

        if not force_remake and motion_path.exists():
            converted_clips.append((str(motion_path), weight))
            continue

        if not npy_path.exists():
            print(f"  MISSING: {npy_path} — skipping")
            continue

        try:
            rotation_np, root_translation_np, fps, node_names = load_poselib_npy(str(npy_path))

            if not skeleton_verified:
                verify_skeleton_order(
                    node_names,
                    kinematic_info.body_names,
                    poselib_root_name=poselib_root_name,
                    mjcf_root_name=mjcf_root_name,
                )
                skeleton_verified = True

            motion = convert_clip(
                rotation_np=rotation_np,
                root_translation_np=root_translation_np,
                fps=fps,
                kinematic_info=kinematic_info,
                device=device,
                dtype=dtype,
                output_fps=output_fps,
            )

            if motion is None:
                print(f"  TOO SHORT: {npy_filename} — skipping")
                continue

            torch.save(motion.to_dict(), str(motion_path))
            converted_clips.append((str(motion_path), weight))

        except Exception as e:
            print(f"  ERROR: {npy_filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not converted_clips:
        print("No clips converted — aborting.")
        raise SystemExit(1)

    print(f"\nConverted {len(converted_clips)} / {len(entries)} clips")

    # Write intermediate YAML for MotionLib
    intermediate_yaml = output.parent / "clips.yaml"
    clips_yaml_data = {
        "motions": [
            {"file": os.path.relpath(path, start=str(output.parent)), "weight": w}
            for path, w in converted_clips
        ]
    }
    with open(intermediate_yaml, "w") as f:
        yaml.dump(clips_yaml_data, f, default_flow_style=False)
    print(f"Wrote intermediate YAML: {intermediate_yaml}")

    # Pack into single .pt via MotionLib
    print(f"Packing into {output} ...")
    lib = MotionLib(
        config=MotionLibConfig(motion_file=str(intermediate_yaml)),
        device=device,
    )
    lib.save_to_file(str(output))
    print(f"Done. Saved {lib.num_motions()} motions ({lib.get_total_length():.1f}s) to {output}")


if __name__ == "__main__":
    with torch.no_grad():
        app()
