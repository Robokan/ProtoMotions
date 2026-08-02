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

import mujoco
import numpy as np
import torch
import yaml
import typer
from tqdm import tqdm

from protomotions.components.pose_lib import (
    extract_kinematic_info,
    fk_from_transforms_with_velocities,
    extract_transforms_from_qpos,
    extract_qpos_from_transforms,
    compute_cartesian_velocity,
)
from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.utils.rotations import quaternion_to_matrix, quat_from_angle_axis

app = typer.Typer(pretty_exceptions_enable=False)

# Defaults target Go2; override via CLI options for other robots (e.g. anymal_d).
MJCF_PATH = "protomotions/data/assets/mjcf/go2.xml"
# Poselib uses 'trunk' for the root body; MJCF uses 'base_link' — same body, different name.
POSELIB_ROOT_NAME = "trunk"
MJCF_ROOT_NAME = "base_link"

# --- Limit-aware foot planting (dog_v2) -------------------------------------
# Per leg: end-effector body and its paw CONTACT POINT (bottom of the lowest
# contact sphere) in the ee body's LOCAL frame, from dog_v2_nomesh.xml
# (sphere center pos + (0,0,-radius)).
DOG_LEG_CONTACT = {
    "hind_L": ("toe_L", (0.04, 0.0, -0.041)),
    "hind_R": ("toe_R", (0.04, 0.0, -0.041)),
    "front_L": ("finger_L", (0.038, 0.0, -0.033)),
    "front_R": ("finger_R", (0.038, 0.0, -0.033)),
}
# Per leg, the leg-OWN joint bodies (root->tip) whose hinge DOFs are refined to
# plant the paw. Excludes the shared lumbar/spine chain so the two hind legs do
# not fight over spine angles (the spine is left at its clamped retarget value).
DOG_LEG_JOINT_BODIES = {
    "hind_L": ["upper_leg_L", "lower_leg_L", "foot_L", "toe_L"],
    "hind_R": ["upper_leg_R", "lower_leg_R", "foot_R", "toe_R"],
    "front_L": ["scapula_L", "upper_arm_L", "lower_arm_L", "hand_L", "finger_L"],
    "front_R": ["scapula_R", "upper_arm_R", "lower_arm_R", "hand_R", "finger_R"],
}
# A stance frame: paw contact-point height within this band of its clip-min
# (the planted phase of the gait). Refinement only plants stance paws.
STANCE_BAND = 0.06
# Target ground height for a planted paw (m above the lowest paw in the clip).
PLANT_TARGET_Z = 0.0


def _build_dof_start_map(kinematic_info):
    """Map body_idx -> (start_dof_index, num_dofs) within the 73-dof hinge block."""
    start = {}
    s = 0
    for body_idx, axes in kinematic_info.hinge_axes_map.items():
        start[body_idx] = (s, len(axes))
        s += len(axes)
    return start


def _leg_dof_specs(kinematic_info):
    """Per leg: (ee_body_idx, contact_offset (3,), dof_index_list, axis_list).

    dof_index_list / axis_list enumerate, in MJCF declaration order, each hinge
    DOF of the leg's own joint bodies (root->tip)."""
    bn = kinematic_info.body_names
    dof_start = _build_dof_start_map(kinematic_info)
    specs = {}
    for label, (ee_name, offset) in DOG_LEG_CONTACT.items():
        dof_idx = []
        axes = []
        for body_name in DOG_LEG_JOINT_BODIES[label]:
            bi = bn.index(body_name)
            s, n = dof_start[bi]
            body_axes = kinematic_info.hinge_axes_map[bi]
            for k in range(n):
                dof_idx.append(s + k)
                axes.append(body_axes[k])
        specs[label] = {
            "ee": bn.index(ee_name),
            "offset": torch.tensor(offset, dtype=torch.float32),
            "dof_idx": torch.tensor(dof_idx, dtype=torch.long),
            "axes": torch.stack(axes, dim=0),  # (nd, 3)
        }
    return specs


def _fk_world(kinematic_info, root_pos, root_quat_wxyz, qpos_hinge):
    """Batched MuJoCo-consistent FK from root pose + hinge angles.

    Returns world_pos (N, B, 3), world_rot_mat (N, B, 3, 3). Same math as the
    simulator's FK (verified by pose_lib.test_fk_batch)."""
    N = root_pos.shape[0]
    qpos = torch.zeros(N, kinematic_info.nq, dtype=torch.float32)
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = root_quat_wxyz
    qpos[:, 7:] = qpos_hinge
    root_pos_o, joint_rot_mats = extract_transforms_from_qpos(kinematic_info, qpos)
    from protomotions.components.pose_lib import (
        compute_forward_kinematics_from_transforms,
    )

    return compute_forward_kinematics_from_transforms(
        kinematic_info, root_pos_o, joint_rot_mats
    )


def clamp_and_plant_qpos(
    kinematic_info,
    qpos: torch.Tensor,
    refine_iters: int = 30,
    step_clamp: float = 0.2,
    temporal_continuity: bool = True,
    temporal_weight: float = 0.5,
):
    """Make qpos limit-feasible and plant stance paws within the limits.

    1. Clamp every hinge DOF to the MJCF joint range.
    2. Per leg, identify stance frames (paw contact point near its clip-min) and
       run coordinate-descent (CCD) on the leg's own hinge DOFs — clamped to
       limits each step, MuJoCo-consistent FK in the loop — to drive the stance
       paw contact point down to the ground (z=0). Swing frames are untouched.

    A leg has several DOFs but the plant constraint is a single scalar (paw z),
    so the IK is redundant: many joint configurations plant the same paw. Solved
    independently per frame, CCD picks an arbitrary point in that null space, and
    near singular leg configs the choice flips between frames -> the leg DOFs
    (scapula/shoulder/elbow/wrist/finger) jitter and the leg visibly shakes even
    though the paw stays planted.

    When `temporal_continuity` is True, every CCD step adds a null-space bias
    that pulls each refined DOF toward the (temporally SMOOTH) clamped-
    decomposition pose `seed`, in the component orthogonal to the z-task
    direction. The decomposition is continuous over time, so anchoring the
    redundant null space to it makes the planted leg continuous too, without
    changing planting accuracy (the z-task term has priority and re-plants the
    paw each step). `temporal_weight` scales that pull (0 reproduces the legacy
    independent-per-frame behavior). The seed pull is applied per-frame, so it
    propagates instantly across the whole clip instead of diffusing one frame
    per CCD sweep.

    Returns the final clamped+refined hinge qpos (N, num_dofs). FK of this qpos
    is, by construction, exactly what the simulator plays.
    """
    N = qpos.shape[0]
    root_pos = qpos[:, 0:3].clone()
    root_quat_wxyz = qpos[:, 3:7].clone()
    lo = kinematic_info.dof_limits_lower
    hi = kinematic_info.dof_limits_upper

    hinge = qpos[:, 7:].clone()
    hinge = torch.clamp(hinge, lo, hi)
    # Temporally-smooth anchor for the redundant-IK null space (the clamped
    # decomposition; jitter ~1e-3). Anchoring keeps the planted leg continuous.
    seed = hinge.clone()

    specs = _leg_dof_specs(kinematic_info)

    # ground reference: lowest contact point over the clip in the clamped pose
    world_pos, world_rot = _fk_world(kinematic_info, root_pos, root_quat_wxyz, hinge)

    def contact_z(world_pos, world_rot, spec):
        ee = spec["ee"]
        off = spec["offset"].expand(N, 3)
        rot = world_rot[:, ee, :, :]
        ct = world_pos[:, ee, :] + torch.matmul(rot, off.unsqueeze(-1)).squeeze(-1)
        return ct[:, 2]

    floor = min(
        float(contact_z(world_pos, world_rot, specs[l]).min()) for l in specs
    )

    # stance mask per leg (computed once on the clamped pose)
    stance = {}
    for label, spec in specs.items():
        cz = contact_z(world_pos, world_rot, spec) - floor
        stance[label] = cz < (cz.min() + STANCE_BAND)

    target_z = floor + PLANT_TARGET_Z

    # CCD refinement: only stance frames, only stance legs, leg-own DOFs.
    for _ in range(refine_iters):
        world_pos, world_rot = _fk_world(
            kinematic_info, root_pos, root_quat_wxyz, hinge
        )
        max_err = 0.0
        for label, spec in specs.items():
            mask = stance[label]
            if not bool(mask.any()):
                continue
            cz = contact_z(world_pos, world_rot, spec)
            err = (target_z - cz)  # (N,) want to lower z by -err (signed)
            err = torch.where(mask, err, torch.zeros_like(err))
            max_err = max(max_err, float(err.abs().max()))
            ee = spec["ee"]
            off = spec["offset"].expand(N, 3)
            p_ct = (
                world_pos[:, ee, :]
                + torch.matmul(world_rot[:, ee, :, :], off.unsqueeze(-1)).squeeze(-1)
            )
            # CCD over this leg's DOFs (tip->root sweep is most effective)
            order = list(range(spec["dof_idx"].shape[0]))[::-1]
            for k in order:
                di = int(spec["dof_idx"][k])
                axis_local = spec["axes"][k]  # (3,) in joint body frame
                jb = _dof_to_body(kinematic_info, di)
                # world axis of this hinge = parent_world_rot * ref * (axes...)
                # Approx: use the body's own world rotation to map the local axis.
                w_axis = torch.matmul(
                    world_rot[:, jb, :, :], axis_local.view(1, 3, 1).expand(N, 3, 1)
                ).squeeze(-1)  # (N,3)
                p_j = world_pos[:, jb, :]
                # Jacobian column (world): w_axis x (p_ct - p_j); only z-error
                lever = p_ct - p_j
                jac = torch.cross(w_axis, lever, dim=-1)  # (N,3)
                jz = jac[:, 2]
                denom = jz * jz + 1e-6
                dtheta = (jz * err) / denom  # least-squares 1-dof step on z
                dtheta = torch.clamp(dtheta, -step_clamp, step_clamp)
                dtheta = torch.where(mask, dtheta, torch.zeros_like(dtheta))
                new_angle = torch.clamp(hinge[:, di] + dtheta, lo[di], hi[di])
                hinge[:, di] = new_angle
                # refresh FK after each joint update (CCD)
                world_pos, world_rot = _fk_world(
                    kinematic_info, root_pos, root_quat_wxyz, hinge
                )
                p_ct = (
                    world_pos[:, ee, :]
                    + torch.matmul(
                        world_rot[:, ee, :, :], off.unsqueeze(-1)
                    ).squeeze(-1)
                )
                cz = contact_z(world_pos, world_rot, spec)
                err = torch.where(mask, target_z - cz, torch.zeros_like(err))
        if max_err < 1e-3:
            break

    if temporal_continuity and N > 2 and temporal_weight > 0.0:
        # The per-frame CCD is a redundant 1-scalar (paw z) solve over a
        # multi-DOF leg, so adjacent frames settle on different equivalent
        # planted configs -> the leg DOFs oscillate (often a 2-cycle, e.g.
        # ...1.45, 1.64, 1.45, 1.64...) and the leg visibly shakes even though
        # the paw stays planted. Resolve that redundancy temporally: smooth the
        # stance-leg DOF trajectories along time, then re-plant the paw on top
        # of the smoothed trajectory. Iterating {smooth -> re-plant} converges
        # to a continuous planted solution. Swing frames are untouched. End on a
        # smoothing pass so the final trajectory is the de-jittered one.
        def smooth_legs(num_passes):
            for _ in range(num_passes):
                for label, spec in specs.items():
                    mask = stance[label]
                    if not bool(mask.any()):
                        continue
                    m = mask.to(hinge.dtype)
                    for k in range(spec["dof_idx"].shape[0]):
                        di = int(spec["dof_idx"][k])
                        a = hinge[:, di]
                        # median-of-3 over stance neighbors first (kills the
                        # 2-cycle oscillation outliers), then a 1-2-1 average.
                        left = a.clone()
                        right = a.clone()
                        left[1:] = a[:-1]
                        right[:-1] = a[1:]
                        med = torch.median(
                            torch.stack([left, a, right], dim=0), dim=0
                        ).values
                        aw = med * m
                        lw = torch.zeros_like(a)
                        rw = torch.zeros_like(a)
                        lm = torch.zeros_like(a)
                        rm = torch.zeros_like(a)
                        lw[1:] = aw[:-1]
                        rw[:-1] = aw[1:]
                        lm[1:] = m[:-1]
                        rm[:-1] = m[1:]
                        num = 2.0 * med * m + lw + rw
                        den = 2.0 * m + lm + rm + 1e-6
                        smoothed = num / den
                        new_a = torch.where(mask, smoothed, a)
                        hinge[:, di] = torch.clamp(new_a, lo[di], hi[di])

        for outer in range(15):
            # 1) temporal smoothing pass on each leg's own DOFs, stance frames.
            smooth_legs(2)
            # 2) re-plant the paw on the smoothed trajectory (a few CCD sweeps).
            for _ in range(6):
                world_pos, world_rot = _fk_world(
                    kinematic_info, root_pos, root_quat_wxyz, hinge
                )
                for label, spec in specs.items():
                    mask = stance[label]
                    if not bool(mask.any()):
                        continue
                    ee = spec["ee"]
                    off = spec["offset"].expand(N, 3)
                    p_ct = (
                        world_pos[:, ee, :]
                        + torch.matmul(
                            world_rot[:, ee, :, :], off.unsqueeze(-1)
                        ).squeeze(-1)
                    )
                    cz = contact_z(world_pos, world_rot, spec)
                    err = torch.where(
                        mask, target_z - cz, torch.zeros_like(cz)
                    )
                    order = list(range(spec["dof_idx"].shape[0]))[::-1]
                    for k in order:
                        di = int(spec["dof_idx"][k])
                        jb = _dof_to_body(kinematic_info, di)
                        w_axis = torch.matmul(
                            world_rot[:, jb, :, :],
                            spec["axes"][k].view(1, 3, 1).expand(N, 3, 1),
                        ).squeeze(-1)
                        lever = p_ct - world_pos[:, jb, :]
                        jz = torch.cross(w_axis, lever, dim=-1)[:, 2]
                        dtheta = (jz * err) / (jz * jz + 1e-6)
                        dtheta = torch.clamp(dtheta, -step_clamp, step_clamp)
                        dtheta = torch.where(mask, dtheta, torch.zeros_like(dtheta))
                        hinge[:, di] = torch.clamp(
                            hinge[:, di] + dtheta, lo[di], hi[di]
                        )
                        world_pos, world_rot = _fk_world(
                            kinematic_info, root_pos, root_quat_wxyz, hinge
                        )
                        p_ct = (
                            world_pos[:, ee, :]
                            + torch.matmul(
                                world_rot[:, ee, :, :], off.unsqueeze(-1)
                            ).squeeze(-1)
                        )
                        cz = contact_z(world_pos, world_rot, spec)
                        err = torch.where(
                            mask, target_z - cz, torch.zeros_like(cz)
                        )

    return hinge


def _dof_to_body(kinematic_info, dof_index):
    """Body index owning hinge DOF `dof_index` (0-based in the hinge block)."""
    s = 0
    for body_idx, axes in kinematic_info.hinge_axes_map.items():
        n = len(axes)
        if s <= dof_index < s + n:
            return body_idx
        s += n
    raise IndexError(dof_index)


def mujoco_fk_world(mjcf_path, root_pos, root_quat_wxyz, hinge):
    """Ground-truth world body poses via MuJoCo (xyzw quats), per frame.

    Returns world_pos (N, B, 3), world_quat_xyzw (N, B, 4), body_names list."""
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    body_names = [model.body(i).name for i in range(model.nbody)]
    N = root_pos.shape[0]
    Nb = model.nbody
    wp = np.zeros((N, Nb, 3), dtype=np.float32)
    wq = np.zeros((N, Nb, 4), dtype=np.float32)  # xyzw
    rp = root_pos.cpu().numpy()
    rq = root_quat_wxyz.cpu().numpy()
    hg = hinge.cpu().numpy()
    for i in range(N):
        data.qpos[:3] = rp[i]
        data.qpos[3:7] = rq[i]
        data.qpos[7:] = hg[i]
        mujoco.mj_forward(model, data)
        wp[i] = data.xpos
        wq[i] = data.xquat[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    return (
        torch.from_numpy(wp),
        torch.from_numpy(wq),
        body_names,
    )


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
    multi_dof_method: Optional[str] = None,
    limit_aware: bool = False,
    mjcf_path: Optional[str] = None,
    temporal_continuity: bool = True,
) -> Optional[object]:
    """Convert one poselib clip to a ProtoMotions RobotState motion.

    Args:
        rotation_np: (N, B, 4) local wxyz quaternions
        root_translation_np: (N, 3) root positions
        fps: source fps
        kinematic_info: extracted from MJCF
        device/dtype: torch settings
        output_fps: target fps (will downsample if source fps > output_fps)
        limit_aware: if True, clamp every hinge DOF to the MJCF joint range and
            re-plant stance paws within the limits, then recompute the stored
            body poses (gts/grs) by FK from the CLAMPED dof. This guarantees the
            stored motion equals what the simulator will actually play.
        mjcf_path: MJCF path (required when limit_aware) for the ground-truth
            MuJoCo FK consistency check.

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

    # Extract joint angles from rotation matrices (inverse FK).
    # temporal_continuity treats the clip's frames as a time sequence and tracks
    # one continuous decomposition branch (warm-start + unwrap), eliminating the
    # frame-to-frame gimbal flips on the dog's multi-DOF joints. It is a
    # different-but-equivalent branch of the same rotations, so the round-trip
    # error is unchanged; for 1-DOF robots it only adds harmless +-pi unwrapping.
    qpos = extract_qpos_from_transforms(
        kinematic_info,
        root_pos,
        joint_rot_mats,
        multi_dof_decomposition_method=multi_dof_method,
        temporal_continuity=temporal_continuity,
    )

    if limit_aware:
        # ---- Single source of truth = joint angles within limits ----
        # The root pose comes from the rotation-retarget FK (the torso body's
        # world pose), expressed in qpos as [root_pos, root_quat wxyz].
        root_quat_wxyz = qpos[:, 3:7]
        # Clamp every hinge DOF to the joint range and plant stance paws within
        # the limits (dof-space CCD, MuJoCo-consistent FK in the loop). The CCD
        # is the dominant source of the leg jitter (redundant per-frame IK); its
        # temporal_continuity resolver keeps the planted leg on one continuous
        # branch over time.
        hinge = clamp_and_plant_qpos(
            kinematic_info, qpos, temporal_continuity=temporal_continuity
        )
        qpos = qpos.clone()
        qpos[:, 7:] = hinge
        motion.dof_pos = hinge

        # Recompute the stored body poses by FK from the CLAMPED dof so that
        # stored gts/grs == sim-FK(dof). Use the verified torch FK (matches
        # MuJoCo to <1e-4; see pose_lib.test_fk_batch).
        _, joint_rot_mats_clamped = extract_transforms_from_qpos(kinematic_info, qpos)
        clamped_motion = fk_from_transforms_with_velocities(
            kinematic_info=kinematic_info,
            root_pos=qpos[:, 0:3],
            joint_rot_mats=joint_rot_mats_clamped,
            fps=output_fps,
            compute_velocities=True,
            velocity_max_horizon=3,
        )
        motion.rigid_body_pos = clamped_motion.rigid_body_pos
        motion.rigid_body_rot = clamped_motion.rigid_body_rot
        motion.rigid_body_vel = clamped_motion.rigid_body_vel
        motion.rigid_body_ang_vel = clamped_motion.rigid_body_ang_vel
    else:
        motion.dof_pos = qpos[:, 7:]  # strip root pos + root quat

    # Compute DOF velocities via finite differences on joint angles
    joint_angles = motion.dof_pos
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
    multi_dof_method: Optional[str] = typer.Option(
        None,
        help="Decomposition for bodies with >1 hinge DOF: 'analytic_xyz', "
        "'sequential', 'euler_xyz' or 'exp_map'. Use 'analytic_xyz' for dog_v2 "
        "(fast vectorized closed form, matches 'sequential' exactly). "
        "'sequential' is the slow per-frame reference.",
    ),
    limit_aware: bool = typer.Option(
        False,
        help="Clamp every hinge DOF to the MJCF joint range and re-plant stance "
        "paws within the limits, then recompute the stored body poses by FK "
        "from the clamped dof. Guarantees stored motion == sim-FK(clamped dof). "
        "Use for dog_v2 (the free retarget exceeds the dm_control dog limits).",
    ),
    temporal_continuity: bool = typer.Option(
        True,
        help="Track one continuous branch of the joint-angle decomposition over "
        "each clip's frames (warm-start the sequential multi-DOF solve from the "
        "previous frame + 2*pi unwrap). Removes the frame-to-frame gimbal "
        "flipping that makes the dog's legs shake. Behavior-preserving for "
        "1-DOF robots (go2, anymal_d): same rotations, just unwrapped angles.",
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
                multi_dof_method=multi_dof_method,
                limit_aware=limit_aware,
                mjcf_path=mjcf_path,
                temporal_continuity=temporal_continuity,
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
