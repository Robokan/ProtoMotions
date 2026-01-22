#!/usr/bin/env python3
"""Extract keypoints from SMPL motion files for Adam Lite retargeting."""

import numpy as np
import torch
import glob
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# SMPL body names (from motion files)
SMPL_BODY_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 
    'L_Ankle', 'R_Ankle', 'Spine3', 'L_Toe', 'R_Toe', 'Neck', 'L_Collar',
    'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
    'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
]

# 18 keypoints: 15 main + 3 auxiliary (matching the original format)
KEYPOINT_MAP = [
    ('pelvis', 'Pelvis'),
    ('left_hip', 'L_Hip'),
    ('right_hip', 'R_Hip'),
    ('left_knee', 'L_Knee'),
    ('right_knee', 'R_Knee'),
    ('left_ankle', 'L_Ankle'),
    ('right_ankle', 'R_Ankle'),
    ('left_foot', 'L_Toe'),
    ('right_foot', 'R_Toe'),
    ('left_shoulder', 'L_Shoulder'),
    ('right_shoulder', 'R_Shoulder'),
    ('left_elbow', 'L_Elbow'),
    ('right_elbow', 'R_Elbow'),
    ('left_wrist', 'L_Wrist'),
    ('right_wrist', 'R_Wrist'),
    # 3 Auxiliary keypoints
    ('left_hand', 'L_Hand'),
    ('right_hand', 'R_Hand'),
    ('head', 'Head'),
]

KEYPOINT_NAMES = [k[0] for k in KEYPOINT_MAP]
KEYPOINT_INDICES = [SMPL_BODY_NAMES.index(k[1]) for k in KEYPOINT_MAP]

# Foot indices
LEFT_ANKLE_IDX = SMPL_BODY_NAMES.index('L_Ankle')
LEFT_TOE_IDX = SMPL_BODY_NAMES.index('L_Toe')
RIGHT_ANKLE_IDX = SMPL_BODY_NAMES.index('R_Ankle')
RIGHT_TOE_IDX = SMPL_BODY_NAMES.index('R_Toe')


def main():
    motion_files = sorted(glob.glob("data/amass_raw/extracted/**/*.motion", recursive=True))
    print(f"Processing {len(motion_files)} motion files...")

    # Clear old files
    output_dir = Path("data/adam_retargeted/keypoints_all")
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in output_dir.glob("*.npy"):
        f.unlink()

    for mf in motion_files:
        basename = Path(mf).stem
        output_file = output_dir / f"{basename}_keypoints.npy"
        
        try:
            motion = torch.load(mf, map_location='cpu', weights_only=False)
            
            body_pos = motion['rigid_body_pos'].numpy()
            body_rot = motion['rigid_body_rot'].numpy()
            contacts = motion['rigid_body_contacts'].numpy()
            
            num_frames = body_pos.shape[0]
            num_keypoints = len(KEYPOINT_INDICES)
            
            # Extract 18 keypoints
            keypoint_pos = body_pos[:, KEYPOINT_INDICES, :]
            keypoint_rot_wxyz = body_rot[:, KEYPOINT_INDICES, :]
            
            # Convert to rotation matrices
            keypoint_rot_matrices = np.zeros((num_frames, num_keypoints, 3, 3), dtype=np.float32)
            for i in range(num_keypoints):
                quat_xyzw = np.concatenate([keypoint_rot_wxyz[:, i, 1:], keypoint_rot_wxyz[:, i, :1]], axis=-1)
                rot = R.from_quat(quat_xyzw)
                keypoint_rot_matrices[:, i, :, :] = rot.as_matrix()
            
            # Foot contacts
            left_foot_contacts = np.stack([contacts[:, LEFT_ANKLE_IDX], contacts[:, LEFT_TOE_IDX]], axis=1)
            right_foot_contacts = np.stack([contacts[:, RIGHT_ANKLE_IDX], contacts[:, RIGHT_TOE_IDX]], axis=1)
            
            # Save (matching original format)
            result = {
                'positions': keypoint_pos.astype(np.float32),
                'orientations': keypoint_rot_matrices.astype(np.float32),
                'left_foot_contacts': left_foot_contacts.astype(np.float32),
                'right_foot_contacts': right_foot_contacts.astype(np.float32),
            }
            
            np.save(output_file, result, allow_pickle=True)
            print(f"  {basename}: {keypoint_pos.shape}")
            
        except Exception as e:
            print(f"  Error {basename}: {e}")

    print(f"\nDone! {len(list(output_dir.glob('*.npy')))} files created")


if __name__ == "__main__":
    main()
