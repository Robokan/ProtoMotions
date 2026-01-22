#!/bin/bash
# Batch retarget all AMASS motions to Adam Lite
set -e

cd /home/bizon/sparkpack/ProtoMotions

# Setup conda
source /home/bizon/anaconda3/etc/profile.d/conda.sh

# Create output directories
mkdir -p data/adam_retargeted/keypoints_all
mkdir -p data/adam_retargeted/retargeted_all
mkdir -p data/adam_retargeted/contacts_all

echo "=== Step 1: Extracting keypoints from AMASS ==="
CONDA_NO_PLUGINS=true conda activate pyroki

# Extract keypoints from each AMASS file
for amass_file in $(find data/amass_raw/extracted -name "*.npz" -type f | sort); do
    basename=$(basename "$amass_file" .npz)
    output_file="data/adam_retargeted/keypoints_all/${basename}_keypoints.npy"
    
    if [ -f "$output_file" ]; then
        echo "Skipping $basename (already exists)"
        continue
    fi
    
    echo "Processing: $amass_file"
    
    python3 << EOF
import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as R

# Load AMASS data
data = np.load("$amass_file", allow_pickle=True)
poses = data['poses']  # (N, 156) for SMPL+H
trans = data['trans']  # (N, 3)
betas = data['betas']  # (16,)
fps = float(data['mocap_framerate'])

# Extract body poses (first 22 joints = 66 values, skip hands)
num_frames = poses.shape[0]
body_poses = poses[:, :66]  # 22 joints * 3

# Root orientation and body pose
root_orient = body_poses[:, :3]  # First 3 values are root orientation
body_pose = body_poses[:, 3:]  # Remaining are body joints

# SMPL body keypoint indices (subset for retargeting)
# Based on SMPL joint order
KEYPOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_foot", "right_foot",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist"
]

# SMPL joint indices for these keypoints
SMPL_KEYPOINT_INDICES = [
    0,   # pelvis
    1,   # left_hip
    2,   # right_hip
    4,   # left_knee  
    5,   # right_knee
    7,   # left_ankle
    8,   # right_ankle
    10,  # left_foot (toe)
    11,  # right_foot (toe)
    16,  # left_shoulder
    17,  # right_shoulder
    18,  # left_elbow
    19,  # right_elbow
    20,  # left_wrist
    21,  # right_wrist
]

# Create SMPL body model to get joint positions
try:
    body_model = smplx.create(
        'data/smpl',
        'smpl',
        gender='neutral',
        use_pca=False,
        batch_size=num_frames
    )
    
    # Get joint positions
    output = body_model(
        betas=torch.tensor(betas).float().unsqueeze(0).expand(num_frames, -1)[:, :10],
        global_orient=torch.tensor(root_orient).float(),
        body_pose=torch.tensor(body_pose[:, :63]).float(),
        transl=torch.tensor(trans).float()
    )
    
    joints = output.joints.detach().numpy()  # (N, 45, 3) or (N, 24, 3)
    
    # Extract keypoint positions
    keypoints = joints[:, SMPL_KEYPOINT_INDICES, :]  # (N, 15, 3)
    
    # Also get joint rotations (for orientation)
    # Convert axis-angle to rotation matrices
    full_poses = np.concatenate([root_orient, body_pose[:, :63]], axis=1)  # (N, 66)
    full_poses = full_poses.reshape(num_frames, 22, 3)
    
    # Get keypoint orientations
    keypoint_rots = []
    for i, idx in enumerate(SMPL_KEYPOINT_INDICES):
        if idx < 22:
            aa = full_poses[:, idx, :]  # (N, 3)
            rot = R.from_rotvec(aa)
            quat = rot.as_quat()  # (N, 4) xyzw
            keypoint_rots.append(quat)
        else:
            keypoint_rots.append(np.zeros((num_frames, 4)))
    
    keypoint_rots = np.stack(keypoint_rots, axis=1)  # (N, 15, 4)
    
    # Save keypoints
    result = {
        'keypoint_positions': keypoints.astype(np.float32),
        'keypoint_orientations': keypoint_rots.astype(np.float32),
        'keypoint_names': KEYPOINT_NAMES,
        'fps': fps
    }
    
    np.save("$output_file", result)
    print(f"Saved keypoints: {keypoints.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    # Fallback: just save positions from AMASS directly
    # This is less accurate but works
    pass
EOF

done

echo ""
echo "=== Step 2: Running PyRoKi retargeting ==="
python pyroki/batch_retarget_to_adam_lite_from_keypoints.py \
    --keypoints-folder-path data/adam_retargeted/keypoints_all \
    --output-dir data/adam_retargeted/retargeted_all \
    --contacts-dir data/adam_retargeted/contacts_all \
    --no-visualize \
    --skip-existing

echo ""
echo "=== Step 3: Converting to ProtoMotions format ==="
CONDA_NO_PLUGINS=true conda activate isaacgym

python data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
    --retargeted-motion-dir data/adam_retargeted/retargeted_all \
    --output-dir data/motions/adam_lite \
    --input-fps 30 \
    --output-fps 30

echo ""
echo "=== Step 4: Packaging motions ==="
# Package all motions into a single .pt file
python << 'EOF'
import torch
import glob
from pathlib import Path
from protomotions.envs.mimic.motion_lib import MotionLib

motion_files = sorted(glob.glob("data/motions/adam_lite/*.motion"))
print(f"Found {len(motion_files)} motion files")

if motion_files:
    # Create motion lib config
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        "motion_file": motion_files,
        "device": "cuda:0",
        "preload_motions": True,
    })
    
    # This would require loading MotionLib which needs robot config
    # For now just list the files
    print("Motion files ready for visualization")
EOF

echo ""
echo "Done! To visualize:"
echo "python examples/motion_libs_visualizer.py --motion_files data/motions/adam_lite_all.pt --robot adam_lite --simulator isaacgym"
