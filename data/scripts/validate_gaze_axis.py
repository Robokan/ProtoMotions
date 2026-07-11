"""Ground-truth gaze-axis validation: humans walk face-forward.

Over SEED locomotion clips, the head-frame gaze axis is correct iff it
aligns with the walking direction. Reports mean alignment for -y (the fix)
and +x (the old calc_heading convention) in the HEAD body frame.
"""

import glob
import torch

from protomotions.components.pose_lib import extract_kinematic_info
from protomotions.utils import rotations

ki = extract_kinematic_info("protomotions/data/assets/mjcf/soma23_humanoid.xml")
head_idx = ki.body_names.index("Head")

clips = sorted(glob.glob("/workspace/sparkpack/bones-seed/motions/breadth/*walk*.motion"))[:40]
print(f"validating over {len(clips)} walking clips")

dots_neg_y, dots_pos_x, frames_used = [], [], 0
for path in clips:
    d = torch.load(path, map_location="cpu", weights_only=False)
    pos = d["rigid_body_pos"]  # [T, 23, 3] z-up
    rot = d["rigid_body_rot"]  # [T, 23, 4] w-last
    fps = float(d["fps"])
    root_vel_xy = (pos[1:, 0, :2] - pos[:-1, 0, :2]) * fps
    speed = root_vel_xy.norm(dim=-1)
    moving = speed > 0.5  # only clearly-walking frames
    if moving.sum() < 10:
        continue
    walk_dir = torch.nn.functional.normalize(root_vel_xy[moving], dim=-1)
    head_rot = rot[:-1][moving][:, head_idx]

    for axis, sink in ((torch.tensor([0.0, -1.0, 0.0]), dots_neg_y),
                       (torch.tensor([1.0, 0.0, 0.0]), dots_pos_x)):
        gaze = rotations.quat_rotate(head_rot, axis.expand(len(head_rot), 3), True)
        gaze_xy = torch.nn.functional.normalize(gaze[:, :2], dim=-1)
        sink.append((gaze_xy * walk_dir).sum(dim=-1))
    frames_used += int(moving.sum())

neg_y = torch.cat(dots_neg_y)
pos_x = torch.cat(dots_pos_x)
print(f"frames: {frames_used}")
print(f"head-frame -y vs walk direction: mean dot = {neg_y.mean():.3f} (correct axis => close to +1)")
print(f"head-frame +x vs walk direction: mean dot = {pos_x.mean():.3f} (old axis: sideways => close to 0)")
