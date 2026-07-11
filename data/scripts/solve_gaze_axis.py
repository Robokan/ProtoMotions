"""Solve for the empirical face axis: E[head_rot^-1 @ walk_dir] per clip.

Prints the optimal head-local axis per clip (multimodal if backward walks
are mixed in) and the global consensus.
"""

import glob
import torch

from protomotions.components.pose_lib import extract_kinematic_info
from protomotions.utils import rotations

ki = extract_kinematic_info("protomotions/data/assets/mjcf/soma23_humanoid.xml")
head_idx = ki.body_names.index("Head")

clips = sorted(glob.glob("/workspace/sparkpack/bones-seed/motions/breadth/*walk*.motion"))[:40]

per_clip = []
for path in clips:
    d = torch.load(path, map_location="cpu", weights_only=False)
    pos, rot, fps = d["rigid_body_pos"], d["rigid_body_rot"], float(d["fps"])
    root_vel_xy = (pos[1:, 0, :2] - pos[:-1, 0, :2]) * fps
    speed = root_vel_xy.norm(dim=-1)
    moving = speed > 0.5
    if moving.sum() < 10:
        continue
    walk3 = torch.zeros(int(moving.sum()), 3)
    walk3[:, :2] = torch.nn.functional.normalize(root_vel_xy[moving], dim=-1)
    head_rot = rot[:-1][moving][:, head_idx]
    local = rotations.quat_rotate_inverse(head_rot, walk3, True)
    axis = torch.nn.functional.normalize(local.mean(dim=0), dim=0)
    per_clip.append((path.split("/")[-1][:44], [round(float(v), 2) for v in axis],
                     round(float(local.mean(dim=0).norm()), 2)))

for name, axis, conf in per_clip[:14]:
    print(f"{name:46s} axis={axis} confidence={conf}")

axes = torch.tensor([a for _, a, _ in per_clip], dtype=torch.float)
consensus = torch.nn.functional.normalize(axes.mean(dim=0), dim=0)
print("\nconsensus head-local face axis:", [round(float(v), 3) for v in consensus])
