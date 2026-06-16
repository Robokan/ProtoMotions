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
"""Smoke validation of the BVH -> dog_v2 retarget + conversion pipeline.

Retargets a couple of clips (default: 0.bvh and 37.bvh, a walk), runs the
full ProtoMotions conversion path (poselib NPY -> .motion -> packed .pt) on
them with the dog_v2_nomesh MJCF, and asserts:
  - all values finite
  - stance foot heights near the ground
  - root (torso) height around dog standing height (~0.4-0.5 m)

No GUI, no IsaacLab. Run from the repo root:
    python data/scripts/verify_dog_retarget.py [--clips 0,37]
"""

import argparse
import os
import subprocess
import sys

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MJCF = "protomotions/data/assets/mjcf/dog_v2_nomesh.xml"
DEFAULT_BVH_DIR = "/home/bizon/eric/Mode Adaptive/mocap"
FOOT_BODIES = ["toe_L", "toe_R", "finger_L", "finger_R"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh-dir", default=DEFAULT_BVH_DIR)
    parser.add_argument("--clips", default="0,37")
    parser.add_argument("--work-dir", default="data/motions/dog_v2/verify")
    args = parser.parse_args()

    clips = [c.strip() for c in args.clips.split(",")]
    npy_dir = os.path.join(args.work_dir, "npy")
    out_pt = os.path.join(args.work_dir, "dog_v2_verify.pt")
    os.makedirs(npy_dir, exist_ok=True)

    python = sys.executable

    # 1. retarget
    subprocess.run(
        [
            python,
            "data/scripts/retarget_bvh_to_dog.py",
            "--bvh-dir",
            args.bvh_dir,
            "--output-dir",
            npy_dir,
            "--clips",
            ",".join(clips),
        ],
        check=True,
    )

    # 2. conversion path (NPY -> .motion -> packed .pt)
    yaml_path = os.path.join(npy_dir, "verify.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(
            {"motions": [{"file": f"{c}.npy", "weight": 1.0} for c in clips]}, f
        )

    subprocess.run(
        [
            python,
            "data/scripts/convert_quadruped_poselib_to_proto.py",
            "--yaml-file",
            yaml_path,
            "--motion-dir",
            npy_dir,
            "--output",
            out_pt,
            "--mjcf-path",
            MJCF,
            "--poselib-root-name",
            "torso",
            "--mjcf-root-name",
            "torso",
            "--multi-dof-method",
            "sequential",
            "--force-remake",
        ],
        check=True,
    )

    # 3. checks on the per-clip .motion files
    from protomotions.components.pose_lib import extract_kinematic_info

    ki = extract_kinematic_info(MJCF)
    foot_ids = [ki.body_names.index(b) for b in FOOT_BODIES]
    torso_id = ki.body_names.index("torso")

    clips_dir = os.path.join(args.work_dir, "clips")
    failures = []
    print("\n=== verification ===")
    for clip in clips:
        motion_path = os.path.join(clips_dir, f"{clip}.motion")
        data = torch.load(motion_path, weights_only=False)
        pos = data["rigid_body_pos"]  # (N, B, 3)
        dof_pos = data["dof_pos"]

        finite = all(
            torch.isfinite(v).all().item()
            for v in data.values()
            if isinstance(v, torch.Tensor)
        )

        root_z = pos[:, torso_id, 2]
        print(f"clip {clip}: {pos.shape[0]} frames, finite={finite}")
        print(
            f"  root (torso) z: mean={root_z.mean():.3f} "
            f"min={root_z.min():.3f} max={root_z.max():.3f}"
        )
        if not finite:
            failures.append(f"clip {clip}: non-finite values")
        if not (0.25 < root_z.mean() < 0.65):
            failures.append(f"clip {clip}: implausible root height {root_z.mean():.3f}")

        for name, bid in zip(FOOT_BODIES, foot_ids):
            z = pos[:, bid, 2]
            stance_z = z.quantile(0.1)  # stance feet ~ lowest 10%
            print(
                f"  {name:9s} z: stance(10pct)={stance_z:.3f} "
                f"min={z.min():.3f} max={z.max():.3f}"
            )
            if not (-0.05 < stance_z < 0.15):
                failures.append(
                    f"clip {clip}: {name} stance height {stance_z:.3f} not near ground"
                )

        if dof_pos.abs().max() > 3.5:
            failures.append(
                f"clip {clip}: extreme dof_pos {dof_pos.abs().max():.2f} rad"
            )
        print(f"  dof_pos: max |angle| = {dof_pos.abs().max():.2f} rad")

    # 4. packed .pt loads
    assert os.path.exists(out_pt), f"missing packed output {out_pt}"
    print(f"\npacked MotionLib file: {out_pt} ({os.path.getsize(out_pt)//1024} KiB)")

    if failures:
        print("\nFAILURES:")
        for fmsg in failures:
            print(f"  - {fmsg}")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
