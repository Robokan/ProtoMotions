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
"""IDENTITY retarget: dog BVH mocap -> the BVH-matched dog_v2 skeleton.

The new dog_v2_nomesh.xml skeleton (data/scripts/generate_dog_mjcf.py) is the
BVH joint tree 1:1 -- same body hierarchy, same offsets, same DOF -- with the
root body renamed Hips -> 'trunk'. The retarget is therefore a near-identity
COPY of the BVH local rotations:

  * Non-root local rotations are parent-frame-relative and frame-invariant
    (independent of the Y-up vs Z-up world convention), so they are copied
    VERBATIM to the matching body. No reference-pose delta, no chain slerp,
    no IK, no spine distribution.
  * The ROOT (Hips/trunk) local rotation is the only one expressed in the world
    frame, so it is rotated Y-up -> Z-up by LEFT-multiplying YUP_TO_ZUP.
  * Root translation = the Hips position, rotated Y-up -> Z-up, then scaled
    cm -> m (CM_TO_M).

BVH End Sites are not bodies in the target, so they are dropped. --mirror swaps
L/R joints and reflects across the sagittal plane. Output is a poselib
SkeletonMotion NPY whose skeleton_tree node order matches the MJCF body order.

Usage:
    python data/scripts/retarget_bvh_to_dog.py \
        --bvh-dir "/home/bizon/eric/Mode Adaptive/mocap" \
        --output-dir /tmp/dognew/npy [--clips 0,37] [--mirror]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poselib_vendor import (  # noqa: E402
    Skeleton,
    quat_mul,
    quat_normalize,
    quat_rotate,
    save_skeleton_motion_npy,
    parse_bvh_file,
)

DEFAULT_BVH_DIR = "/home/bizon/eric/Mode Adaptive/mocap"
DEFAULT_MJCF = "protomotions/data/assets/mjcf/dog_v2_nomesh.xml"

# Standing-height scale cm -> m (matches generate_dog_mjcf.CM_TO_M).
CM_TO_M = 0.01018

# Y-up (BVH) -> Z-up (sim): +90 deg about X, wxyz.
YUP_TO_ZUP = torch.tensor([0.7071068, 0.7071068, 0.0, 0.0])

ROOT_BVH_NAME = "Hips"
ROOT_BODY_NAME = "trunk"

# BVH left/right joint name swap for --mirror (real joints only).
MIRROR_PAIRS = [
    ("LeftShoulder", "RightShoulder"),
    ("LeftArm", "RightArm"),
    ("LeftForeArm", "RightForeArm"),
    ("LeftHand", "RightHand"),
    ("LeftUpLeg", "RightUpLeg"),
    ("LeftLeg", "RightLeg"),
    ("LeftFoot", "RightFoot"),
]


def load_bvh_real_joints(path):
    """Parse a BVH and drop End Sites.

    Returns (names, parents(list), offsets_m(J,3), local_rot(N,J,4) wxyz,
             root_trans_cm(N,3), fps) over the REAL joints only."""
    names, parents, root_trans, offsets, local_rot, fps = parse_bvh_file(path)
    parents = parents.tolist()
    real = [i for i, n in enumerate(names) if not n.endswith("_end")]
    old_to_new = {old: k for k, old in enumerate(real)}
    r_names = [names[i] for i in real]
    r_parents = [old_to_new[parents[i]] if parents[i] >= 0 else -1 for i in real]
    r_offsets = offsets[real] * CM_TO_M
    r_local = local_rot[:, real, :].clone()
    return r_names, r_parents, r_offsets, r_local, root_trans, fps


def target_skeleton(mjcf_path):
    return Skeleton.from_mjcf(mjcf_path)


def mirror_local(names, local_rot, root_trans):
    """Mirror across the sagittal plane (BVH world Z is the lateral axis here:
    left/right offsets differ in the Z component). We reflect the local
    rotations and swap L/R joints.

    A local rotation q=(w,x,y,z) under a reflection that negates the lateral
    axis (z) becomes (w, -x, -y, z): a rotation's axis components ALONG the
    mirrored plane normal keep sign, the two in-plane axis components flip
    (equivalently conjugate-then-negate the normal component). For the BVH bone
    frame the lateral axis is the body-local z, so the in-plane axes are x,y."""
    idx = {n: i for i, n in enumerate(names)}
    perm = list(range(len(names)))
    for a, b in MIRROR_PAIRS:
        if a in idx and b in idx:
            perm[idx[a]], perm[idx[b]] = idx[b], idx[a]
    lr = local_rot[:, perm, :].clone()
    # reflect each local rotation: negate x and y components (flip in-plane axes)
    lr[..., 1] *= -1.0
    lr[..., 2] *= -1.0
    # root translation: reflect the lateral (BVH world Z) component
    rt = root_trans.clone()
    rt[..., 2] *= -1.0
    return lr, rt


def retarget_clip(bvh_path, tgt_skel, mirror=False):
    """Returns (local_rot (N, J, 4) wxyz, root_trans (N, 3) m, fps)."""
    names, parents, offsets_m, local_rot, root_trans, fps = load_bvh_real_joints(
        bvh_path
    )

    if mirror:
        local_rot, root_trans = mirror_local(names, local_rot, root_trans)

    N = local_rot.shape[0]

    # Identity copy: non-root locals are frame-invariant; copy verbatim. Reorder
    # to the target skeleton body order (which is the BVH real-joint order with
    # Hips renamed trunk -- so the order already matches, but map by name to be
    # safe).
    name_to_src = {n: i for i, n in enumerate(names)}
    J = len(tgt_skel.node_names)
    out_local = torch.zeros(N, J, 4)
    out_local[..., 0] = 1.0
    for ti, tname in enumerate(tgt_skel.node_names):
        sname = ROOT_BVH_NAME if tname == ROOT_BODY_NAME else tname
        si = name_to_src[sname]
        out_local[:, ti, :] = local_rot[:, si, :]

    # Root (trunk) local rotation: world-frame, rotate Y-up -> Z-up.
    yup = YUP_TO_ZUP.expand(N, 4)
    out_local[:, 0, :] = quat_normalize(quat_mul(yup, out_local[:, 0, :]))

    # Root translation: Hips position cm, Y-up -> Z-up, then scale cm -> m.
    rt = quat_rotate(YUP_TO_ZUP.expand(N, 4), root_trans) * CM_TO_M
    return out_local, rt, fps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh-dir", default=DEFAULT_BVH_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--clips",
        default=None,
        help="Comma-separated clip stems (e.g. '0,37'). Default: all .bvh files.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Also emit left/right-mirrored copies (<clip>_mirror.npy)",
    )
    parser.add_argument("--mjcf-path", default=DEFAULT_MJCF)
    args = parser.parse_args()

    if args.clips:
        stems = [c.strip() for c in args.clips.split(",") if c.strip()]
    else:
        stems = sorted(
            (f[:-4] for f in os.listdir(args.bvh_dir) if f.endswith(".bvh")),
            key=lambda s: (len(s), s),
        )

    os.makedirs(args.output_dir, exist_ok=True)
    tgt_skel = target_skeleton(args.mjcf_path)
    print(f"Target skeleton ({len(tgt_skel.node_names)} bodies): {tgt_skel.node_names}")

    for stem in stems:
        bvh_path = os.path.join(args.bvh_dir, f"{stem}.bvh")
        variants = [(False, f"{stem}.npy")]
        if args.mirror:
            variants.append((True, f"{stem}_mirror.npy"))
        for mirror, out_name in variants:
            local_rot, root_trans, fps = retarget_clip(
                bvh_path, tgt_skel, mirror=mirror
            )
            out_path = os.path.join(args.output_dir, out_name)
            save_skeleton_motion_npy(
                out_path, tgt_skel, local_rot, root_trans, fps
            )
            print(
                f"{out_name}: {local_rot.shape[0]} frames @ {fps:.0f} fps, "
                f"root z [{root_trans[:, 2].min():.3f}, "
                f"{root_trans[:, 2].max():.3f}] m"
            )


if __name__ == "__main__":
    with torch.no_grad():
        main()
