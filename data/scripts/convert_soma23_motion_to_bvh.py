# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""SOMA23 motions -> SOMASkeleton77 BVH (inverse of convert_soma23_bvh_to_proto).

Purpose: any SOMA motion library (e.g. the Reallusion combat/drunken clips that
never existed as BVH) can be emitted as SEED-convention BVH and flow through
the PROVEN GMR soma_bvh -> robot retargeting path — no frame-convention
guessing (that approach mangled; see retarget_soma_motionlib.py post-mortem).

Inverse math (exact mirror of the forward converter):
  forward: L_bvh --FK--> G_bvh --(@O^T)--> G_std --to_local--> L_std[77]
           --subselect--> L_std[23] --create_motion(rot1 right, rot2 left)-->
           stored local_rigid_body_rot L_f, root G_f = R2 @ G_yup @ R1
  inverse: L_yup(j>=1) = R1 @ L_f @ R1^T          (conjugation)
           G_yup(root) = R2^T @ L_f(root) @ R1^T
           pos_yup     = R2^T @ pos_zup, m -> cm
           77 assembly: 23 mapped via SOMASKEL77_TO_MJCF_INDICES; the other
           joints take the template's frame-0 standard-T-pose locals.
           G_std --(@O)--> G_bvh --to_local--> L_bvh --as_euler(native)--> BVH

Validation: round-trip a SEED clip (BVH -> .motion -> BVH' -> .motion') and
diff body positions — should match to numerical precision.

Usage (battle container):
    python data/scripts/convert_soma23_motion_to_bvh.py \\
        --lib data/soma_drunken_combat.pt --output-dir /tmp/soma_bvh_out \\
        --template-bvh <any SEED .bvh>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bvh import (  # noqa: E402
    SkeletonBvh,
    load_bvh_animation,
    change_tpose,
    bvh_local_to_global_rotations,
    global_rots_to_local_rots,
)
from convert_soma23_to_proto import SOMASKEL77_TO_MJCF_INDICES  # noqa: E402

TPOSE_OFFSETS_PATH = "data/soma/standard_t_pose_global_offsets_rots.p"

# The exact frame rotations create_motion_from_soma23_global_rotations applies
R1 = R.from_euler("xyz", [-np.pi / 2, 0, 0]).as_matrix()
R2 = R.from_euler("xyz", [-np.pi / 2, np.pi, 0]).as_matrix()


def quat_xyzw_to_mat(q):
    return R.from_quat(np.asarray(q).reshape(-1, 4)).as_matrix().reshape(
        q.shape[:-1] + (3, 3)
    )


def load_template(template_bvh, offsets):
    """Template skeleton + its frame-0 standard-T-pose locals (77)."""
    skel = SkeletonBvh()
    skel.load_from_bvh(str(template_bvh), exclude_bones={"Root"})
    parents = skel.get_parent_indices()
    _, local0 = load_bvh_animation(str(template_bvh), skel)
    local0 = torch.tensor(local0[:1])  # [1, 77, 3, 3]
    l_std0, _ = change_tpose(local0, offsets, parents)
    # Header text: everything up to and including the MOTION line
    lines = Path(template_bvh).read_text().splitlines()
    header = []
    for ln in lines:
        header.append(ln)
        if ln.strip() == "MOTION":
            break
    # Native rotation channel order (assumed uniform, as the loader asserts)
    rot_order = ""
    for ln in lines:
        if "CHANNELS" in ln and "rotation" in ln:
            parts = ln.split()
            rot_order = "".join(
                c[0] for c in parts if c.endswith("rotation")
            )
            break
    return skel, parents, l_std0[0].numpy(), header, rot_order


def motion_to_bvh(l_f, root_pos_zup, template, fps, out_path):
    """l_f: [T,23,4] xyzw local quats; root_pos_zup: [T,3] meters."""
    skel, parents, l_std0, header, rot_order = template
    T = l_f.shape[0]
    Lf = quat_xyzw_to_mat(l_f)  # [T,23,3,3]

    # Undo create_motion's frame rotations
    L_yup = np.einsum("mn,tjno,po->tjmp", R1, Lf, R1)  # R1 @ L @ R1^T
    G_root_yup = np.einsum("nm,tno,po->tmp", R2, Lf[:, 0], R1)  # R2^T @ L0 @ R1^T
    L_yup[:, 0] = G_root_yup
    pos_yup = root_pos_zup @ R2 * 100.0  # (R2^T @ p) rows; m -> cm

    # Assemble 77-joint standard-T-pose locals
    L77 = np.repeat(l_std0[None], T, axis=0)
    for mjcf_j, sk77 in enumerate(SOMASKEL77_TO_MJCF_INDICES):
        L77[:, sk77] = L_yup[:, mjcf_j]

    # Invert change_tpose: G_std -> G_bvh = G_std @ O -> locals
    offsets = torch.load(TPOSE_OFFSETS_PATH, weights_only=False).numpy()
    G_std = bvh_local_to_global_rotations(torch.tensor(L77), parents).numpy()
    G_bvh = np.einsum("tjmn,jno->tjmo", G_std, offsets)
    L_bvh = global_rots_to_local_rots(torch.tensor(G_bvh), parents).numpy()

    eulers = np.rad2deg(
        R.from_matrix(L_bvh.reshape(-1, 3, 3)).as_euler(rot_order)
    ).reshape(T, -1, 3)

    with open(out_path, "w") as f:
        f.write("\n".join(header) + "\n")
        f.write(f"Frames: {T}\n")
        f.write(f"Frame Time: {1.0 / fps:.8f}\n")
        for t in range(T):
            # Layout: Root 6 channels (excluded bone — loader ignores; zeros),
            # then Hips 6 (position + rotation), then remaining joints x3.
            vals = [0.0] * 6
            vals.extend(pos_yup[t])
            vals.extend(eulers[t, 0])
            for j in range(1, eulers.shape[1]):
                vals.extend(eulers[t, j])
            f.write(" ".join(f"{v:.6f}" for v in vals) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--lib", required=True, help="packed SOMA23 MotionLib .pt")
    p.add_argument("--template-bvh", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--names", default=None, help="comma-separated clip stems")
    args = p.parse_args()

    offsets = torch.load(TPOSE_OFFSETS_PATH, weights_only=False)
    template = load_template(args.template_bvh, offsets)
    print(f"template: {len(template[1])} joints, rot order {template[4]}")

    d = torch.load(args.lib, map_location="cpu", weights_only=False)
    files = [str(f).split("/")[-1].replace(".motion", "") for f in d["motion_files"]]
    starts, nf = d["length_starts"], d["motion_num_frames"]
    want = set(args.names.split(",")) if args.names else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, name in enumerate(files):
        if want and name not in want:
            continue
        s, e = int(starts[i]), int(starts[i]) + int(nf[i])
        fps = round(1.0 / float(d["motion_dt"][i]))
        motion_to_bvh(
            d["lrs"][s:e].numpy(),
            d["gts"][s:e, 0].numpy(),
            template,
            fps,
            args.output_dir / f"{name}.bvh",
        )
        n += 1
        print(f"  wrote {name}.bvh ({e - s} frames @ {fps} fps)")
    print(f"done: {n} BVH files -> {args.output_dir}")


if __name__ == "__main__":
    main()
