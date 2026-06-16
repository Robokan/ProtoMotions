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
"""Compare a retargeted dog .motion against its source BVH.

Rotation-only retargeting preserves joint angles but not foot contact when
the target's limb proportions differ from the source. This tool quantifies
the mismatch: it reports, per limb (and the pelvis/root), the ground-relative
height profile in the SOURCE mocap vs the retargeted TARGET — so a floating
foot shows up as a number, not a guess.

Usage:
    python data/scripts/compare_dog_retarget.py \
        --bvh "/home/bizon/eric/Mode Adaptive/mocap/37.bvh" \
        --motion data/motions/dog_v2/clips/37.motion \
        --mjcf protomotions/data/assets/mjcf/dog_v2_nomesh.xml
"""

import sys
from pathlib import Path

import numpy as np
import torch
import typer

sys.path.insert(0, str(Path(__file__).parent))
from retarget_bvh_to_dog import load_bvh_zup, fk_global  # noqa: E402
from poselib_vendor import quat_rotate  # noqa: E402

app = typer.Typer(pretty_exceptions_enable=False)

# Source BVH end-effector / landmark joints -> limb label.
SRC_LANDMARKS = {
    "front_L": "LeftHand_end",
    "front_R": "RightHand_end",
    "hind_L": "LeftFoot_end",
    "hind_R": "RightFoot_end",
    "pelvis": "Hips",
}
# Target dog bodies for the same landmarks (paw tips).
TGT_LANDMARKS = {
    "front_L": "finger_L",
    "front_R": "finger_R",
    "hind_L": "toe_L",
    "hind_R": "toe_R",
    "pelvis": "pelvis",
}
# Paw contact point (bottom of the lowest contact sphere) in the target body's
# LOCAL frame, from dog_v2_nomesh.xml: sphere center pos + (0,0,-radius). The
# offset already nets the radius, so its world Z is the contact-sphere bottom.
# pelvis has no contact sphere (origin used as-is).
TGT_CONTACT_OFFSET = {
    "front_L": (0.038, 0.0, -0.033),
    "front_R": (0.038, 0.0, -0.033),
    "hind_L": (0.04, 0.0, -0.041),
    "hind_R": (0.04, 0.0, -0.041),
    "pelvis": (0.0, 0.0, 0.0),
}

CONTACT_BAND = 0.05  # [m] a foot within this of clip-min counts as "planted"


def height_stats(z: np.ndarray) -> dict:
    floor = float(z.min())
    rel = z - floor
    planted = (rel < CONTACT_BAND).mean()
    return {
        "min": floor,
        "max": float(z.max()),
        "range": float(z.max() - z.min()),
        "planted_frac": float(planted),
    }


@app.command()
def main(
    bvh: Path = typer.Option(..., help="Source BVH file"),
    motion: Path = typer.Option(..., help="Retargeted .motion file"),
    mjcf: Path = typer.Option(
        "protomotions/data/assets/mjcf/dog_v2_nomesh.xml", help="Dog MJCF"
    ),
):
    # --- source: FK the BVH to global landmark heights ---
    src_skel, local_rot, root_trans, fps = load_bvh_zup(str(bvh))
    g_rot, g_pos = fk_global(src_skel, local_rot, root_trans)
    # BVH is in cm; scale to meters using the source standing hip height so it
    # is commensurate with the target. (Hips ~0.47 m standing.)
    hips_z = float(g_pos[:, {n: i for i, n in enumerate(src_skel.node_names)}["Hips"], 2].mean())
    src_to_m = 0.47 / hips_z
    g_pos = g_pos * src_to_m
    src_idx = {n: i for i, n in enumerate(src_skel.node_names)}
    # normalize the whole source so its global min z is 0
    src_floor = float(g_pos[..., 2].min())

    # --- target: paw contact-point heights from the converted motion ---
    # TARGET = the paw CONTACT POINT (bottom of the lowest contact sphere):
    # body_pos + R_body * contact_offset, using rigid_body_rot. This is what
    # determines visible ground contact, not the toe/finger body origin.
    m = torch.load(motion, weights_only=False, map_location="cpu")
    body_pos = m["rigid_body_pos"]  # (N, B, 3)
    body_rot = m["rigid_body_rot"]  # (N, B, 4) xyzw (RobotState COMMON)
    # vendored quat_rotate is wxyz -> reorder xyzw -> wxyz
    body_rot_wxyz = body_rot[..., [3, 0, 1, 2]]
    # body order matches the MJCF; read names from kinematic info
    from protomotions.components.pose_lib import extract_kinematic_info

    ki = extract_kinematic_info(str(mjcf))
    tgt_idx = {n: i for i, n in enumerate(ki.body_names)}

    def tgt_contact_z(label):
        bi = tgt_idx[TGT_LANDMARKS[label]]
        off = torch.tensor(TGT_CONTACT_OFFSET[label], dtype=body_pos.dtype)
        pos = body_pos[:, bi, :]  # (N,3)
        rot = body_rot_wxyz[:, bi, :]  # (N,4)
        ct = pos + quat_rotate(rot, off.expand(pos.shape[0], 3))
        return ct[:, 2].numpy()  # world Z = contact-sphere bottom

    # floor = min over all paw contact points (true ground reference)
    tgt_floor = float(
        min(tgt_contact_z(lbl).min() for lbl in ["front_L", "front_R", "hind_L", "hind_R"])
    )

    print(f"clip: {bvh.name}  src_frames={g_pos.shape[0]}  tgt_frames={body_pos.shape[0]}")
    print("TARGET = paw contact point (contact-sphere bottom: body pos + "
          "R_body * contact_offset)")
    print(f"{'landmark':10s} | {'SOURCE (mocap)':^34s} | {'TARGET (dog)':^34s}")
    print(f"{'':10s} | {'minZ':>7s} {'rangeZ':>7s} {'plant%':>7s} {'':4s} | "
          f"{'minZ':>7s} {'rangeZ':>7s} {'plant%':>7s}")
    print("-" * 84)
    n = min(g_pos.shape[0], body_pos.shape[0])
    for label in ["front_L", "front_R", "hind_L", "hind_R", "pelvis"]:
        sz = g_pos[:n, src_idx[SRC_LANDMARKS[label]], 2].numpy() - src_floor
        tz = tgt_contact_z(label)[:n] - tgt_floor
        s, t = height_stats(sz), height_stats(tz)
        # temporal correlation of the height profile = did the gait phase map?
        corr = float(np.corrcoef(sz, tz)[0, 1]) if sz.std() > 1e-6 and tz.std() > 1e-6 else float("nan")
        print(f"{label:10s} | {s['min']:7.3f} {s['range']:7.3f} {s['planted_frac']*100:6.1f}% "
              f"    | {t['min']:7.3f} {t['range']:7.3f} {t['planted_frac']*100:6.1f}%  | corr={corr:+.2f}")

    print("\nGait-phase correlation (last column): +1 = foot rises/falls in sync "
          "with the mocap. Low/negative = wrong phase or limb mis-mapped.\n"
          "Range mismatch = foot lifts too little/much (proportions / no foot IK).")


if __name__ == "__main__":
    app()
