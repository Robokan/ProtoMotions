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
"""Limit-aware validation of a retargeted dog .motion against the ACTUAL
playable pose.

Loads a converted .motion's dof_pos + root pose, runs MuJoCo FK (the same FK
the simulator uses) with qpos CLAMPED to the MJCF joint limits, and reports:
  (a) number of hinge joints out of limit in the stored dof_pos,
  (b) per-paw contact-sphere-bottom world Z (min and stance-mean), ground z=0,
  (c) torso pitch (deg),
  (d) NaNs.

This measures what the sim will really play, not the limit-free retarget.

Usage:
    python data/scripts/verify_dog_limits.py --motion <clip>.motion \
        --mjcf protomotions/data/assets/mjcf/dog_v2_nomesh.xml
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import typer

sys.path.insert(0, str(Path(__file__).parent))
from protomotions.components.pose_lib import extract_kinematic_info  # noqa: E402

app = typer.Typer(pretty_exceptions_enable=False)

# Paw contact point (bottom of the lowest contact sphere) in the ee body's
# LOCAL frame, from dog_v2_nomesh.xml (sphere center + (0,0,-radius)).
CONTACT = {
    "front_L": ("finger_L", (0.038, 0.0, -0.033)),
    "front_R": ("finger_R", (0.038, 0.0, -0.033)),
    "hind_L": ("toe_L", (0.04, 0.0, -0.041)),
    "hind_R": ("toe_R", (0.04, 0.0, -0.041)),
}
STANCE_BAND = 0.06  # frames within this of the paw's clip-min are "stance"


def quat_rotate_wxyz(q, v):
    """Rotate vec v (3,) by wxyz quat q (4,). numpy."""
    w, x, y, z = q
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


@app.command()
def main(
    motion: Path = typer.Option(..., help="Retargeted .motion file"),
    mjcf: str = typer.Option(
        "protomotions/data/assets/mjcf/dog_v2_nomesh.xml", help="Dog MJCF"
    ),
    label: str = typer.Option("", help="Optional label for the report header"),
):
    ki = extract_kinematic_info(mjcf)
    lo = ki.dof_limits_lower.numpy()
    hi = ki.dof_limits_upper.numpy()
    dof_names = ki.dof_names

    m = torch.load(str(motion), weights_only=False, map_location="cpu")
    dof_pos = m["dof_pos"].numpy()  # (N, nd)
    body_pos = m["rigid_body_pos"].numpy()  # (N, B, 3)
    body_rot = m["rigid_body_rot"].numpy()  # (N, B, 4) xyzw (COMMON)
    N = dof_pos.shape[0]

    finite = (
        np.isfinite(dof_pos).all()
        and np.isfinite(body_pos).all()
        and np.isfinite(body_rot).all()
    )

    # (a) joints out of limit in the STORED dof_pos (the values the sim drives).
    eps = 1e-4
    below = dof_pos < (lo[None, :] - eps)
    above = dof_pos > (hi[None, :] + eps)
    oob = below | above  # (N, nd)
    oob_any_frame = oob.any(axis=0)  # (nd,)
    n_oob_joints = int(oob_any_frame.sum())
    # worst overshoot per offending joint
    overshoot = np.maximum(lo[None, :] - dof_pos, dof_pos - hi[None, :])
    worst = []
    for j in np.where(oob_any_frame)[0]:
        worst.append((dof_names[j], float(overshoot[:, j].max())))
    worst.sort(key=lambda t: -t[1])

    # MuJoCo ground-truth FK with qpos clamped to limits.
    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)
    # qpos layout: [root_pos(3), root_quat wxyz(4), hinge dofs...]; our dof order
    # (kinematic_info.dof_names) is MJCF document order, == MuJoCo qpos[7:] order
    # for an all-hinge tree. Verify counts match.
    assert model.nq - 7 == dof_pos.shape[1], (
        f"dof count mismatch model {model.nq - 7} vs motion {dof_pos.shape[1]}"
    )

    # root pose taken from the stored torso body (index 0). Convert xyzw->wxyz.
    root_pos = body_pos[:, 0, :]
    root_quat_xyzw = body_rot[:, 0, :]
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    dof_clamped = np.clip(dof_pos, lo[None, :], hi[None, :])

    # body indices in mujoco
    bid = {lbl: model.body(CONTACT[lbl][0]).id for lbl in CONTACT}
    torso_bid = model.body("torso").id

    contact_z = {lbl: np.zeros(N) for lbl in CONTACT}
    torso_pitch = np.zeros(N)
    mj_stored_match = []  # |mujoco xpos - stored rigid_body_pos| using CLAMPED dof

    for i in range(N):
        data.qpos[:3] = root_pos[i]
        data.qpos[3:7] = root_quat_wxyz[i]
        data.qpos[7:] = dof_clamped[i]
        mujoco.mj_forward(model, data)
        for lbl in CONTACT:
            off = np.array(CONTACT[lbl][1])
            bp = data.xpos[bid[lbl]]
            bq = data.xquat[bid[lbl]]  # wxyz
            ct = bp + quat_rotate_wxyz(bq, off)
            contact_z[lbl][i] = ct[2]
        # torso pitch: rotate body x-axis, measure elevation angle
        tq = data.xquat[torso_bid]
        xaxis = quat_rotate_wxyz(tq, np.array([1.0, 0.0, 0.0]))
        torso_pitch[i] = np.degrees(np.arcsin(np.clip(xaxis[2], -1, 1)))

    # ground reference: lowest contact point across all paws over the clip
    floor = min(contact_z[lbl].min() for lbl in CONTACT)

    print(f"\n=== limit-aware validation: {label or motion.name} ===")
    print(f"frames={N}  finite={finite}")
    print(f"(a) hinge joints out of limit (stored dof): {n_oob_joints} / {len(dof_names)}")
    for name, ov in worst[:12]:
        print(f"      {name:18s} worst overshoot {ov:+.3f} rad")
    print(f"(c) torso pitch: mean={torso_pitch.mean():+.1f} deg  "
          f"absmax={np.abs(torso_pitch).max():.1f} deg")
    print("(b) per-paw contact-sphere-bottom world Z (ground=0, floor-shifted):")
    print(f"      {'paw':8s} {'min':>8s} {'stance-mean':>12s} {'max':>8s}")
    paw_mins = {}
    paw_stance = {}
    for lbl in CONTACT:
        z = contact_z[lbl] - floor
        stance = z[z < (z.min() + STANCE_BAND)]
        sm = float(stance.mean()) if stance.size else float("nan")
        paw_mins[lbl] = float(z.min())
        paw_stance[lbl] = sm
        print(f"      {lbl:8s} {z.min():8.3f} {sm:12.3f} {z.max():8.3f}")
    fh = max(paw_mins["front_L"], paw_mins["front_R"])
    hh = max(paw_mins["hind_L"], paw_mins["hind_R"])
    print(f"      front/hind min asymmetry = {abs(fh - hh)*100:.1f} cm")

    return {
        "n_oob": n_oob_joints,
        "paw_mins": paw_mins,
        "paw_stance": paw_stance,
        "torso_pitch_absmax": float(np.abs(torso_pitch).max()),
        "finite": bool(finite),
    }


if __name__ == "__main__":
    app()
