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
"""Bake cosmetic bone meshes onto the BVH-matched dog skeleton (legs first).

The dm_control dog ships detailed per-bone STLs, but they live in one global
rest-pose frame (each placed by a large geom `pos`). Our dog (dog_v2_nomesh.xml)
is a different, BVH-matched skeleton whose bones are capsules along local +X.

For each leg segment we take the matching dm_control bone(s), place them in the
dm body frame (vertices + dm geom pos), then fit them to OUR bone by:
  1. rotating the dm bone's long axis (its child-joint direction) onto our bone
     axis (+X),
  2. uniformly scaling by our_bone_length / dm_bone_length, and
  3. (the dm proximal end already sits at the joint = body origin).
The transformed mesh is baked in our body's local frame, so it attaches with an
identity geom. Bones are COSMETIC ONLY: contype/conaffinity=0 (the capsules keep
doing collision) and density=0 (no mass/inertia change -- the model is unchanged).

Outputs baked OBJs to data/assets/mesh/dog_v2_bones/ and writes
data/assets/mjcf/dog_v2_bones.xml (= dog_v2_nomesh.xml + the bone assets/geoms).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import trimesh
from trimesh import transformations as tf

REPO = Path(__file__).resolve().parents[2]
DM_ASSETS = Path("/home/bizon/eric/projects/deepmind dog/dog_assets")
SRC_MJCF = REPO / "protomotions/data/assets/mjcf/dog_v2_nomesh.xml"
OUT_MJCF = REPO / "protomotions/data/assets/mjcf/dog_v2_bones.xml"
OUT_MESH = REPO / "protomotions/data/assets/mesh/dog_v2_bones"

# Per hind/front segment (LEFT side): dm bone STLs, the dm geom `pos` that places
# them in the dm body frame, the dm child-joint offset (= bone long axis), and
# our bone length along +X. Right side is the mirror (y negated, _R meshes).
SEGMENTS = {
    "UpLeg": dict(
        stls=["Femoris", "Femoris_fabellae", "Patella"],  # _L_1/_2 expanded below
        gpos=(0.32629, -0.047237, -0.46357),
        child=(-0.023037, 0.0311, -0.16702),
        our_len=0.16288,
    ),
    "Leg": dict(
        stls=["Tibia", "Fibula"],
        gpos=(0.34933, -0.078337, -0.29655),
        child=(-0.17914, -0.0056423, -0.13824),
        our_len=0.18324,
    ),
    "Shoulder": dict(
        stls=["Scapula"],
        gpos=(-0.1204, -0.02, -0.55522),
        child=(0.075, 0.033, -0.13),
        our_len=0.08144,
    ),
    "Arm": dict(
        stls=["humerus"],
        gpos=(-0.1954, -0.053, -0.42522),
        child=(-0.05, 0.015, -0.145),
        our_len=0.154736,
    ),
    "ForeArm": dict(
        stls=["Radius", "Ulna", "Carpal_ulnar", "Carpal_accessory"],
        gpos=(-0.1454, -0.068, -0.28022),
        child=(0.003, -0.015, -0.19),
        our_len=0.181204,
    ),
}
# our body name per (side, segment)
BODY = {"UpLeg": "{S}UpLeg", "Leg": "{S}Leg", "Shoulder": "{S}Shoulder",
        "Arm": "{S}Arm", "ForeArm": "{S}ForeArm"}


def _resolve_stls(base_names, side):
    """Expand base bone names to actual _L/_R STL filenames present on disk."""
    out = []
    for b in base_names:
        # try a few suffix patterns dm_control uses
        for cand in (f"BONE{b}_{side}.stl", f"BONE{b}_{side}_1.stl",
                     f"BONE{b}_{side}_2.stl"):
            p = DM_ASSETS / cand
            if p.exists():
                out.append(p)
    return out


def bake_segment(seg, side):
    spec = SEGMENTS[seg]
    gpos = np.array(spec["gpos"], float)
    child = np.array(spec["child"], float)
    if side == "Right":  # mirror the dm left bone across the sagittal (y) plane
        gpos = gpos * [1, -1, 1]
        child = child * [1, -1, 1]
    stls = _resolve_stls(spec["stls"], "L" if side == "Left" else "R")
    if not stls:
        raise FileNotFoundError(f"no STLs for {seg} {side}: {spec['stls']}")

    meshes = []
    for p in stls:
        m = trimesh.load(p, process=False)
        m.vertices = m.vertices + gpos  # into dm body frame (joint at origin)
        meshes.append(m)
    mesh = trimesh.util.concatenate(meshes)

    dm_axis = child / np.linalg.norm(child)
    our_axis = np.array([1.0, 0.0, 0.0])
    s = spec["our_len"] / np.linalg.norm(child)  # uniform length-ratio scale
    R = tf.rotation_matrix(
        np.arccos(np.clip(dm_axis @ our_axis, -1, 1)),
        np.cross(dm_axis, our_axis)
        if np.linalg.norm(np.cross(dm_axis, our_axis)) > 1e-9 else [0, 0, 1],
    )[:3, :3]
    mesh.vertices = (s * (R @ mesh.vertices.T)).T  # fit into OUR body frame
    return mesh


def main():
    OUT_MESH.mkdir(parents=True, exist_ok=True)
    geoms = {}  # body_name -> mesh asset name
    for side in ("Left", "Right"):
        for seg in SEGMENTS:
            mesh = bake_segment(seg, side)
            body = BODY[seg].format(S=side)
            asset = f"bone_{body}"
            mesh.export(OUT_MESH / f"{asset}.obj")
            geoms[body] = asset
            print(f"baked {asset}: {len(mesh.vertices)} verts  "
                  f"bbox={np.round(mesh.bounds[1] - mesh.bounds[0], 3)}")

    # patch the MJCF: add <asset> meshes and a cosmetic <geom> in each leg body
    xml = SRC_MJCF.read_text()
    assets = "\n".join(
        f'    <mesh name="{a}" file="../mesh/dog_v2_bones/{a}.obj"/>'
        for a in geoms.values()
    )
    xml = xml.replace(
        "<worldbody>", f"  <asset>\n{assets}\n  </asset>\n  <worldbody>", 1
    )
    for body, asset in geoms.items():
        # insert the bone geom right after this body's collision capsule geom
        marker = f'<geom name="geom_{body}"'
        idx = xml.index(marker)
        line_end = xml.index("/>", idx) + 2
        bone = (f'\n          <geom name="{asset}" type="mesh" mesh="{asset}" '
                f'contype="0" conaffinity="0" density="0" group="2" '
                f'rgba="0.93 0.90 0.83 1"/>')
        xml = xml[:line_end] + bone + xml[line_end:]
    OUT_MJCF.write_text(xml)
    print(f"\nwrote {OUT_MJCF}  ({len(geoms)} bone geoms added)")


if __name__ == "__main__":
    main()
