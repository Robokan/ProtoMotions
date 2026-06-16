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
# Bone STL names are the LEFT-side dm_control mesh names (file = BONE<name>.stl);
# the right side substitutes _L -> _R. `child` is the dm child-joint offset (the
# bone's long axis); `our_len` our bone length; `our_axis` our bone direction
# (default +X; the hind foot points -Y). `scale` overrides the default
# our_len/|child| sizing for bones that aren't simple between-joints segments.
SEGMENTS = {
    "UpLeg": dict(
        stls=["Femoris_L", "Femoris_fabellae_L_1", "Femoris_fabellae_L_2",
              "Patella_L"],
        gpos=(0.32629, -0.047237, -0.46357),
        child=(-0.023037, 0.0311, -0.16702),
        our_len=0.16288,
    ),
    "Leg": dict(
        stls=["Tibia_L", "Fibula_L"],
        gpos=(0.34933, -0.078337, -0.29655),
        child=(-0.17914, -0.0056423, -0.13824),
        our_len=0.18324,
    ),
    "Shoulder": dict(
        stls=["Scapula_L"],
        gpos=(-0.1204, -0.02, -0.55522),
        child=(0.075, 0.033, -0.13),
        our_len=0.08144,
        # The scapula is a large flat blade, NOT a between-joints limb segment;
        # our short Shoulder->Arm bone (0.081) would shrink it. Keep it near the
        # overall dog scale (~0.95, like the matched limb bones) so it reads as a
        # real shoulder blade extending up toward the withers.
        scale=0.95,
    ),
    "Arm": dict(
        stls=["humerus_L"],
        gpos=(-0.1954, -0.053, -0.42522),
        child=(-0.05, 0.015, -0.145),
        our_len=0.154736,
    ),
    "ForeArm": dict(
        stls=["Radius_L", "Ulna_L", "Carpal_ulnar_L", "Carpal_accessory_L"],
        gpos=(-0.1454, -0.068, -0.28022),
        child=(0.003, -0.015, -0.19),
        our_len=0.181204,
    ),
    "Foot": dict(  # hind paw: metatarsals + tarsal bones; our bone points -Y
        stls=["Calcaneal_tuber_L", "Metatarsi_L_1", "Metatarsi_L_2",
              "Metatarsi_L_3", "Metatarsi_L_4", "Tarsus_L_I", "Tarsus_L_II",
              "Tarsus_L_III", "Tarsus_L_IV", "Tarsus_central_L",
              "Tibial_tarsal_L"],
        gpos=(0.52847, -0.072695, -0.15831),
        child=(-0.015043, 0.0081311, -0.11993),
        our_len=0.109944,
        our_axis=(0.0, -1.0, 0.0),
    ),
    "Hand": dict(  # front paw: metacarpals + carpal (wrist) bones
        stls=["Carpal_III_L", "Carpal_II_L", "Carpal_IV_L", "Carpal_I_L",
              "Carpal_L", "Carpal_Sesamoid_L", "Os_metacarpale_III_L",
              "Os_metacarpale_II_L", "Os_metacarpale_IV_L", "Os_metacarpale_I_L",
              "Os_metacarpale_V_L.001", "Os_metacarpale_V_L",
              "Phalanx_distalis_digiti_I_L"],
        gpos=(-0.1484, -0.053, -0.090221),
        child=(0.02, 0, -0.06),
        our_len=0.073296,
    ),
}
# our body name per (side, segment)
BODY = {seg: "{S}" + seg for seg in SEGMENTS}


def _resolve_stls(names, side):
    """Map dm left-side bone names to existing STL paths (R substitutes _L->_R)."""
    out = []
    for n in names:
        if side == "Right":
            n = n.replace("_L", "_R")
        p = DM_ASSETS / f"BONE{n}.stl"
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
    stls = _resolve_stls(spec["stls"], side)
    if not stls:
        raise FileNotFoundError(f"no STLs for {seg} {side}: {spec['stls']}")

    meshes = []
    for p in stls:
        m = trimesh.load(p, process=False)
        m.vertices = m.vertices + gpos  # into dm body frame (joint at origin)
        meshes.append(m)
    mesh = trimesh.util.concatenate(meshes)

    dm_axis = child / np.linalg.norm(child)
    our_axis = np.array(spec.get("our_axis", (1.0, 0.0, 0.0)))
    if side == "Right":  # our right foot uses the same -Y axis as the left
        pass
    s = spec.get("scale") or spec["our_len"] / np.linalg.norm(child)
    cross = np.cross(dm_axis, our_axis)
    R = tf.rotation_matrix(
        np.arccos(np.clip(dm_axis @ our_axis, -1, 1)),
        cross if np.linalg.norm(cross) > 1e-9 else [0, 0, 1],
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
        # find this body's collision capsule geom
        marker = f'<geom name="geom_{body}"'
        idx = xml.index(marker)
        line_end = xml.index("/>", idx) + 2
        capsule = xml[idx:line_end]
        # hide the capsule visually (keep collision) so the bone shows through,
        # then insert the cosmetic bone geom right after it
        hidden = capsule[:-2] + ' rgba="0.5 0.5 0.5 0"/>'
        bone = (f'\n          <geom name="{asset}" type="mesh" mesh="{asset}" '
                f'contype="0" conaffinity="0" density="0" group="2" '
                f'rgba="0.93 0.90 0.83 1"/>')
        xml = xml[:idx] + hidden + bone + xml[line_end:]
    OUT_MJCF.write_text(xml)
    print(f"\nwrote {OUT_MJCF}  ({len(geoms)} bone geoms added)")


if __name__ == "__main__":
    main()
