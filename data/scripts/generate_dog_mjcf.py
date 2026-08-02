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
"""Generate a NEW dog_v2 MJCF whose skeleton matches the BVH mocap EXACTLY.

See data/scripts/DOG_REBUILD_PLAN.md (owner's original/ "Alternative" decision).

WHY this replaces the dm_control-framed dog: the dm_control dog's joint DOF
structure does not match the BVH (non-orthogonal scapula axes, 1-DOF
elbow/wrist, 7-segment lumbar over-distributing the BVH's 2 spine joints), so
the retarget + sequential decomposition produce shaking front legs and an
exaggerated back arch. Building a skeleton that is the BVH tree 1:1 (same body
hierarchy, same offsets, 3 ORTHOGONAL hinges per joint) makes the retarget a
near-identity rotation copy and the decomposition exact, eliminating both
problems. Bone meshes are DEFERRED -- each bone is drawn as a CAPSULE (cylinder)
for now.

Construction:
  * Read the BVH HIERARCHY of a reference clip (default 0.bvh) with the vendored
    parser (data/scripts/poselib_vendor parse_bvh_file).
  * Body tree = the 21 real BVH joints, 1:1. Root body 'trunk' (= Hips) has a
    free joint. Every other body keeps its BVH joint name verbatim (Spine,
    Spine1, Neck, Head, LeftShoulder, ... Tail1) so the retarget maps by
    identity. BVH End Sites are NOT bodies (they carry no DOF); their offsets
    are used only to size the leaf-body capsules.
  * Body offset (pos) = BVH OFFSET * CM_TO_M. No body quat (identity reference
    frame), so the 3 hinge axes are the body-frame world axes.
  * Each non-root body has 3 ORTHOGONAL hinge joints with axes X(1,0,0),
    Y(0,1,0), Z(0,0,1) in declaration order X,Y,Z, range +-180 deg, plus one PD
    position actuator per hinge. Orthogonal XYZ-in-order means the converter's
    'sequential' decomposition is exactly intrinsic-Euler-XYZ -> exact + stable.
  * Geom = one CAPSULE per body spanning the body origin to its child offset
    (the bone). Leaf bodies (Head, LeftHand, RightHand, LeftFoot, RightFoot,
    Tail1) use a short capsule toward their BVH End Site. The trunk gets a
    capsule spanning Hips -> Spine1 (the body axis).

Usage:
    python data/scripts/generate_dog_mjcf.py \
        [--bvh /home/bizon/eric/Mode\\ Adaptive/mocap/0.bvh] \
        [--output protomotions/data/assets/mjcf/dog_v2_nomesh.xml]
"""

import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poselib_vendor import parse_bvh_file  # noqa: E402

DEFAULT_BVH = "/home/bizon/eric/Mode Adaptive/mocap/0.bvh"
DEFAULT_OUTPUT = "protomotions/data/assets/mjcf/dog_v2_nomesh.xml"

# Standing-height scale cm -> m: BVH standing Hips height ~46.77 cm -> ~0.476 m.
CM_TO_M = 0.01018

# BVH root joint -> sim root body name. Every other body keeps its BVH name.
ROOT_BVH_NAME = "Hips"
ROOT_BODY_NAME = "trunk"

# Capsule radii (m) per anatomical region.
RADIUS_TRUNK = 0.045
RADIUS_SPINE = 0.040
RADIUS_NECK = 0.035
RADIUS_HEAD = 0.045
RADIUS_LEG_UPPER = 0.028
RADIUS_LEG_LOWER = 0.022
RADIUS_PAW = 0.020
RADIUS_TAIL = 0.018
RADIUS_DEFAULT = 0.025

# Per-region PD actuator peak torque (Nm). kp = 2*effort, kv = kp/10.
EFFORT_SPINE = 60.0
EFFORT_NECK = 20.0
EFFORT_HEAD = 10.0
EFFORT_LEG_UPPER = 40.0  # hip / shoulder
EFFORT_LEG_MID = 30.0    # knee / elbow
EFFORT_PAW = 15.0        # foot / hand (ankle / wrist)
EFFORT_TAIL = 1.0
EFFORT_DEFAULT = 10.0

JOINT_RANGE_DEG = 180.0
CTRL_RANGE_DEG = 180.0

AXES = [("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")]


def region_radius(name: str) -> float:
    if name in ("Spine", "Spine1"):
        return RADIUS_SPINE
    if name == "Neck":
        return RADIUS_NECK
    if name == "Head":
        return RADIUS_HEAD
    if re.fullmatch(r"(Left|Right)(UpLeg|Shoulder|Arm)", name):
        return RADIUS_LEG_UPPER
    if re.fullmatch(r"(Left|Right)(Leg|ForeArm)", name):
        return RADIUS_LEG_LOWER
    if re.fullmatch(r"(Left|Right)(Foot|Hand)", name):
        return RADIUS_PAW
    if name in ("Tail", "Tail1"):
        return RADIUS_TAIL
    return RADIUS_DEFAULT


def region_effort(name: str) -> float:
    if name in ("Spine", "Spine1"):
        return EFFORT_SPINE
    if name == "Neck":
        return EFFORT_NECK
    if name == "Head":
        return EFFORT_HEAD
    if re.fullmatch(r"(Left|Right)(UpLeg|Shoulder|Arm)", name):
        return EFFORT_LEG_UPPER
    if re.fullmatch(r"(Left|Right)(Leg|ForeArm)", name):
        return EFFORT_LEG_MID
    if re.fullmatch(r"(Left|Right)(Foot|Hand)", name):
        return EFFORT_PAW
    if name in ("Tail", "Tail1"):
        return EFFORT_TAIL
    return EFFORT_DEFAULT


def _fmt_vec(v):
    return " ".join(f"{x:.6g}" for x in v)


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return n


def build_bvh_tree(bvh_path):
    """Return (names, parents, offsets_m, children, end_offsets_m).

    Only the 21 REAL joints are kept as bodies; End Sites are dropped (their
    offsets are returned per-parent in end_offsets_m to size leaf capsules)."""
    names, parents, _root_trans, offsets, _local_rot, _fps = parse_bvh_file(bvh_path)
    parents = parents.tolist()
    offsets_cm = offsets.tolist()

    real_idx = [i for i, n in enumerate(names) if not n.endswith("_end")]
    end_idx = [i for i, n in enumerate(names) if n.endswith("_end")]

    # remap to compact indices over real joints only
    old_to_new = {old: new for new, old in enumerate(real_idx)}
    r_names = [names[i] for i in real_idx]
    r_parents = [old_to_new[parents[i]] if parents[i] >= 0 else -1 for i in real_idx]
    r_offsets = [[c * CM_TO_M for c in offsets_cm[i]] for i in real_idx]

    # children adjacency (real-joint indices)
    children = {i: [] for i in range(len(r_names))}
    for i, p in enumerate(r_parents):
        if p >= 0:
            children[p].append(i)

    # end-site offset per parent real-joint index (for leaf-capsule sizing)
    end_offsets_m = {}
    for ei in end_idx:
        pe = parents[ei]  # parent is a real joint
        end_offsets_m[old_to_new[pe]] = [c * CM_TO_M for c in offsets_cm[ei]]

    return r_names, r_parents, r_offsets, children, end_offsets_m


def add_capsule(body_el, name, vec, radius, group="0"):
    """Add a capsule geom from the body origin to point `vec` (body frame).

    For a near-zero-length bone, emit a small sphere instead."""
    L = _norm(vec)
    if L < 1e-4:
        ET.SubElement(
            body_el,
            "geom",
            {
                "name": f"geom_{name}",
                "type": "sphere",
                "size": f"{radius:.6g}",
                "pos": "0 0 0",
                "group": group,
            },
        )
        return
    ET.SubElement(
        body_el,
        "geom",
        {
            "name": f"geom_{name}",
            "type": "capsule",
            "fromto": f"0 0 0 {_fmt_vec(vec)}",
            "size": f"{radius:.6g}",
            "group": group,
        },
    )


def add_capsule_fromto(body_el, name, p0, p1, radius, group="0"):
    """Add a capsule geom spanning p0 -> p1 (both in the body frame)."""
    if _norm([p1[a] - p0[a] for a in range(3)]) < 1e-4:
        add_capsule(body_el, name, p1, radius, group)
        return
    ET.SubElement(
        body_el,
        "geom",
        {
            "name": f"geom_{name}",
            "type": "capsule",
            "fromto": f"{_fmt_vec(p0)} {_fmt_vec(p1)}",
            "size": f"{radius:.6g}",
            "group": group,
        },
    )


def build_mjcf(bvh_path, output_path):
    names, parents, offsets_m, children, end_offsets_m = build_bvh_tree(bvh_path)

    def body_name(i):
        return ROOT_BODY_NAME if names[i] == ROOT_BVH_NAME else names[i]

    out = ET.Element("mujoco", {"model": "dog_v2"})
    ET.SubElement(out, "compiler", {"angle": "degree", "autolimits": "true"})
    ET.SubElement(out, "option", {"timestep": "0.005"})

    default = ET.SubElement(out, "default")
    ET.SubElement(
        default,
        "geom",
        {"contype": "1", "conaffinity": "1", "condim": "3", "density": "1000"},
    )
    ET.SubElement(default, "joint", {"limited": "true", "armature": "0.01", "damping": "0.1"})

    worldbody = ET.SubElement(out, "worldbody")

    joint_order = []  # actuator declaration order

    def emit_body(i, parent_el):
        bname = body_name(i)
        pos = offsets_m[i] if parents[i] >= 0 else [0.0, 0.0, 0.0]
        body_el = ET.SubElement(
            parent_el, "body", {"name": bname, "pos": _fmt_vec(pos)}
        )

        if parents[i] < 0:
            # root: free joint + a small inertial-anchoring trunk capsule.
            ET.SubElement(body_el, "freejoint", {"name": "root"})
        else:
            for ax_letter, ax_vec in AXES:
                jn = f"{bname}_{ax_letter}"
                ET.SubElement(
                    body_el,
                    "joint",
                    {
                        "name": jn,
                        "type": "hinge",
                        "axis": ax_vec,
                        "pos": "0 0 0",
                        # Free-spinning (unlimited): the quaternion->3-hinge
                        # decomposition can produce angles that wrap past +-pi
                        # (an identical rotation). A hinge LIMIT would clamp the
                        # wrapped value and make the leg pop for a frame. Sim-only
                        # skeleton, so leave the hinges unlimited.
                        "limited": "false",
                    },
                )
                joint_order.append((jn, names[i]))

        # geom: capsule spanning to the (primary) child offset, else to end site.
        radius = region_radius(names[i])
        kids = children[i]
        if parents[i] < 0:
            # trunk = PELVIS: a short lateral capsule between the two hind-hip
            # sockets (LeftUpLeg <-> RightUpLeg). Do NOT span up the spine — the
            # Spine/Spine1 bodies already capsule the back. A long Hips->Spine1
            # rod overlapped the tail (Tail starts at +0.07m on the same axis
            # the spine runs) and penetrated the ground when the pelvis pitched
            # down while sitting.
            lu = ru = None
            for k in kids:
                if names[k] == "LeftUpLeg":
                    lu = offsets_m[k]
                elif names[k] == "RightUpLeg":
                    ru = offsets_m[k]
            if lu is not None and ru is not None:
                add_capsule_fromto(body_el, bname, lu, ru, RADIUS_TRUNK)
            else:
                # fallback: a short stub toward the first child
                tip = offsets_m[kids[0]] if kids else [0.06, 0.0, 0.0]
                add_capsule(body_el, bname, tip, RADIUS_TRUNK)
        elif kids:
            # span to the first child's offset (the bone direction)
            tip = offsets_m[kids[0]]
            if _norm(tip) < 1e-4 and len(kids) > 1:
                tip = offsets_m[kids[1]]
            add_capsule(body_el, bname, tip, radius)
        else:
            # leaf: span toward its BVH end site
            tip = end_offsets_m.get(i, [0.06, 0.0, 0.0])
            add_capsule(body_el, bname, tip, radius)

        for k in kids:
            emit_body(k, body_el)

    emit_body(0, worldbody)

    # actuators: one PD position actuator per hinge, in declaration order.
    act = ET.SubElement(out, "actuator")
    ctrl = CTRL_RANGE_DEG
    for jn, owner in joint_order:
        eff = region_effort(owner)
        kp = 2.0 * eff
        ET.SubElement(
            act,
            "position",
            {
                "name": jn,
                "joint": jn,
                "kp": f"{kp:.6g}",
                "kv": f"{kp / 10.0:.6g}",
                "ctrlrange": f"{-ctrl:.6g} {ctrl:.6g}",
                "forcerange": f"{-eff:.6g} {eff:.6g}",
            },
        )

    xml_str = minidom.parseString(ET.tostring(out)).toprettyxml(indent="  ")
    xml_str = "\n".join(line for line in xml_str.split("\n") if line.strip())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(xml_str + "\n")
    print(f"Wrote {output_path}  ({len(joint_order)} actuated hinges)")
    return [body_name(i) for i in range(len(names))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", default=DEFAULT_BVH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    body_names = build_mjcf(args.bvh, args.output)

    import mujoco as mj

    model = mj.MjModel.from_xml_path(args.output)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    mj_body_names = [model.body(i).name for i in range(model.nbody)]
    print(
        f"Compiled OK: nbody={model.nbody} (incl world), nq={model.nq}, "
        f"nv={model.nv}, nu={model.nu}, mass={sum(model.body_mass):.3f} kg"
    )
    print(f"Bodies ({model.nbody - 1}): {mj_body_names[1:]}")
    ti = mj_body_names.index(ROOT_BODY_NAME)
    print(f"Rest-pose {ROOT_BODY_NAME} z: {data.xpos[ti][2]:.4f} m")
    # report a few paw heights in rest pose
    for paw in ("LeftFoot", "RightFoot", "LeftHand", "RightHand"):
        if paw in mj_body_names:
            pi = mj_body_names.index(paw)
            print(f"  {paw} origin z: {data.xpos[pi][2]:.4f} m")


if __name__ == "__main__":
    main()
