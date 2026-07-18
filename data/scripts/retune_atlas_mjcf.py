# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Retune the GMR Atlas MJCF from an IK rig into a physics-ready robot.

The GMR export (GMR/assets/atlas_mujoco/atlas.xml) was built for kinematic
retargeting: every geom defaults to water density (total 114.8 kg / 253 lb),
every actuator is a unit-gear motor with ctrlrange +-1 (max 1 Nm), and the
ankle joints are locked (range [0, 0]).

This script produces atlas_physics.xml alongside it with:
- total mass scaled to 150 lb (68.04 kg) via collision-geom density,
- per-joint effort limits (actuatorfrcrange) + motor ctrlranges at EngineAI
  strength-to-weight (SA01: 140 Nm hips/knees @ 33 kg; PM01: 164 Nm @ 41 kg
  => ~4.1 Nm/kg primaries => 280 Nm at 68 kg), so it can drive the mocap the
  way a T800-class robot could,
- armature/frictionloss per the repo's g1 conventions,
- ankles unlocked (+-0.79 rad) so balance control is physically possible.

Usage:
    python data/scripts/retune_atlas_mjcf.py \\
        --input  /workspace/sparkpack/GMR/assets/atlas_mujoco/atlas.xml \\
        --output /workspace/sparkpack/GMR/assets/atlas_mujoco/atlas_physics.xml
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

TARGET_MASS_KG = 68.04  # 150 lb

# Joint-name pattern -> (effort Nm, velocity rad/s, armature, frictionloss)
# EngineAI-derived strength-to-weight at 68 kg; distal groups follow the
# PM01 52 Nm class scaled by mass (~86) or wrist/neck-small values.
JOINT_SPEC = [
    (r"^Leg_[135]_[LR]_Joint$", (280.0, 26.0, 0.04, 0.1)),   # hip cluster
    (r"^Leg_8_[LR]_Joint$",     (280.0, 26.0, 0.04, 0.1)),   # knee
    (r"^Foot_[LR]_Pitch$", (90.0, 35.0, 0.015, 0.1)),   # ankle pitch
    (r"^Foot_[LR]_(Roll|Yaw)$", (60.0, 35.0, 0.01, 0.1)),  # ankle roll/swivel
    (r"^(Twist|Backbone)_Joint$", (140.0, 26.0, 0.03, 0.1)),  # waist
    (r"^Arm_[13]_[LR]_Joint$",  (90.0, 35.0, 0.015, 0.1)),   # shoulder
    (r"^Arm_4_[LR]_Joint$",     (60.0, 35.0, 0.01, 0.1)),    # upper-arm yaw
    (r"^Arm_6_[LR]_Joint$",     (90.0, 35.0, 0.015, 0.1)),   # elbow
    (r"^Arm_[789]_[LR]_Joint$", (30.0, 35.0, 0.005, 0.1)),   # wrist triplet
    (r"^(Neck_2|Head)_Joint$",  (25.0, 35.0, 0.005, 0.1)),   # neck/head
]

ANKLE_RANGE = (-0.7854, 0.7854)


def spec_for(name: str):
    for pat, spec in JOINT_SPEC:
        if re.match(pat, name):
            return spec
    raise ValueError(f"no torque spec for joint {name!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import mujoco

    base = mujoco.MjModel.from_xml_path(str(args.input))
    base_mass = float(base.body_mass.sum())
    density_scale = TARGET_MASS_KG / base_mass
    print(f"base mass {base_mass:.1f} kg -> target {TARGET_MASS_KG} kg "
          f"(density x{density_scale:.4f})")

    tree = ET.parse(args.input)
    root = tree.getroot()

    # 1. Mass: scale EXPLICIT nonzero density attributes only (the rig
    # declares exactly two: visual class density=0, collision default
    # density=1000 — all mass flows from the latter). Do NOT stamp densities
    # onto attribute-less geoms: visual meshes inherit density=0 from their
    # class and must stay massless.
    for el in root.iter():
        if el.tag == "geom":
            d = el.get("density")
            if d is not None and float(d) > 0:
                el.set("density", f"{float(d) * density_scale:.2f}")

    # 2a. Ankles: the rig has PASSIVE BALL joints at the feet (Foot_L/R_Joint,
    # unactuated). Real Atlas ankles are 2-DOF (pitch + roll). We build them as
    # CHAINED SINGLE-JOINT BODIES — Leg_4 -> Ankle_<side> (pitch) -> Foot
    # (roll) — because (a) the framework requires 1-or-3 joints per body and
    # (b) the IsaacLab MJCF importer merges multi-joint bodies into one D6
    # joint whose axis names never receive PD gains (the robot-falls-over
    # root cause). Knee flexion axis is x, so ankle pitch = x, roll = y.
    new_ankle_joints = []
    parent_of = {c: par for par in root.iter() for c in par}
    for body in list(root.iter("body")):
        for joint in list(body.findall("joint")):
            if joint.get("type") == "ball" and (joint.get("name") or "").startswith("Foot_"):
                side = "L" if "_L_" in joint.get("name") else "R"
                parent = parent_of[body]
                # Intermediate pitch body takes the Foot body's placement
                ankle = ET.Element("body", dict(
                    name=f"Ankle_{side}",
                    pos=body.get("pos") or "0 0 0",
                ))
                if body.get("quat"):
                    ankle.set("quat", body.get("quat"))
                ET.SubElement(ankle, "inertial", dict(
                    pos="0 0 0", mass="0.02",
                    diaginertia="1e-05 1e-05 1e-05",
                ))
                pj = ET.SubElement(ankle, "joint", dict(
                    name=f"Foot_{side}_Pitch", type="hinge", pos="0 0 0",
                    axis="1 0 0",
                    range=f"{ANKLE_RANGE[0]:g} {ANKLE_RANGE[1]:g}",
                    limited="true",
                ))
                # Foot becomes the roll body, at the ankle origin
                idx = list(parent).index(body)
                parent.remove(body)
                body.set("pos", "0 0 0")
                if body.get("quat"):
                    del body.attrib["quat"]
                body.remove(joint)
                rj = ET.Element("joint", dict(
                    name=f"Foot_{side}_Roll", type="hinge", pos="0 0 0",
                    axis="0 1 0", range="-0.5236 0.5236", limited="true",
                ))
                body.insert(0, rj)
                ankle.append(body)
                parent.insert(idx, ankle)
                new_ankle_joints += [f"Foot_{side}_Pitch", f"Foot_{side}_Roll"]
    print("ankle balls replaced with hinges:", new_ankle_joints)

    # 2b. Joints: effort/armature/friction
    for joint in root.iter("joint"):
        name = joint.get("name")
        if not name or joint.get("type") == "free":
            continue
        effort, vel, arm, fric = spec_for(name)
        joint.set("actuatorfrcrange", f"-{effort:g} {effort:g}")
        joint.set("armature", f"{arm:g}")
        joint.set("frictionloss", f"{fric:g}")

    # 2c. Motors for the new ankle hinges
    actuator_el = root.find("actuator")
    for jname in new_ankle_joints:
        effort, *_ = spec_for(jname)
        ET.SubElement(actuator_el, "motor", dict(
            name=f"motor_{jname}", joint=jname,
            ctrlrange=f"-{effort:g} {effort:g}", ctrllimited="true",
        ))

    # 2d. Bake the root rotation: the rig's Hip body frame is rotated ~96deg
    # about x, but the framework requires an identity root quat (world-aligned
    # root frame). Rotate everything inside Hip by the old quat so world-space
    # geometry is unchanged. NOTE: motion converters must compose retargeted
    # root orientations with this same constant (see ATLAS_ROOT_BAKE_QUAT).
    import numpy as np

    def _qmul(a, b):
        w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def _qrot(q, v):
        qv = np.array([0.0, *v])
        qc = q * np.array([1, -1, -1, -1])
        return _qmul(_qmul(q, qv), qc)[1:]

    root_body = next(b for b in root.iter("body") if b.get("name") == "Hip")
    q_bake = np.array([float(x) for x in (root_body.get("quat") or "1 0 0 0").split()])
    q_bake /= np.linalg.norm(q_bake)
    if abs(q_bake[0]) < 0.99999:
        root_body.set("quat", "1 0 0 0")
        for el in list(root_body):
            if el.tag in ("geom", "body", "site", "inertial", "camera"):
                for bad in ("euler", "axisangle", "xyaxes", "zaxis", "fromto"):
                    if el.get(bad):
                        raise ValueError(f"root child {el.tag} uses {bad}; bake unsupported")
                p = np.array([float(x) for x in (el.get("pos") or "0 0 0").split()])
                el.set("pos", " ".join(f"{v:.8g}" for v in _qrot(q_bake, p)))
                qc = np.array([float(x) for x in (el.get("quat") or "1 0 0 0").split()])
                el.set("quat", " ".join(f"{v:.8g}" for v in _qmul(q_bake, qc)))
        print(f"root quat baked: {np.round(q_bake, 5).tolist()} -> identity "
              "(ATLAS_ROOT_BAKE_QUAT for motion converters)")

    # 3. Actuators: real torque motors (ctrlrange = effort in Nm, gear 1)
    for motor in root.iter("motor"):
        jname = motor.get("joint")
        if not jname:
            continue
        effort, *_ = spec_for(jname)
        motor.set("ctrlrange", f"-{effort:g} {effort:g}")
        motor.attrib.pop("ctrllimited", None)
        motor.set("ctrllimited", "true")
    # Clear the unit ctrlrange in <default><motor>
    for default in root.iter("default"):
        for motor in default.findall("motor"):
            motor.attrib.pop("ctrlrange", None)

    with open(args.output, "w") as fh:
        tree.write(fh, encoding="unicode", xml_declaration=False)

    # 4. Verify with mujoco
    m = mujoco.MjModel.from_xml_path(str(args.output))
    print(f"retuned mass: {m.body_mass.sum():.2f} kg "
          f"({m.body_mass.sum() * 2.2046:.1f} lb)")
    import numpy as np
    fr = m.jnt_actfrcrange if hasattr(m, "jnt_actfrcrange") else None
    print("actuator ctrlranges (Nm):",
          sorted(set(float(x) for x in m.actuator_ctrlrange[:, 1])))
    locked = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
              for j in range(m.njnt)
              if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
              and m.jnt_range[j][0] == m.jnt_range[j][1]]
    print("locked joints remaining:", locked or "none")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
