#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Vendor official ANYmal-D and Go2 URDFs, stripped to the motion skeleton.

The manufacturer files carry cameras, rotors, and payload joints that do not
exist in ProtoMotions clips. This script copies official meshes and writes a
17-link URDF whose remaining body/joint names match mjcf/anymal_d_nomesh.xml
and mjcf/go2_nomesh.xml so Newton and Isaac Gym can play the same motions.

Sources (cloned to /tmp by default):
  ANYbotics/anymal_d_simple_description  (BSD-3-Clause)
  unitreerobotics/unitree_ros robots/go2_description
"""

from __future__ import annotations

import argparse
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def _rpy_to_mat(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _mat_to_rpy(m):
    sy = math.hypot(m[0][0], m[1][0])
    if sy > 1e-8:
        roll = math.atan2(m[2][1], m[2][2])
        pitch = math.atan2(-m[2][0], sy)
        yaw = math.atan2(m[1][0], m[0][0])
    else:
        roll = math.atan2(-m[1][2], m[1][1])
        pitch = math.atan2(-m[2][0], sy)
        yaw = 0.0
    return (roll, pitch, yaw)


def _matmul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _matvec(m, v):
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))


def compose_origin(parent, child):
    """Compose two URDF origins: T_parent * T_child."""
    pm, pc = _rpy_to_mat(parent[3:]), parent[:3]
    cm, cc = _rpy_to_mat(child[3:]), child[:3]
    xyz = tuple(pc[i] + _matvec(pm, cc)[i] for i in range(3))
    return xyz + _mat_to_rpy(_matmul(pm, cm))


def _parse_origin(elem):
    if elem is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    xyz = tuple(float(x) for x in elem.get("xyz", "0 0 0").split())
    rpy = tuple(float(x) for x in elem.get("rpy", "0 0 0").split())
    return xyz + rpy


def _set_origin(elem, origin):
    xyz = elem.find("origin")
    if xyz is None:
        xyz = ET.SubElement(elem, "origin")
    xyz.set("xyz", f"{origin[0]:.8g} {origin[1]:.8g} {origin[2]:.8g}")
    xyz.set("rpy", f"{origin[3]:.8g} {origin[4]:.8g} {origin[5]:.8g}")


def _rewrite_mesh(filename: str, mesh_prefix: str) -> str:
    name = filename.split("/")[-1]
    return f"{mesh_prefix}/{name}"


def strip_urdf(
    src: Path,
    dst: Path,
    keep_links: list[str],
    *,
    rename: dict[str, str] | None = None,
    mesh_prefix: str = "meshes",
    robot_name: str | None = None,
) -> None:
    rename = rename or {}
    tree = ET.parse(src)
    root = tree.getroot()
    if robot_name:
        root.set("name", robot_name)

    def name_of(link: str) -> str:
        return rename.get(link, link)

    links = {}
    for link in list(root.findall("link")):
        links[link.get("name")] = link

    joints = []
    child_to_joint = {}
    for joint in list(root.findall("joint")):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        rec = {
            "elem": joint,
            "name": joint.get("name"),
            "type": joint.get("type"),
            "parent": parent,
            "child": child,
            "origin": _parse_origin(joint.find("origin")),
        }
        joints.append(rec)
        child_to_joint[child] = rec

    # Build new tree: start fresh robot element.
    new_root = ET.Element("robot", {"name": root.get("name", "robot")})

    for src_name in links:
        dst_name = name_of(src_name)
        if dst_name not in keep_links:
            continue
        link_el = links[src_name]
        new_link = ET.SubElement(new_root, "link", {"name": dst_name})
        for child in list(link_el):
            new_link.append(_copy_with_meshes(child, mesh_prefix))

        # Fold visuals/collisions of dropped descendants whose nearest kept
        # ancestor is this link, walking only fixed joints.
        for other, other_el in links.items():
            if name_of(other) in keep_links:
                continue
            acc = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            cur = other
            all_fixed = True
            ancestor = None
            while cur in child_to_joint:
                j = child_to_joint[cur]
                acc = compose_origin(j["origin"], acc)
                if j["type"] != "fixed":
                    all_fixed = False
                    break
                if name_of(j["parent"]) in keep_links:
                    ancestor = name_of(j["parent"])
                    break
                cur = j["parent"]
            if not all_fixed or ancestor != dst_name:
                continue
            for geom_tag in ("visual", "collision"):
                for geom in other_el.findall(geom_tag):
                    copied = _copy_with_meshes(geom, mesh_prefix)
                    g_origin = _parse_origin(copied.find("origin"))
                    _set_origin(copied, compose_origin(acc, g_origin))
                    new_link.append(copied)

    kept_set = set(keep_links)
    for rec in joints:
        child_dst = name_of(rec["child"])
        if child_dst not in kept_set:
            continue
        # nearest kept ancestor of parent
        parent_src = rec["parent"]
        composed = rec["origin"]
        while name_of(parent_src) not in kept_set:
            if parent_src not in child_to_joint:
                parent_src = None
                break
            pj = child_to_joint[parent_src]
            composed = compose_origin(pj["origin"], composed)
            parent_src = pj["parent"]
        if parent_src is None:
            continue
        parent_dst = name_of(parent_src)
        new_j = ET.SubElement(
            new_root,
            "joint",
            {"name": rec["name"], "type": rec["type"]},
        )
        _set_origin(new_j, composed)
        ET.SubElement(new_j, "parent", {"link": parent_dst})
        ET.SubElement(new_j, "child", {"link": child_dst})
        for tag in ("axis", "limit", "dynamics", "safety_controller"):
            old = rec["elem"].find(tag)
            if old is not None:
                new_j.append(_copy_elem(old))

    dst.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(new_root, space="  ")
    tree = ET.ElementTree(new_root)
    tree.write(dst, encoding="utf-8", xml_declaration=True)


def _copy_elem(elem: ET.Element) -> ET.Element:
    return ET.fromstring(ET.tostring(elem))


def _copy_with_meshes(elem: ET.Element, mesh_prefix: str) -> ET.Element:
    copied = _copy_elem(elem)
    for mesh in copied.iter("mesh"):
        fn = mesh.get("filename")
        if fn:
            mesh.set("filename", _rewrite_mesh(fn, mesh_prefix))
    return copied


ANYMAL_LINKS = [
    "base",
    "LF_HIP",
    "LF_THIGH",
    "LF_SHANK",
    "LF_FOOT",
    "RF_HIP",
    "RF_THIGH",
    "RF_SHANK",
    "RF_FOOT",
    "LH_HIP",
    "LH_THIGH",
    "LH_SHANK",
    "LH_FOOT",
    "RH_HIP",
    "RH_THIGH",
    "RH_SHANK",
    "RH_FOOT",
]

GO2_LINKS = [
    "base_link",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FL_foot",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FR_foot",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RL_foot",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RR_foot",
]


def _dae_to_obj_and_rewrite(urdf_path: Path, mesh_dir: Path) -> None:
    """Newton's DAE loader chokes on some ANYmal Collada files; OBJ is reliable."""
    try:
        import trimesh
    except ImportError:
        print("trimesh not available; leaving DAE meshes as-is")
        return

    converted = set()
    for dae in sorted(mesh_dir.glob("*.dae")):
        scene = trimesh.load(str(dae), force="mesh")
        verts = getattr(scene, "vertices", None)
        if verts is None or len(verts) < 3:
            continue
        obj = dae.with_suffix(".obj")
        scene.export(str(obj))
        converted.add(dae.name)

    root = ET.parse(urdf_path).getroot()
    for mesh in list(root.iter("mesh")):
        fn = mesh.get("filename", "")
        if not fn.endswith(".dae"):
            continue
        obj_rel = fn[:-4] + ".obj"
        if (urdf_path.parent / obj_rel).is_file():
            mesh.set("filename", obj_rel)
    # Drop refs to meshes that were empty / missing.
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for geom in list(link.findall(tag)):
                mesh = geom.find("geometry/mesh")
                if mesh is None:
                    continue
                if not (urdf_path.parent / mesh.get("filename", "")).is_file():
                    link.remove(geom)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(urdf_path, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anymal-src",
        default="/tmp/anymal_d_simple_description",
        type=Path,
    )
    parser.add_argument(
        "--go2-src",
        default="/tmp/unitree_ros/robots/go2_description",
        type=Path,
    )
    parser.add_argument(
        "--dest",
        default=Path(__file__).resolve().parents[2]
        / "protomotions/data/assets/urdf",
        type=Path,
    )
    args = parser.parse_args()

    anymal_dst = args.dest / "anymal_d"
    anymal_dst.mkdir(parents=True, exist_ok=True)
    mesh_dst = anymal_dst / "meshes"
    if mesh_dst.exists():
        shutil.rmtree(mesh_dst)
    shutil.copytree(args.anymal_src / "meshes", mesh_dst)
    shutil.copy2(args.anymal_src / "LICENSE", anymal_dst / "LICENSE")
    strip_urdf(
        args.anymal_src / "urdf" / "anymal.urdf",
        anymal_dst / "anymal.urdf",
        ANYMAL_LINKS,
        mesh_prefix="meshes",
        robot_name="anymal_d",
    )
    _dae_to_obj_and_rewrite(anymal_dst / "anymal.urdf", mesh_dst)

    go2_dst = args.dest / "go2"
    go2_dst.mkdir(parents=True, exist_ok=True)
    dae_dst = go2_dst / "dae"
    if dae_dst.exists():
        shutil.rmtree(dae_dst)
    shutil.copytree(args.go2_src / "dae", dae_dst)
    readme = args.go2_src / "README.md"
    if readme.exists():
        shutil.copy2(readme, go2_dst / "README.md")
    strip_urdf(
        args.go2_src / "urdf" / "go2_description.urdf",
        go2_dst / "go2.urdf",
        GO2_LINKS,
        rename={"base": "base_link"},
        mesh_prefix="dae",
        robot_name="go2",
    )

    print(f"Wrote {anymal_dst / 'anymal.urdf'}")
    print(f"Wrote {go2_dst / 'go2.urdf'}")


if __name__ == "__main__":
    main()
