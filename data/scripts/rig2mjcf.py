# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""rig2mjcf: convert a rigged UsdSkel character into a MuJoCo humanoid.

Generates a 23-body robot with soma23_humanoid.xml's exact topology, joint
stack, gains and actuators (proven RL-trainable), but the CHARACTER's
proportions, per-bone capsule geometry (fit from skin weights) and masses.
Body names are the SOMA names; frames are world-aligned at the T-pose zero
configuration (soma convention), with A-pose arm binds re-posed to T via
the rest-rel constants.

Run (needs pxr + numpy; anaconda base has both):
    ~/anaconda3/bin/python data/scripts/rig2mjcf.py \
        --character protomotions/data/assets/overlay/red_samurai.usd \
        --template protomotions/data/assets/mjcf/soma23_humanoid.xml \
        --out protomotions/data/assets/mjcf/samurai.xml

See SAMURAI_ROBOT_PLAN.md for the full pipeline this belongs to.
"""
import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from protomotions.simulator.isaaclab.overlay_map import (  # noqa: E402
    SOMA23_TO_UE,
    UE_REST_REL,
    SOMA23_TPOSE_POS,
    SOMA23_PARENT_UE,
)

TARGET_MASS_KG = 75.0
MIN_RADIUS = 0.03
MAX_RADIUS = 0.16
# Vertices owned by these bone-name substrings are ignored entirely
# (props that would otherwise inflate their ancestor's capsule).
EXCLUDE_BONES = ("weapon",)
# Per-body radius overrides (meters): floors fix thin-vertex fallbacks,
# caps keep strike surfaces honest (fist-size hands) and trim plumes.
RADIUS_FLOOR = {"Chest": 0.10, "Spine1": 0.08, "Spine2": 0.08,
                "Neck1": 0.04, "Neck2": 0.04}
RADIUS_CAP = {"Head": 0.12, "Hips": 0.14, "LeftHand": 0.05,
              "RightHand": 0.05, "LeftFoot": 0.06, "RightFoot": 0.06}
# Capsule AXIS overrides in the T-pose WORLD frame: leaf bodies have no
# child to infer a direction from (Head defaulted horizontal), and the
# Hips' mean-of-children axis is meaningless.
AXIS_WORLD = {"Hips": (0, 0, 1), "Head": (0, 0, 1),
              "LeftHand": (1, 0, 0), "RightHand": (-1, 0, 0),
              "LeftToeBase": (0, -1, 0), "RightToeBase": (0, -1, 0)}
# Extent clamps along the capsule axis (meters, from the joint origin, in
# T-world units): the Hips otherwise swallow the skirt/torso vertex cloud
# (capsule reached the ribcage and poked out the back when bending) and
# the Head grabs helmet plume.
EXTENT_CLAMP = {"Hips": (-0.10, 0.13), "Head": (-0.02, 0.16)}

# Anatomical hinge ranges (degrees), mined from the SOMA v6 corpus dof
# tracks (0.1/99.9 percentiles + 15% margin): same 66-dof layout and sign
# convention, so the human data defines each hinge's true range. Without
# these, GMR's IK parks joints on the wrong ±pi Euler branch (seen on the
# first samurai sanity retarget: RightArm_x pinned at pi for 44% of frames).
DOF_RANGES = {
    "Spine1_x": (-18, 39), "Spine1_y": (-13, 14), "Spine1_z": (-13, 12),
    "Spine2_x": (-18, 34), "Spine2_y": (-14, 14), "Spine2_z": (-22, 22),
    "Chest_x": (-7, 47), "Chest_y": (-18, 22), "Chest_z": (-24, 22),
    "Neck1_x": (-54, 57), "Neck1_y": (-37, 36), "Neck1_z": (-35, 36),
    "Neck2_x": (-7, 3), "Neck2_y": (-14, 15), "Neck2_z": (-39, 39),
    "Head_x": (-69, 43), "Head_y": (-25, 25), "Head_z": (-69, 70),
    "RightShoulder_x": (-13, 34), "RightShoulder_y": (-42, 57),
    "RightShoulder_z": (-32, 49),
    "RightArm_x": (-104, 78), "RightArm_y": (-115, 62),
    "RightArm_z": (-63, 134),
    "RightForeArm_x": (-4, 10), "RightForeArm_y": (-18, 35),
    "RightForeArm_z": (-31, 178),
    "RightHand_x": (-178, 122), "RightHand_y": (-43, 99),
    "RightHand_z": (-61, 67),
    "LeftShoulder_x": (-14, 33), "LeftShoulder_y": (-57, 42),
    "LeftShoulder_z": (-45, 32),
    "LeftArm_x": (-103, 77), "LeftArm_y": (-62, 108),
    "LeftArm_z": (-134, 65),
    "LeftForeArm_x": (-4, 10), "LeftForeArm_y": (-33, 19),
    "LeftForeArm_z": (-178, 32),
    "LeftHand_x": (-108, 85), "LeftHand_y": (-99, 42),
    "LeftHand_z": (-55, 45),
    "RightLeg_x": (-144, 54), "RightLeg_y": (-61, 68),
    "RightLeg_z": (-66, 50),
    "RightShin_x": (-37, 178), "RightShin_y": (-15, 7),
    "RightShin_z": (-10, 4),
    "RightFoot_x": (-56, 81), "RightFoot_y": (-55, 41),
    "RightFoot_z": (-65, 55),
    "RightToeBase_x": (-70, 19), "RightToeBase_y": (-13, 10),
    "RightToeBase_z": (-7, 10),
    "LeftLeg_x": (-144, 55), "LeftLeg_y": (-68, 53),
    "LeftLeg_z": (-49, 71),
    "LeftShin_x": (-36, 178), "LeftShin_y": (-7, 15),
    "LeftShin_z": (-4, 10),
    "LeftFoot_x": (-55, 81), "LeftFoot_y": (-44, 56),
    "LeftFoot_z": (-60, 65),
    "LeftToeBase_x": (-70, 19), "LeftToeBase_y": (-10, 13),
    "LeftToeBase_z": (-10, 7),
}


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_character(usd_path):
    from pxr import Usd, UsdSkel, UsdGeom, Gf  # noqa: F401

    stage = Usd.Stage.Open(str(usd_path))
    skel_prim = next(p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton))
    skel = UsdSkel.Skeleton(skel_prim)
    joints = list(skel.GetJointsAttr().Get())
    leaf = [j.split("/")[-1] for j in joints]
    bind = skel.GetBindTransformsAttr().Get()
    bind_world = np.array(
        [[list(m[i]) for i in range(4)] for m in bind], dtype=np.float64
    )
    # skel-space == prim space here (skel l2w measured identity for these
    # assets); if not, compose l2w the way overlay.py does.
    idx = {j: i for i, j in enumerate(joints)}
    parents = np.full(len(joints), -1, dtype=np.int64)
    for i, j in enumerate(joints):
        p = "/".join(j.split("/")[:-1])
        parents[i] = idx.get(p, -1)

    # skinned vertices in skel space, tagged by dominant joint
    pts_all, own_all = [], []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        b = UsdSkel.BindingAPI(prim)
        ji = b.GetJointIndicesPrimvar()
        if not ji or not ji.GetAttr().HasValue():
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
        gbt = b.GetGeomBindTransformAttr()
        if gbt and gbt.HasValue():
            m = np.array(
                [[gbt.Get()[i][j] for j in range(4)] for i in range(4)]
            )
            pts = pts @ m[:3, :3] + m[3, :3]
        es = ji.GetElementSize()
        jidx = np.array(ji.Get(), dtype=np.int64).reshape(-1, es)
        if es > 1:
            jw = np.array(
                b.GetJointWeightsPrimvar().Get(), dtype=np.float64
            ).reshape(-1, es)
            dom = jidx[np.arange(len(jidx)), jw.argmax(1)]
        else:
            dom = jidx[:, 0]
        # meshes may carry a local joint order (skel:joints on the mesh)
        mj = b.GetJointsAttr()
        if mj and mj.Get():
            remap = np.array([idx[j] for j in mj.Get()], dtype=np.int64)
            dom = remap[dom]
        pts_all.append(pts)
        own_all.append(dom)
    return {
        "joints": joints, "leaf": leaf, "parents": parents,
        "bind_world": bind_world,
        "pts": np.concatenate(pts_all), "own": np.concatenate(own_all),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-mass", type=float, default=TARGET_MASS_KG)
    args = ap.parse_args()

    ch = load_character(args.character)
    leaf, parents, bind = ch["leaf"], ch["parents"], ch["bind_world"]

    # ---- articulated subset: soma name -> character joint index ----------
    soma_names = list(SOMA23_TO_UE.keys())
    art = {}
    for soma, bone in SOMA23_TO_UE.items():
        if bone not in leaf:
            raise SystemExit(f"bone {bone} (for {soma}) not in skeleton")
        art[soma] = leaf.index(bone)
    art_set = set(art.values())
    ji_to_soma = {v: k for k, v in art.items()}

    # every joint -> nearest articulated ancestor (for vertex ownership)
    anc = np.full(len(leaf), -1, dtype=np.int64)
    for j in range(len(leaf)):
        if any(t in leaf[j].lower() for t in EXCLUDE_BONES):
            anc[j] = -1  # prop bones: drop their vertices
            continue
        k = j
        while k >= 0 and k not in art_set:
            k = parents[k]
        anc[j] = k
    vert_owner = anc[ch["own"]]

    # ---- fit scale: similarity fit of hip-relative bind joint positions --
    hip_ji = art["Hips"]
    hip_pos = bind[hip_ji, 3, :3]
    C, R = [], []
    for soma, ji in art.items():
        if soma in UE_REST_REL:  # A-posed bodies are not T-pose samples
            continue
        C.append(bind[ji, 3, :3] - hip_pos)
        R.append(np.asarray(SOMA23_TPOSE_POS[soma]))
    C, R = np.stack(C), np.stack(R)
    s = float((C * R).sum() / max((C * C).sum(), 1e-9))
    print(f"fit scale: {s:.5f} (1/{1/s:.1f})")

    # ---- T-pose re-pose (A-pose arm chains -> T) --------------------------
    # world rot/pos per articulated body at the robot's zero (T) pose
    rotT, posT = {}, {}
    order = [s_ for s_ in soma_names]  # SOMA23_TO_UE is ordered parent-first
    for soma in order:
        ji = art[soma]
        c = np.asarray(UE_REST_REL.get(soma, (1.0, 0, 0, 0)))
        Rc = quat_to_mat(np.array([c[0], -c[1], -c[2], -c[3]]))  # conj(c_b)
        rotT[soma] = Rc @ bind[ji, :3, :3].T  # column-vector world rotation
        pb = SOMA23_PARENT_UE.get(soma)
        if pb is None or pb not in art:
            posT[soma] = bind[ji, 3, :3].copy()
        else:
            cp = np.asarray(UE_REST_REL.get(pb, (1.0, 0, 0, 0)))
            Rcp = quat_to_mat(np.array([cp[0], -cp[1], -cp[2], -cp[3]]))
            d = bind[ji, 3, :3] - bind[art[pb], 3, :3]
            posT[soma] = posT[pb] + Rcp @ d

    # ---- capsule fit per body (bone-local frame; invariant to re-pose) ---
    children = {}
    for soma, pb in SOMA23_PARENT_UE.items():
        children.setdefault(pb, []).append(soma)
    geoms = {}
    dens_vols = {}
    for soma, ji in art.items():
        sel = vert_owner == ji
        v = ch["pts"][sel]
        Rb = bind[ji, :3, :3].T          # column rotation, bone -> world
        tb = bind[ji, 3, :3]
        vl = (v - tb) @ Rb               # world -> bone local (R^T = R^-1)
        kid = children.get(soma, [])
        if soma in AXIS_WORLD:
            ax_w = np.asarray(AXIS_WORLD[soma], dtype=np.float64)
            # give it a nominal length for the no-vertex fallback path
            axis_l = rotT[soma].T @ (ax_w * 0.15 / s)
        elif kid:
            ke = np.mean(
                [(posT[k] - posT[soma]) for k in kid], axis=0)
            # express T-pose child direction in the T-posed bone frame
            axis_l = rotT[soma].T @ ke
        else:
            axis_l = np.array([0.0, 1.0, 0.0])
        L = np.linalg.norm(axis_l)
        axis_n = axis_l / max(L, 1e-9)
        if len(vl) < 50:
            a, b_, rad = np.zeros(3), axis_l, 0.04 / s
        else:
            t = vl @ axis_n
            lo, hi = np.percentile(t, 2), np.percentile(t, 98)
            if kid and soma not in AXIS_WORLD:
                hi = max(hi, L)
                lo = max(lo, -0.15 * L)
            if soma in EXTENT_CLAMP:
                clo, chi = EXTENT_CLAMP[soma]
                lo, hi = max(lo, clo / s), min(hi, chi / s)
            radial = np.linalg.norm(
                vl - np.outer(t, axis_n), axis=1)
            rad = np.percentile(radial, 80)
            a, b_ = axis_n * lo, axis_n * hi
        rad_m = float(np.clip(
            rad * s,
            RADIUS_FLOOR.get(soma, MIN_RADIUS),
            RADIUS_CAP.get(soma, MAX_RADIUS)))
        # endpoints in the world-aligned T body frame, meters, capsule
        # shrunk so length = seg - 2r (capsule total length incl. caps)
        aw = rotT[soma] @ a * s
        bw = rotT[soma] @ b_ * s
        seg = bw - aw
        sl = np.linalg.norm(seg)
        if sl > 2.2 * rad_m:
            sn = seg / sl
            aw = aw + sn * rad_m
            bw = bw - sn * rad_m
        else:  # short/blobby bodies -> sphere-ish capsule
            mid = (aw + bw) / 2
            aw = mid - np.array([0, 0, 1e-3])
            bw = mid + np.array([0, 0, 1e-3])
        geoms[soma] = (aw, bw, rad_m)
        h = max(np.linalg.norm(bw - aw), 2e-3)
        dens_vols[soma] = np.pi * rad_m**2 * h + 4 / 3 * np.pi * rad_m**3

    total_vol = sum(dens_vols.values())
    density = args.total_mass / total_vol
    print(f"capsule volume total {total_vol*1000:.1f} L -> density "
          f"{density:.0f} kg/m3 for {args.total_mass:.0f} kg")

    # ---- emit MJCF from the soma23 template -------------------------------
    src = Path(args.template).read_text()

    def body_offset(soma):
        pb = SOMA23_PARENT_UE.get(soma)
        if pb is None:
            return np.zeros(3)
        return (posT[soma] - posT[pb]) * s

    out = src
    out = re.sub(r"<mujoco model='[^']*'|<mujoco model=\"[^\"]*\"",
                 "<mujoco model=\"samurai\"", out, count=1)
    for soma in soma_names:
        if soma == "Hips":
            continue
        off = body_offset(soma)
        out = re.sub(
            rf"(<body name='{soma}' pos=')[^']*(')",
            lambda m, o=off: f"{m.group(1)}{o[0]:.4f} {o[1]:.4f} {o[2]:.4f}"
                             f"{m.group(2)}",
            out, count=1)
    # replace each body's first geom with the fitted capsule
    for soma in soma_names:
        aw, bw, rad = geoms[soma]
        cap = (f"<geom type='capsule' size='{rad:.4f}' "
               f"fromto='{aw[0]:.4f} {aw[1]:.4f} {aw[2]:.4f} "
               f"{bw[0]:.4f} {bw[1]:.4f} {bw[2]:.4f}' "
               f"density='{density:.0f}' material='geom'/>")
        # first geom following this body's tag
        pat = rf"(<body name='{soma}'[^>]*>(?:\s*<freejoint[^/]*/>)?" \
              rf"(?:\s*<joint[^/]*/>)*\s*)<geom[^/]*/>"
        out2 = re.sub(pat, lambda m, c=cap: m.group(1) + c, out, count=1)
        if out2 == out:
            print(f"WARN: geom not replaced for {soma}")
        out = out2
    # anatomical joint ranges (see DOF_RANGES docstring) + slack: the robot's
    # proportions differ from the human's, so the IK legitimately needs to
    # exceed human dof extremes; 30 deg of slack per side keeps that freedom
    # while still fencing off the +-pi wrong-Euler-branch region.
    RANGE_PAD = 30.0
    for jname, (lo, hi) in DOF_RANGES.items():
        a, b = max(lo - RANGE_PAD, -178.0), min(hi + RANGE_PAD, 178.0)
        out, n = re.subn(
            rf"(<joint name='{jname}'[^>]*range=')[^']*(')",
            lambda m, a=a, b=b: f"{m.group(1)}{a:.4f} {b:.4f}{m.group(2)}",
            out, count=1)
        if n == 0:
            print(f"WARN: no range replaced for joint {jname}")
    Path(args.out).write_text(out)
    print(f"wrote {args.out}")
    # summary
    for soma in soma_names:
        aw, bw, rad = geoms[soma]
        print(f"  {soma:14s} off {np.round(body_offset(soma),3)} "
              f"r={rad:.3f} len={np.linalg.norm(bw-aw)+2*rad:.3f}")


if __name__ == "__main__":
    main()
