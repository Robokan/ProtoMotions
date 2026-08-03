# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""fbx2robot: build a MuJoCo robot from ANY rigged FBX skeleton.

Generalizes the samurai approach (rig2mjcf.py maps onto the fixed soma23
humanoid template) to arbitrary morphologies: the robot's topology comes
from the FBX skeleton itself, reduced by a per-robot KEEP-list. Skipped
bones collapse into the kept chain (offsets folded). Body frames are
WORLD-ALIGNED at the bind pose (soma convention: joint zero = bind pose,
hinges about global x,y,z), which makes the companion animation converter
(fbx_anim_to_motion.py) a pure world-rotation-copy.

Conventions (hard-won, see RAPTOR_TIGER_PLAN.md):
- 3 hinge joints (x,y,z) per body — IsaacLab merges multi-joint bodies
  into undriven D6s; chained single-hinge bodies or 1/3-hinge bodies only.
- capsule geoms along the parent->child bone axis (radius ~ bone length),
- explicit collision density scaled to --target-mass,
- EngineAI-class strength-to-weight actuators (~4 Nm/kg on primaries).

Run one FBX per process (ufbx: keepalive + single attribute family per
pass + gc.disable + os._exit).

Usage:
    python data/scripts/fbx2robot.py --robot raptor \
        --out protomotions/data/assets/mjcf/raptor.xml
"""
import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------
# Per-robot import specs: source FBX, kept joints (order = MJCF order),
# root joint, mass, and per-body tweaks.
# ----------------------------------------------------------------------
ROBOTS = {
    "raptor": dict(
        fbx="/home/bizon/sparkpack/UnrealExportedAssets/Raptor/Game/"
            "RaptorDinosaur/Model/Raptor_Gameplay.FBX",
        root="Hips",
        up_axis="z",
        # Full skeleton: every bone under the root (fingers, toes, jaw,
        # tongue, eyes...). --min-bone-length prunes short bones later.
        keep="all",
        target_mass=40.0,   # velociraptor-class at this stature
        radius_cap={"Head": 0.09, "Jaw": 0.04,
                    "LeftHand": 0.035, "RightHand": 0.035,
                    "LeftToeBase": 0.04, "RightToeBase": 0.04,
                    "Tail5": 0.035},
        radius_floor={"Hips": 0.10, "Spine": 0.09, "Spine1": 0.09},
        # leaf capsules need an explicit direction (no child to aim at):
        # world-frame axis at bind
        leaf_axis={"Head": (0, 1, 0), "Jaw": (0, 1, 0),
                   "LeftToeBase": (0, 1, 0), "RightToeBase": (0, 1, 0),
                   "LeftHand": (0, 1, 0), "RightHand": (0, 1, 0),
                   "Tail5": (0, -1, 0)},
        dof_ranges={"Jaw": ((-5, 30), (-5, 5), (-5, 5))},
    ),
    "tiger": dict(
        fbx="/home/bizon/sparkpack/UnrealExportedAssets/Tiger/Game/"
            "Animalia/Tiger_M/Meshes/Tiger_M.FBX",
        root="RigPelvis",
        # NOT "y": the Animalia FBX bakes a -90deg armature rotation into
        # the bone world transforms, so they are ALREADY z-up — applying
        # the y->z conversion double-rotated the tiger nose-down-vertical
        # (bind AND every converted motion). Verified on Drinking.motion:
        # identity transform puts all four ankles on the ground plane.
        up_axis="z",
        # Full skeleton (digits, claws, jaw, tongue, ears, whiskers...);
        # --min-bone-length prunes short bones later.
        keep="all",
        target_mass=200.0,
        radius_cap={"RigHead": 0.11, "RigJaw1": 0.05, "RigTail5": 0.035,
                    "RigLBLegAnkle": 0.045, "RigRBLegAnkle": 0.045,
                    "RigLFLegAnkle": 0.045, "RigRFLegAnkle": 0.045},
        radius_floor={"RigPelvis": 0.13, "RigChest": 0.14,
                      "RigSpine1": 0.12, "RigSpine3": 0.12},
        leaf_axis={"RigHead": (0, 1, 0), "RigJaw1": (0, 1, 0),
                   "RigTail5": (0, -1, 0),
                   "RigLBLegAnkle": (0, 1, 0), "RigRBLegAnkle": (0, 1, 0),
                   "RigLFLegAnkle": (0, 1, 0), "RigRFLegAnkle": (0, 1, 0)},
        dof_ranges={"RigJaw1": ((-5, 35), (-5, 5), (-5, 5))},
    ),
}

DEFAULT_RANGE = (-90, 90)  # v1; mine real ranges from the corpus later
MIN_RADIUS, MAX_RADIUS = 0.025, 0.14
RADIUS_PER_LEN = 0.22


def _q_to_mat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    n = x * x + y * y + z * z + w * w
    s = 2.0 / max(n, 1e-12)
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


def load_bind(fbx_path: str):
    """FBX -> {name: world_pos[3] (meters, source axes)} + parent map.

    ufbx rules: keepalive, single pass, anim fetched once, gc off.
    """
    import ufbx

    keep_alive = []
    scene = ufbx.load_file(fbx_path)
    keep_alive.append(scene)

    ordered, seen = [], set()

    def _add(n):
        if n is None or id(n) in seen:
            return
        _add(n.parent)
        seen.add(id(n))
        ordered.append(n)

    for n in scene.nodes:
        _add(n)
    parent_name = {}
    idx = {id(n): i for i, n in enumerate(ordered)}
    anim = scene.anim
    t0 = float(anim.time_begin)

    Rw = [np.eye(3)] * len(ordered)
    pw = [np.zeros(3)] * len(ordered)
    world = {}
    for i, node in enumerate(ordered):
        tr = node.evaluate_transform(anim, t0)
        Rl = _q_to_mat(tr.rotation)
        tl = np.array([tr.translation.x, tr.translation.y, tr.translation.z])
        sl = np.array([tr.scale.x, tr.scale.y, tr.scale.z])
        tl = tl * sl.mean() if abs(sl.mean() - 1.0) > 1e-6 else tl
        p = node.parent
        if p is None:
            Rw[i], pw[i] = Rl, tl
        else:
            j = idx[id(p)]
            Rw[i] = Rw[j] @ Rl
            pw[i] = pw[j] + Rw[j] @ tl
        name = node.name
        parent_name[name] = p.name if p is not None else None
        world[name] = pw[i].copy()
    return world, parent_name


def kept_parent(name, parent_name, keep_set):
    p = parent_name.get(name)
    while p is not None and p not in keep_set:
        p = parent_name.get(p)
    return p


def build(robot: str, out_path: Path, min_bone_length: float = 0.0,
          keep_bones=None, drop_bones=None):
    spec = ROBOTS[robot]
    world_raw, parent_name = load_bind(spec["fbx"])

    root = spec["root"]
    if spec["keep"] == "all":
        # every descendant of the root bone, tree order
        def under_root(n):
            p = n
            while p is not None:
                if p == root:
                    return True
                p = parent_name.get(p)
            return False
        keep = [n for n in world_raw if n and under_root(n)]
    else:
        keep = list(spec["keep"])

    # --min-bone-length pruning by LEAF EROSION: repeatedly delete LEAF
    # bones whose offset from their kept parent is below the threshold, so
    # chains erode from the tips (phalanges, tongue, eyes) and interior
    # bones (Head, hips, leg segments) can never fold away mid-chain.
    # --keep-bones protects named bones — and, automatically, their whole
    # ancestor chain (a protected bone never erodes, so its parents never
    # become prunable leaves).
    w_all = {k: v * 0.01 for k, v in world_raw.items()}
    protected = set(keep_bones or [])
    if min_bone_length > 0:
        keep_set_ = set(keep)
        changed = True
        while changed:
            changed = False
            has_child = set()
            for n in keep_set_:
                kp = kept_parent(n, parent_name, keep_set_)
                if kp is not None:
                    has_child.add(kp)
            for n in list(keep_set_):
                if n == root or n in has_child or n in protected:
                    continue
                kp = kept_parent(n, parent_name, keep_set_)
                if kp and np.linalg.norm(w_all[n] - w_all[kp]) < min_bone_length:
                    keep_set_.discard(n)
                    changed = True
        keep = [n for n in keep if n in keep_set_]
        print(f"min_bone_length={min_bone_length}: {len(keep)} bones kept")

    # --drop-bones: cull named subtrees regardless of length (whiskers,
    # eyes, eyelids — long offsets that leaf-erosion never removes but a
    # robot has no use for). Drops the bone AND all its descendants.
    if drop_bones:
        drop_roots = set(drop_bones)

        def dropped(n):
            p = n
            while p is not None:
                if p in drop_roots:
                    return True
                p = parent_name.get(p)
            return False
        before = len(keep)
        keep = [n for n in keep if not dropped(n)]
        print(f"drop_bones: removed {before - len(keep)} bones "
              f"({len(keep)} kept)")

    # Units/axes: UE exports are cm; convert to Z-up meters.
    w = {k: w_all[k] for k in keep}
    if spec["up_axis"] == "y":  # Y-up -> Z-up
        rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        w = {k: rot @ v for k, v in w.items()}
    print(f"root {root} world (m, z-up): {np.round(w[root], 3)}")

    keep_set = set(keep)
    kpar = {n: kept_parent(n, parent_name, keep_set) for n in keep}
    children = {n: [c for c in keep if kpar[c] == n] for n in keep}

    # capsule geometry per body
    def capsule(name):
        pos = w[name]
        kids = children[name]
        if kids:
            ends = [w[c] for c in kids]
            end = min(ends, key=lambda e: np.linalg.norm(e - pos)) \
                if name == root else np.mean(ends, axis=0)
        else:
            axis = spec["leaf_axis"].get(name)
            if axis is None:
                parent = kpar[name]
                d = pos - w[parent] if parent else np.array([0, 0, 1.0])
                n_ = np.linalg.norm(d)
                axis = d / n_ if n_ > 1e-6 else np.array([0, 0, 1.0])
            else:
                axis = np.array(axis, float)
            end = pos + axis * max(
                0.05, 0.6 * np.linalg.norm(
                    pos - w[kpar[name]]) if kpar[name] else 0.05)
        seg = end - pos
        length = np.linalg.norm(seg)
        radius = np.clip(RADIUS_PER_LEN * length, MIN_RADIUS, MAX_RADIUS)
        radius = max(radius, spec["radius_floor"].get(name, 0.0))
        radius = min(radius, spec["radius_cap"].get(name, MAX_RADIUS))
        # shrink the segment so end caps stay inside the joint span
        a = pos + seg * 0.12
        b = end - seg * 0.12 if length > 2 * radius else pos + seg * 0.5
        return a, b, radius

    lines = [
        f'<mujoco model="{robot}">',
        '  <compiler angle="radian" />',
        '  <option timestep="0.016667" />',
        '  <default>',
        '    <joint limited="true" armature="0.02" frictionloss="0.1" />',
        '    <geom type="capsule" density="1000" contype="1" conaffinity="1" '
        'friction="1.0 0.05 0.05" />',
        '    <motor ctrllimited="true" />',
        '  </default>',
        '  <worldbody>',
    ]
    indent = {root: 4}
    order = [root] + [n for n in keep if n != root]

    def emit(name, depth):
        pad = " " * depth
        pos = w[name]
        parent = kpar[name]
        rel = pos - w[parent] if parent else pos
        lines.append(
            f'{pad}<body name="{name}" pos="{rel[0]:.5g} {rel[1]:.5g} {rel[2]:.5g}">'
        )
        if parent is None:
            lines.append(f'{pad}  <freejoint name="root" />')
        else:
            ranges = spec["dof_ranges"].get(
                name, (DEFAULT_RANGE, DEFAULT_RANGE, DEFAULT_RANGE)
            )
            for ax, (axis, rng) in enumerate(
                zip(("1 0 0", "0 1 0", "0 0 1"), ranges)
            ):
                lo, hi = np.radians(rng[0]), np.radians(rng[1])
                lines.append(
                    f'{pad}  <joint name="{name}_{"xyz"[ax]}" type="hinge" '
                    f'axis="{axis}" pos="0 0 0" range="{lo:.4f} {hi:.4f}" />'
                )
        a, b, radius = capsule(name)
        fa, fb = a - pos, b - pos
        lines.append(
            f'{pad}  <geom name="g_{name}" fromto="{fa[0]:.5g} {fa[1]:.5g} '
            f'{fa[2]:.5g} {fb[0]:.5g} {fb[1]:.5g} {fb[2]:.5g}" '
            f'size="{radius:.4g}" />'
        )
        for c in children[name]:
            emit(c, depth + 2)
        lines.append(f"{pad}</body>")

    emit(root, 4)
    lines.append("  </worldbody>")

    # actuators: strength-to-weight by depth class (primaries near root)
    lines.append("  <actuator>")
    per_kg = 4.0 * spec["target_mass"]  # ~4 Nm/kg primary class
    for name in order:
        if name == root:
            continue
        depth = 0
        p = name
        while kpar[p] is not None:
            depth += 1
            p = kpar[p]
        effort = max(20.0, per_kg / (1.5 ** max(depth - 1, 0)) / 10.0)
        for ax in "xyz":
            lines.append(
                f'    <motor name="m_{name}_{ax}" joint="{name}_{ax}" '
                f'ctrlrange="-{effort:.4g} {effort:.4g}" />'
            )
    lines.append("  </actuator>")
    lines.append("</mujoco>")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({len(order)} bodies)")

    # verify + retarget density to hit target mass
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(out_path))
    mass = float(m.body_mass.sum())
    scale = spec["target_mass"] / mass
    text = out_path.read_text().replace(
        'density="1000"', f'density="{1000 * scale:.1f}"'
    )
    out_path.write_text(text)
    m = mujoco.MjModel.from_xml_path(str(out_path))
    print(
        f"mass {m.body_mass.sum():.1f} kg (target {spec['target_mass']}), "
        f"bodies {m.nbody - 1}, dofs {m.nv - 6}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot", choices=sorted(ROBOTS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--min-bone-length", type=float, default=0.0,
        help="Erode LEAF bones whose offset from their kept parent is "
        "shorter than this (meters). 0 = keep the full skeleton.",
    )
    ap.add_argument(
        "--keep-bones", default="",
        help="Comma list of bones protected from pruning (their ancestor "
        "chains survive automatically).",
    )
    ap.add_argument(
        "--drop-bones", default="",
        help="Comma list of bones to cull along with their whole subtree "
        "(applied after pruning; for whiskers/eyes/etc.).",
    )
    args = ap.parse_args()
    gc.disable()
    build(args.robot, args.out, args.min_bone_length,
          [b.strip() for b in args.keep_bones.split(",") if b.strip()],
          [b.strip() for b in args.drop_bones.split(",") if b.strip()])
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
