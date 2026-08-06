# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Geometrically scale an MJCF robot to a target mass.

Scaling a creature is not "multiply everything by s". Each quantity has a
dimension and scales by its own power, and getting one wrong produces a
robot that looks right and behaves nothing like it should:

    length  (pos, size, fromto)   s^1     geometry
    mass                          s^3     from volume, at fixed density
    torque  (ctrlrange, friction- s^4     m*g*L
             loss)
    inertia (armature)            s^5     m*L^2
    angle   (range)               s^0     dimensionless -- NOT scaled
    density                       s^0     unchanged; it is what makes mass s^3

So a robot 1.71x longer is 5x heavier and needs 8.5x the joint torque just
to hold the same pose against gravity. Scaling geometry without scaling
torque gives a giant that collapses under itself.

    python data/scripts/scale_robot_mjcf.py \
        --in-mjcf protomotions/data/assets/mjcf/raptor.xml \
        --out-mjcf protomotions/data/assets/mjcf/utahraptor.xml \
        --target-mass 200 --model-name utahraptor
"""
from __future__ import annotations

import argparse
import re

import mujoco

# attribute -> exponent of the scale factor
_LENGTH = ("pos", "size", "fromto")
_TORQUE = ("ctrlrange", "frictionloss")
_INERTIA = ("armature",)


def _scale_numbers(text: str, factor: float) -> str:
    """Multiply every float in a whitespace-separated attribute value."""
    out = []
    for tok in text.split():
        try:
            out.append(f"{float(tok) * factor:.6g}")
        except ValueError:
            out.append(tok)
    return " ".join(out)


def scale_mjcf(src: str, s: float, model_name: str | None) -> str:
    def sub_attr(attr: str, factor: float, text: str) -> str:
        return re.sub(
            rf'{attr}="([^"]*)"',
            lambda m: f'{attr}="{_scale_numbers(m.group(1), factor)}"',
            text,
        )

    out = src
    for a in _LENGTH:
        out = sub_attr(a, s, out)
    for a in _TORQUE:
        out = sub_attr(a, s ** 4, out)
    for a in _INERTIA:
        out = sub_attr(a, s ** 5, out)
    # 'range' is joint limits in radians and must NOT be touched; it is left
    # alone by construction since it is not in any list above.
    if model_name:
        out = re.sub(r'<mujoco model="[^"]*"', f'<mujoco model="{model_name}"', out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-mjcf", required=True)
    ap.add_argument("--out-mjcf", required=True)
    ap.add_argument("--target-mass", type=float, required=True)
    ap.add_argument("--model-name", default=None)
    args = ap.parse_args()

    src = open(args.in_mjcf).read()
    m0 = mujoco.MjModel.from_xml_path(args.in_mjcf)
    mass0 = float(m0.body_mass.sum())

    # mass goes as s^3 at fixed density
    s = (args.target_mass / mass0) ** (1.0 / 3.0)
    print(f"source mass {mass0:.2f} kg -> target {args.target_mass:.2f} kg")
    print(f"  length scale s = {s:.6f}   (mass s^3 = {s**3:.3f}, "
          f"torque s^4 = {s**4:.3f}, inertia s^5 = {s**5:.3f})")

    open(args.out_mjcf, "w").write(scale_mjcf(src, s, args.model_name))

    m1 = mujoco.MjModel.from_xml_path(args.out_mjcf)
    n0 = [mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m0.nbody)]
    n1 = [mujoco.mj_id2name(m1, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m1.nbody)]
    d0, d1 = mujoco.MjData(m0), mujoco.MjData(m1)
    mujoco.mj_forward(m0, d0)
    mujoco.mj_forward(m1, d1)
    print(f"\nresult mass {m1.body_mass.sum():.2f} kg "
          f"({m1.nbody-1} bodies, {m1.nu} actuators)")
    hi0 = d0.xpos[:, 2].max() - d0.xpos[:, 2].min()
    hi1 = d1.xpos[:, 2].max() - d1.xpos[:, 2].min()
    print(f"rest-pose height {hi0*100:.1f} -> {hi1*100:.1f} cm "
          f"(ratio {hi1/hi0:.4f}, expected {s:.4f})")
    # a couple of spot checks that the shape is preserved, not just the size
    for b in ("LeftUpLeg", "Head", "Tail"):
        if b in n0 and b in n1:
            g0 = next(i for i in range(m0.ngeom) if n0[m0.geom_bodyid[i]] == b)
            g1 = next(i for i in range(m1.ngeom) if n1[m1.geom_bodyid[i]] == b)
            print(f"  {b:<10} radius {m0.geom_size[g0,0]*100:5.2f} -> "
                  f"{m1.geom_size[g1,0]*100:5.2f} cm   mass "
                  f"{m0.body_mass[n0.index(b)]:6.2f} -> "
                  f"{m1.body_mass[n1.index(b)]:7.2f} kg")
    print(f"\nwritten to {args.out_mjcf}")


main()
