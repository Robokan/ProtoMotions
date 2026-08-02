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
"""Generate a primitive-geometry MJCF for ANYmal-D.

Body offsets come from the poselib skeleton embedded in the retargeted motion
data (which is what the clips were retargeted to), so FK through this MJCF
reproduces the motion data exactly. Joint axes follow the data convention
(world-aligned frames at rest): HAA about +x, HFE/KFE about +y.

Masses are lumped from the official ANYbotics anymal_d_simple_description URDF.
"""

import argparse

# Skeleton offsets extracted from the retargeted motion NPY skeleton_tree
# (identical across clips). Format: body -> (parent, [x, y, z]).
SKELETON = {
    "LF_HIP": ("base", [0.3040, 0.1090, 0.0]),
    "LF_THIGH": ("LF_HIP", [0.0690, 0.0060, 0.0]),
    "LF_SHANK": ("LF_THIGH", [0.0, 0.1805, -0.2850]),
    "LF_FOOT": ("LF_SHANK", [0.1000, 0.0222, -0.3925]),
    "LH_HIP": ("base", [-0.3040, 0.1090, 0.0]),
    "LH_THIGH": ("LH_HIP", [-0.0690, 0.0060, 0.0]),
    "LH_SHANK": ("LH_THIGH", [0.0, 0.1805, -0.2850]),
    "LH_FOOT": ("LH_SHANK", [-0.1000, 0.0223, -0.3925]),
    "RF_HIP": ("base", [0.3040, -0.1090, 0.0]),
    "RF_THIGH": ("RF_HIP", [0.0690, -0.0060, 0.0]),
    "RF_SHANK": ("RF_THIGH", [0.0, -0.1805, -0.2850]),
    "RF_FOOT": ("RF_SHANK", [0.1000, -0.0222, -0.3925]),
    "RH_HIP": ("base", [-0.3040, -0.1090, 0.0]),
    "RH_THIGH": ("RH_HIP", [-0.0690, -0.0060, 0.0]),
    "RH_SHANK": ("RH_THIGH", [-0.0000, -0.1805, -0.2850]),
    "RH_FOOT": ("RH_SHANK", [-0.1000, -0.0222, -0.3925]),
}

# Approximate lumped masses [kg] from anymal_d_simple_description
# (drives/adapters merged into their parent segment).
MASSES = {"base": 28.0, "HIP": 2.781, "THIGH": 3.071, "SHANK": 0.58}

EFFORT = 80.0  # ANYdrive 3.0 [Nm]


def leg(prefix: str) -> str:
    """Emit one leg subtree (HIP -> THIGH -> SHANK -> FOOT)."""
    hip_p = SKELETON[f"{prefix}_HIP"][1]
    thigh_p = SKELETON[f"{prefix}_THIGH"][1]
    shank_p = SKELETON[f"{prefix}_SHANK"][1]
    foot_p = SKELETON[f"{prefix}_FOOT"][1]

    # Limits must match the IsaacLab USD exactly (ProtoMotions validates them
    # against the simulator with 1e-5 tolerance — full URDF precision required).
    # From anymal_d_simple_description URDF, world-frame convention:
    # left legs HAA [-0.7853985, 0.6108655], right legs mirrored; HFE/KFE multi-turn.
    haa_range = (
        "-0.7853985 0.6108655" if prefix.startswith("L") else "-0.6108655 0.7853985"
    )
    pitch_range = "-9.42477796076938 9.42477796076938"

    def v(p):
        return f"{p[0]} {p[1]} {p[2]}"

    return f"""      <body name="{prefix}_HIP" pos="{v(hip_p)}">
        <inertial pos="0 0 0" mass="{MASSES['HIP']}" diaginertia="0.005 0.005 0.005" />
        <joint name="{prefix}_HAA" axis="1 0 0" range="{haa_range}"
          limited="true" damping="0.1" armature="0.01" frictionloss="0.2" />
        <geom size="0.06 0.04" quat="0.707107 0.707107 0 0" type="cylinder" />
        <body name="{prefix}_THIGH" pos="{v(thigh_p)}">
          <inertial pos="0 {thigh_p[1] / 2} -0.14" mass="{MASSES['THIGH']}" diaginertia="0.02 0.02 0.004" />
          <joint name="{prefix}_HFE" axis="0 1 0" range="{pitch_range}"
            limited="true" damping="0.1" armature="0.01" frictionloss="0.2" />
          <geom type="capsule" size="0.045" fromto="0 0 0 {v(shank_p)}" />
          <body name="{prefix}_SHANK" pos="{v(shank_p)}">
            <inertial pos="{foot_p[0] / 2} 0 -0.2" mass="{MASSES['SHANK']}" diaginertia="0.008 0.008 0.001" />
            <joint name="{prefix}_KFE" axis="0 1 0" range="{pitch_range}"
              limited="true" damping="0.1" armature="0.01" frictionloss="0.2" />
            <geom type="capsule" size="0.03" fromto="0 0 0 {v(foot_p)}" />
            <geom size="0.031" pos="{v(foot_p)}" type="sphere" />
            <body name="{prefix}_FOOT" pos="{v(foot_p)}" />
          </body>
        </body>
      </body>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="protomotions/data/assets/mjcf/anymal_d_nomesh.xml",
        help="Output MJCF path",
    )
    args = parser.parse_args()

    legs = "\n".join(leg(p) for p in ["LF", "LH", "RF", "RH"])
    motors = "\n".join(
        f'    <motor name="{p}_{j}" joint="{p}_{j}" ctrllimited="true" ctrlrange="-{EFFORT} {EFFORT}" />'
        for p in ["LF", "LH", "RF", "RH"]
        for j in ["HAA", "HFE", "KFE"]
    )

    xml = f"""<mujoco model="anymal_d">
  <compiler angle="radian" />

  <worldbody>
    <body name="base" pos="0 0 0.6">
      <inertial pos="0 0 0" mass="{MASSES['base']}" diaginertia="0.3 0.7 0.8" />
      <joint name="floating_base_joint" type="free" limited="false" actuatorfrclimited="false" />
      <geom size="0.32 0.14 0.09" type="box" />
{legs}
    </body>
  </worldbody>

  <actuator>
{motors}
  </actuator>

</mujoco>
"""
    with open(args.output, "w") as f:
        f.write(xml)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
