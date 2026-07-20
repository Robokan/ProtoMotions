# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""List the T800 USD's physics joints and their drive APIs.

Expect: SUMMARY revolute=25 d6=0 other=0 with_drive=25
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args)

from pxr import Usd  # noqa: E402

st = Usd.Stage.Open(
    "protomotions/data/assets/usd/t800/configuration/t800_flat_physics.usd"
)
rev = d6 = driven = other = 0
for p in st.Traverse():
    t = str(p.GetTypeName())
    if "Joint" not in t or t == "PhysicsFixedJoint":
        continue
    schemas = [str(s) for s in p.GetAppliedSchemas()]
    drives = [s for s in schemas if "Drive" in s]
    if t == "PhysicsRevoluteJoint":
        rev += 1
    elif "D6" in t:
        d6 += 1
    else:
        other += 1
    if drives:
        driven += 1
    if "ANKLE" in str(p.GetName()) or "FOOT" in str(p.GetName()):
        print("ANKLE:", p.GetName(), "|", t, "| drives:", drives, flush=True)
print(f"SUMMARY revolute={rev} d6={d6} other={other} with_drive={driven}", flush=True)

import os  # noqa: E402

os._exit(0)
