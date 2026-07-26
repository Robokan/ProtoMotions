# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared overlay attachment for inference/tournament entrypoints.

Skins each env's fighter with a rigged character (UsdSkel), adds arena
lighting (dome + fill + per-arena ring floods), optionally hides the robot
capsules, and wraps ``env.simulator.step`` to sync the skins from live sim
state (common xyzw layout). Works headless too — the characters are stage
content, so recordings include them.
"""
from pathlib import Path

import numpy as np

import logging

log = logging.getLogger(__name__)


def attach_overlays(env, simulator_config, robot_config, characters,
                    skeleton="ue", fists=True, hide_robot=True,
                    ambient=50.0):
    try:
        import omni.usd
        from pxr import UsdLux, Gf, UsdGeom
        from protomotions.simulator.isaaclab.overlay import SkinnedOverlay
        from protomotions.simulator.isaaclab.overlay_map import (
            SOMA23_TO_CC, SOMA23_REST_REL, SOMA23_TPOSE_POS, SOMA23_PARENT,
            SOMA23_TO_UE, UE_REST_REL, SOMA23_PARENT_UE,
        )

        characters = [str(Path(p).resolve()) for p in characters]
        if skeleton == "ue":
            _map, _rel, _par = SOMA23_TO_UE, UE_REST_REL, SOMA23_PARENT_UE
        else:
            _map, _rel, _par = SOMA23_TO_CC, SOMA23_REST_REL, SOMA23_PARENT
        bn = list(robot_config.kinematic_info.body_names)
        stage = omni.usd.get_context().get_stage()
        n = simulator_config.num_envs
        overlays = []
        for i in range(n):
            overlays.append(SkinnedOverlay(
                stage=stage,
                character_usd=characters[i % len(characters)],
                prim_root=f"/World/overlay{i}",
                body_names=bn,
                body_rest_rot_wxyz=np.zeros((len(bn), 4)),
                joint_map=_map,
                root_only=True,
                drive_bodies=[b for b in _map if b != "Hips"],
                rest_rel=_rel,
                tpose_pos=SOMA23_TPOSE_POS,
                body_parents=_par,
                fists=fists,
            ))
            if hide_robot:
                overlays[-1].set_capsules_visible(
                    f"/World/envs/env_{i}/Robot", False)
        if ambient > 0:
            dome = UsdLux.DomeLight.Define(stage, "/World/overlayDomeLight")
            dome.GetIntensityAttr().Set(ambient)
            fill = UsdLux.DistantLight.Define(stage, "/World/overlayFill")
            fill.GetIntensityAttr().Set(0.4 * ambient)
            fill.GetAngleAttr().Set(5.0)
            UsdGeom.Xformable(fill.GetPrim()).AddRotateXYZOp().Set(
                Gf.Vec3f(-130.0, 0.0, 0.0))

        ring_placed = [False]

        def _place_ring(st):
            bc = getattr(env, "battle_control", None)
            z = float(st.rigid_body_pos[:, 0, 2].mean())
            if bc is not None and hasattr(bc, "arena_centers"):
                cs = bc.arena_centers.unique(dim=0).cpu().numpy()
                centers = [(float(c[0]), float(c[1]), z) for c in cs]
            else:
                centers = [tuple(map(float, r)) for r in
                           st.rigid_body_pos[:, 0, :].cpu().numpy()]
            for ci, c in enumerate(centers):
                for k in range(4):
                    sl = UsdLux.SphereLight.Define(
                        stage, f"/World/overlayRing{ci}L{k}")
                    sl.GetRadiusAttr().Set(1.0)
                    sl.GetIntensityAttr().Set(6000.0 * ambient)
                    sl.GetNormalizeAttr().Set(True)
                    a = 2.0 * np.pi * k / 4 + 0.7854
                    UsdGeom.Xformable(sl.GetPrim()).AddTranslateOp().Set(
                        Gf.Vec3d(c[0] + 9.0 * np.cos(a),
                                 c[1] + 9.0 * np.sin(a), c[2] + 5.0))
            print(f"[overlay] {len(centers)} arena ring(s) lit", flush=True)

        orig_step = env.simulator.step

        def step_with_overlays(*a, **kw):
            out = orig_step(*a, **kw)
            st = env.simulator.get_bodies_state()
            if not ring_placed[0] and ambient > 0:
                _place_ring(st)
                ring_placed[0] = True
            for i, ov in enumerate(overlays):
                try:
                    ov.sync(st.rigid_body_pos[i], st.rigid_body_rot[i])
                except Exception:
                    pass
            return out

        env.simulator.step = step_with_overlays
        log.info("overlays attached to %d envs", n)
    except Exception:
        import traceback
        log.error("overlay attach failed — continuing without skins:")
        traceback.print_exc()
