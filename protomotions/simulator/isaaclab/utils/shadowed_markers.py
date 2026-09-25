# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Sphere markers that cast shadows: one real prim per env, no PointInstancer.

Isaac Lab's ``VisualizationMarkers`` draw through a ``UsdGeom.PointInstancer``.
Under RTX, instancer instances cast NO shadows -- measured on this box with
both an analytic ``SphereCfg`` and a ``MeshSphereCfg`` prototype, while a
standalone sphere prim of the same size casts a normal one. For a marker
that is meant to read as an object in the world (the chase ball), that is the
difference between a ball on the ground and a sticker floating over it.

This class has the same ``visualize`` / ``set_visibility`` surface the
simulator uses, but spawns ``num_envs`` individual mesh spheres and moves
them by writing their xform ops. Cost is a prim per env, so it is opt-in
(``VisualizationMarkerConfig.cast_shadows``) and meant for viewer / data-gen
env counts, not thousands of training envs.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

import isaaclab.sim as sim_utils


class ShadowedSphereMarkers:
    def __init__(
        self,
        prim_path: str,
        num_envs: int,
        color: Sequence[float],
        radius: float = 1.0,
    ):
        from pxr import Gf, Usd, UsdGeom  # noqa: PLC0415

        self._Gf = Gf
        self.prim_path = sim_utils.get_next_free_prim_path(prim_path)
        self.stage = sim_utils.get_current_stage()
        self.num_envs = num_envs
        UsdGeom.Xform.Define(self.stage, self.prim_path)
        cfg = sim_utils.MeshSphereCfg(
            radius=radius,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(float(color[0]), float(color[1]), float(color[2]))
            ),
        )
        self._translate = []
        self._scale = []
        for i in range(num_envs):
            path = f"{self.prim_path}/inst_{i}"
            cfg.func(path, cfg, translation=(0.0, 0.0, -10.0))
            prim = self.stage.GetPrimAtPath(path)
            xf = UsdGeom.Xformable(prim)
            ops = {op.GetOpType(): op for op in xf.GetOrderedXformOps()}
            t = ops.get(UsdGeom.XformOp.TypeTranslate) or xf.AddTranslateOp()
            s = ops.get(UsdGeom.XformOp.TypeScale) or xf.AddScaleOp()
            self._translate.append(t)
            self._scale.append(s)
        self._parent = UsdGeom.Imageable(self.stage.GetPrimAtPath(self.prim_path))

    def visualize(
        self,
        translations: Optional[torch.Tensor] = None,
        orientations: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        marker_indices=None,
    ) -> None:
        """Move the spheres. Orientation is ignored (spheres)."""
        Gf = self._Gf
        # Match each op's authored value type (spawners author double3 for
        # translate; scale may be float3 or double3) -- Set() is type-strict.
        if translations is not None:
            pos = translations.detach().reshape(-1, 3).cpu().tolist()
            for op, p in zip(self._translate, pos):
                cur = op.Get()
                vec = type(cur) if cur is not None else Gf.Vec3d
                op.Set(vec(p[0], p[1], p[2]))
        if scales is not None:
            sc = scales.detach().reshape(-1, 3).cpu().tolist()
            for op, s in zip(self._scale, sc):
                cur = op.Get()
                vec = type(cur) if cur is not None else Gf.Vec3f
                op.Set(vec(s[0], s[1], s[2]))

    def set_visibility(self, visible: bool) -> None:
        if visible:
            self._parent.MakeVisible()
        else:
            self._parent.MakeInvisible()
