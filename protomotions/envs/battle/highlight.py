# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Viewer body recoloring for battle: champion tint + red hit flash.

The SOMA robot binds a single shared material to every body geom, so
recoloring is done by rebinding individual geom prims to purpose-made
materials:
- Champion tint: the ego half of the paired envs (the fighter being viewed
  as "self") gets its body geoms bound to a purple material at setup, so it
  is visually distinct from the opponent.
- Hit flash: on a scoring hit the struck body's geom is bound to red for a
  few frames, then restored to its base (purple for the champion, original
  for the opponent).

Only hit transitions touch the stage (no per-frame geometry), so it stays
cheap. Isaac's RTX pipeline needs an explicit material reassign to refresh a
rebind, so we call omni.kit.material.library as a fallback after SetTargets.

Unlike the previous version this LOGS what it discovers (prim/material
counts, setup success) instead of failing silently, so a non-effect is
diagnosable from the run log.
"""

import logging
from typing import List, Optional

import torch

log = logging.getLogger(__name__)


class BodyHighlighter:
    RED_PATH = "/World/Looks/battle_hit_red"
    PURPLE_PATH = "/World/Looks/battle_champion_purple"

    def __init__(
        self,
        num_envs: int,
        num_matches: int,
        body_names: List[str],
        damage_body_ids: torch.Tensor,
        robot_prim_fmt: str = "/World/envs/env_{env}/Robot/Hips/{body}",
        champion_tint: bool = True,
    ):
        self.num_envs = num_envs
        self.num_matches = num_matches  # ego half = envs [0, num_matches)
        self.body_names = body_names
        self.damage_body_ids = [int(i) for i in damage_body_ids.tolist()]
        self.robot_prim_fmt = robot_prim_fmt
        self.champion_tint = champion_tint

        self._enabled = True
        self._ready = False
        self._stage = None
        self._rel = {}   # (env, body_id) -> material:binding relationship
        self._base = {}  # (env, body_id) -> base material path (purple/orig)
        self._prev_active = torch.zeros(num_envs, len(body_names), dtype=torch.bool)

    # ------------------------------------------------------------------
    def _make_material(self, stage, path, rgb, emissive=(0.0, 0.0, 0.0)):
        from pxr import UsdShade, Sdf

        if stage.GetPrimAtPath(path):
            return
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgb)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(emissive)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    @staticmethod
    def _find_geom_rel(root_prim):
        """First prim in the subtree carrying a valid material:binding."""
        stack = [root_prim]
        while stack:
            prim = stack.pop()
            rel = prim.GetRelationship("material:binding")
            if rel and rel.IsValid() and rel.GetTargets():
                return rel
            stack.extend(prim.GetChildren())
        return None

    def _setup(self):
        import omni.usd
        from pxr import Sdf

        self._stage = omni.usd.get_context().get_stage()
        stage = self._stage
        self._make_material(stage, self.RED_PATH, (1.0, 0.05, 0.05), (0.6, 0.0, 0.0))
        self._make_material(stage, self.PURPLE_PATH, (0.45, 0.1, 0.75), (0.12, 0.0, 0.2))

        found = 0
        for env in range(self.num_envs):
            is_ego = env < self.num_matches
            for body_id in self.damage_body_ids:
                path = self.robot_prim_fmt.format(env=env, body=self.body_names[body_id])
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    continue
                rel = self._find_geom_rel(prim)
                if rel is None:
                    continue
                self._rel[(env, body_id)] = rel
                orig = rel.GetTargets()
                if self.champion_tint and is_ego:
                    base = self.PURPLE_PATH
                    rel.SetTargets([Sdf.Path(self.PURPLE_PATH)])
                    self._reassign(path, self.PURPLE_PATH)
                else:
                    base = str(orig[0]) if orig else None
                self._base[(env, body_id)] = base
                found += 1

        # Also tint the NON-damage body geoms of ego envs purple, so the whole
        # champion is purple (not just its 3 damage bodies).
        if self.champion_tint:
            self._tint_full_champion(stage)

        self._ready = True
        log.info(
            "BodyHighlighter: %d/%d damage-body bindings found (%d envs, %d damage bodies); "
            "champion_tint=%s. Red flash + purple champion %s.",
            found, self.num_envs * len(self.damage_body_ids),
            self.num_envs, len(self.damage_body_ids), self.champion_tint,
            "ACTIVE" if found else "INACTIVE (no bindings found — check prim paths)",
        )

    def _tint_full_champion(self, stage):
        from pxr import Sdf

        n = 0
        for env in range(self.num_matches):  # ego half only
            for body_name in self.body_names:
                path = self.robot_prim_fmt.format(env=env, body=body_name)
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    continue
                rel = self._find_geom_rel(prim)
                if rel is not None:
                    rel.SetTargets([Sdf.Path(self.PURPLE_PATH)])
                    self._reassign(path, self.PURPLE_PATH)
                    n += 1
        log.info("BodyHighlighter: tinted %d champion body geoms purple", n)

    @staticmethod
    def _reassign(prim_path, material_path):
        """RTX/Fabric refresh — SetTargets alone often doesn't repaint."""
        try:
            import omni.kit.material.library as mat_lib

            mat_lib.apply_material_to_prims(material_path, [prim_path])
        except Exception:
            pass

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, flash_timer: torch.Tensor) -> None:
        if not self._enabled:
            return
        try:
            if not self._ready:
                self._setup()
            from pxr import Sdf

            timer = flash_timer.detach().to("cpu")
            for col, body_id in enumerate(self.damage_body_ids):
                for env in range(self.num_envs):
                    on = bool(timer[env, col] > 0)
                    was = bool(self._prev_active[env, body_id])
                    if on == was:
                        continue
                    rel = self._rel.get((env, body_id))
                    if rel is None:
                        continue
                    path = self.robot_prim_fmt.format(
                        env=env, body=self.body_names[body_id]
                    )
                    if on:
                        rel.SetTargets([Sdf.Path(self.RED_PATH)])
                        self._reassign(path, self.RED_PATH)
                    else:
                        base = self._base.get((env, body_id))
                        if base:
                            rel.SetTargets([Sdf.Path(base)])
                            self._reassign(path, base)
                    self._prev_active[env, body_id] = on
        except Exception as exc:  # never disturb the sim
            self._enabled = False
            log.warning("BodyHighlighter disabled after error: %s", exc)


__all__ = ["BodyHighlighter"]
