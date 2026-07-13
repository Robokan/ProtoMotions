# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Efficient hit visualization: recolor struck body prims red.

Ports IsaacLabASE's approach — rebind the USD ``material:binding`` on each
body's visual prim between a shared red material and its original, and touch
the stage ONLY on hit transitions (a body starting or ending its flash).
This adds zero per-frame geometry, unlike spawning marker spheres (which
re-write transforms every frame and halved the viewer frame rate).

Viewer-only and fully exception-safe: any USD failure disables the
highlighter for the rest of the session rather than disturbing the sim.
"""

from typing import List

import torch


class BodyHighlighter:
    """Rebinds body-prim materials to red on hit, restores on expiry."""

    RED_PATH = "/World/Looks/battle_hit_red"

    def __init__(
        self,
        num_envs: int,
        body_names: List[str],
        damage_body_ids: torch.Tensor,
        robot_prim_fmt: str = "/World/envs/env_{env}/Robot/Hips/{body}",
    ):
        self.num_envs = num_envs
        self.body_names = body_names
        self.damage_body_ids = [int(i) for i in damage_body_ids.tolist()]
        self.robot_prim_fmt = robot_prim_fmt

        self._enabled = True
        self._ready = False
        self._stage = None
        # Per (env, body_id): the visual prim's material:binding relationship
        # and its original target, discovered lazily on first use.
        self._rel = {}
        self._orig = {}
        # Last frame's active state, to act only on transitions.
        self._prev_active = torch.zeros(
            num_envs, len(body_names), dtype=torch.bool
        )

    # ------------------------------------------------------------------
    def _setup(self) -> None:
        """Discover material bindings and create the shared red material."""
        import omni.usd
        from pxr import UsdShade, Sdf

        self._stage = omni.usd.get_context().get_stage()
        stage = self._stage

        # Shared red UsdPreviewSurface
        if not stage.GetPrimAtPath(self.RED_PATH):
            mat = UsdShade.Material.Define(stage, self.RED_PATH)
            shader = UsdShade.Shader.Define(stage, self.RED_PATH + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                (1.0, 0.05, 0.05)
            )
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                (0.6, 0.0, 0.0)
            )
            mat.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )

        # For each (env, damage body) find the deepest visual prim carrying a
        # material:binding relationship, and record its original target.
        for env in range(self.num_envs):
            for body_id in self.damage_body_ids:
                body_prim_path = self.robot_prim_fmt.format(
                    env=env, body=self.body_names[body_id]
                )
                body_prim = stage.GetPrimAtPath(body_prim_path)
                if not body_prim or not body_prim.IsValid():
                    continue
                rel = self._find_material_rel(body_prim)
                if rel is None:
                    continue
                targets = rel.GetTargets()
                self._rel[(env, body_id)] = rel
                self._orig[(env, body_id)] = list(targets) if targets else []
        self._ready = True

    @staticmethod
    def _find_material_rel(root_prim):
        """First prim in the subtree with a valid material:binding rel."""
        stack = [root_prim]
        while stack:
            prim = stack.pop()
            rel = prim.GetRelationship("material:binding")
            if rel and rel.IsValid() and rel.GetTargets():
                return rel
            stack.extend(prim.GetChildren())
        return None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, flash_timer: torch.Tensor) -> None:
        """Recolor on transitions. ``flash_timer`` is [num_envs, num_damage]
        (>0 while a damage body is flashing), aligned with damage_body_ids."""
        if not self._enabled:
            return
        try:
            if not self._ready:
                self._setup()
            active_now = self._prev_active.clone()
            timer_cpu = flash_timer.detach().to("cpu")
            from pxr import Sdf

            for col, body_id in enumerate(self.damage_body_ids):
                for env in range(self.num_envs):
                    on = bool(timer_cpu[env, col] > 0)
                    was = bool(self._prev_active[env, body_id])
                    if on == was:
                        continue  # no transition -> no USD write
                    rel = self._rel.get((env, body_id))
                    if rel is None:
                        continue
                    if on:
                        rel.SetTargets([Sdf.Path(self.RED_PATH)])
                    else:
                        rel.SetTargets(self._orig.get((env, body_id), []))
                    active_now[env, body_id] = on
            self._prev_active = active_now
        except Exception:
            # Any USD/omni failure: disable for the session, never disrupt sim
            self._enabled = False


__all__ = ["BodyHighlighter"]
