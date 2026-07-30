# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-env robot tinting for battle exhibitions.

The robots' visual meshes are USD *instances* (instanceable references), so
material overrides inside a per-env asset copy are silently ignored — the
render comes from the shared prototype. The reliable way to recolor one
fighter is to bind a material at the robot's ROOT prim with
``strongerThanDescendants`` binding strength: ancestor bindings at that
strength take precedence over everything inside the instances.

Used by battle_tournament's ``--opponent-tint`` to make the opponent half of
paired battle envs (envs N/2..N-1) visually distinct. Rendering only — no
physics impact.
"""

from __future__ import annotations

import logging
from typing import Sequence

log = logging.getLogger(__name__)


def tint_env_robots(
    env_indices: Sequence[int],
    color,
    robot_prim_template: str = "/World/envs/env_{i}/Robot",
    material_path: str = "/World/Looks/FighterTint",
) -> int:
    """Bind a solid tint material onto each listed env's robot root.

    Args:
        env_indices: Env indices whose robots get the tint.
        color: RGB triple in [0, 1].
        robot_prim_template: Per-env robot root prim path template.
        material_path: Stage path for the tint material (created if needed).

    Returns:
        Number of robots tinted.
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdShade

    stage = omni.usd.get_context().get_stage()

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*[float(c) for c in color])
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.2)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )

    tinted = 0
    for i in env_indices:
        prim = stage.GetPrimAtPath(robot_prim_template.format(i=i))
        if not prim.IsValid():
            log.warning("tint: no robot prim at env %d", i)
            continue
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        tinted += 1
    log.info("Tinted %d robots %s", tinted, tuple(color))
    return tinted


__all__ = ["tint_env_robots"]
