# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-robot battle body tables.

BattleControlConfig's body-name defaults are SMPL-family (soma23). Any other
robot needs its own strike/damage/key/head tables. This module derives them
from the robot config's semantic mapping (``common_naming_to_robot_body_names``)
plus name-pattern matching over the robot's actual skeleton, with explicit
per-robot overrides where derivation would be ambiguous.

Everything returned here is validated by ``resolve_body_ids`` at
BattleControl construction — unknown names fail fast, so a bad table cannot
train silently.

Usage (experiment file)::

    from protomotions.envs.battle.robot_tables import battle_table_kwargs
    BattleControlConfig(arena_size=..., **battle_table_kwargs(robot_cfg, args.robot_name))
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# Patterns for strike-surface discovery on arbitrary humanoid skeletons.
_HAND_STRIKE_PAT = re.compile(r"(hand|wrist|forearm|fore_arm|elbow|lowerarm)", re.I)
_LEG_STRIKE_PAT = re.compile(r"(foot|toe|ankle|shin|knee|calf|lowerleg|tibia)", re.I)
# Left/right filtering so we never pick up spine/neck links by accident.
_SIDED_PAT = re.compile(r"(^l_|^r_|_l$|_r$|_l_|_r_|left|right)", re.I)


def _semantic(robot_config, key: str) -> List[str]:
    mapping = getattr(robot_config, "common_naming_to_robot_body_names", None) or {}
    val = mapping.get(key, [])
    return list(val) if isinstance(val, (list, tuple)) else [val]


def _body_names(robot_config) -> List[str]:
    return list(robot_config.kinematic_info.body_names)


def _matches(names: List[str], pat: re.Pattern) -> List[str]:
    return [n for n in names if pat.search(n) and _SIDED_PAT.search(n)]


# ---------------------------------------------------------------------------
# Explicit tables. soma23 pins the exact tables the SOMA league has always
# used (must not drift — v4/v5 checkpoints were trained against them).
# ---------------------------------------------------------------------------
_SOMA23 = dict(
    strike_body_names=[
        "LeftArm", "LeftForeArm", "LeftHand",
        "RightArm", "RightForeArm", "RightHand",
        "LeftLeg", "LeftShin", "LeftFoot",
        "RightLeg", "RightShin", "RightFoot",
    ],
    strike_body_group_names={
        "hands": ["LeftArm", "LeftForeArm", "LeftHand",
                  "RightArm", "RightForeArm", "RightHand"],
        "legs": ["LeftLeg", "LeftShin", "LeftFoot",
                 "RightLeg", "RightShin", "RightFoot"],
    },
    damage_body_names=["Head", "Chest", "Spine2", "Spine1", "Hips"],
    damage_multipliers=[2.0, 1.0, 1.25, 1.25, 0.5],
    stun_region_weights=[1.0, 0.1, 0.15, 0.1, 0.05],
    key_body_names=["Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot"],
    head_body_name="Head",
    facing_target_body_name="Chest",
    gaze_forward_axis=(0.0, -1.0, 0.0),  # SOMA (SMPL family) faces -y
)

# T800 (EngineAI): semantic map gives head/torso/hands/feet; strike surfaces
# from LINK_* patterns. Gaze axis +x CALIBRATED 2026-07-31 (corpus walking
# frames measure head-local forward = [0.97, 0.11, -0.21] — +x confirmed).
_T800_EXPLICIT = dict(
    head_body_name="LINK_HEAD_YAW",
    facing_target_body_name="LINK_TORSO_YAW",
    gaze_forward_axis=(1.0, 0.0, 0.0),
)

_ATLAS_EXPLICIT = dict(
    # CALIBRATED 2026-07-31 from corpus standing frames (head local axis
    # aligned with hip facing): the Head frame's face direction is ~+Z with
    # a -Y tilt — NOT +X. The prior assumed (1,0,0) trained overnight
    # fighters that never faced each other (facing reward pointed the
    # robot's SIDE at the opponent).
    gaze_forward_axis=(0.012, -0.319, 0.948),
)

_EXPLICIT: Dict[str, dict] = {
    "soma23": _SOMA23,
    "t800": _T800_EXPLICIT,
    "atlas": _ATLAS_EXPLICIT,
}


def _derive_generic(robot_config) -> dict:
    """Derive tables from semantic mapping + skeleton name patterns."""
    names = _body_names(robot_config)

    hands_sem = _semantic(robot_config, "all_left_hand_bodies") + _semantic(
        robot_config, "all_right_hand_bodies"
    )
    feet_sem = _semantic(robot_config, "all_left_foot_bodies") + _semantic(
        robot_config, "all_right_foot_bodies"
    )
    head = _semantic(robot_config, "head_body_name")
    torso = _semantic(robot_config, "torso_body_name")
    if not (head and torso):
        raise ValueError(
            "Robot config lacks head/torso semantic names; add an explicit "
            "battle table in protomotions/envs/battle/robot_tables.py"
        )

    hand_strikes = sorted(set(hands_sem) | set(_matches(names, _HAND_STRIKE_PAT)))
    leg_strikes = sorted(set(feet_sem) | set(_matches(names, _LEG_STRIKE_PAT)))
    if not hand_strikes or not leg_strikes:
        raise ValueError(
            f"Could not derive strike surfaces (hands={hand_strikes}, "
            f"legs={leg_strikes}); add explicit tables for this robot."
        )

    # Damage: head (2x), torso (1x), root/pelvis (0.5x) when identifiable.
    damage = [head[0], torso[0]]
    mults = [2.0, 1.0]
    stun = [1.0, 0.1]
    root_candidates = [n for n in names if re.search(r"(base|pelvis|hips|waist)", n, re.I)]
    if root_candidates:
        damage.append(root_candidates[0])
        mults.append(0.5)
        stun.append(0.05)

    # Key bodies exposed in opponent observations. ORDER AND COUNT (5) are the
    # league-wide obs contract (obs width = 20 + 6K): head, L hand, R hand,
    # L foot, R foot.
    lh = _semantic(robot_config, "all_left_hand_bodies")
    rh = _semantic(robot_config, "all_right_hand_bodies")
    lf = _semantic(robot_config, "all_left_foot_bodies")
    rf = _semantic(robot_config, "all_right_foot_bodies")
    if not (lh and rh and lf and rf):
        raise ValueError("Robot config lacks hand/foot semantic names for key bodies")
    key_bodies = [head[0], lh[0], rh[0], lf[0], rf[0]]

    return dict(
        strike_body_names=hand_strikes + leg_strikes,
        strike_body_group_names={"hands": hand_strikes, "legs": leg_strikes},
        damage_body_names=damage,
        damage_multipliers=mults,
        stun_region_weights=stun,
        key_body_names=key_bodies,
        head_body_name=head[0],
        facing_target_body_name=torso[0],
        # Kick-attempt bonus feet (BattleControlConfig defaults are SMPL names).
        kick_bonus_left_foot_body=lf[0],
        kick_bonus_right_foot_body=rf[0],
    )


def battle_table_kwargs(robot_config, robot_name: str) -> dict:
    """BattleControlConfig kwargs for this robot's skeleton.

    soma23 returns its historical tables verbatim; other robots derive from
    the semantic mapping + skeleton patterns, overlaid with any explicit
    entries. Validation happens downstream in resolve_body_ids.
    """
    explicit = _EXPLICIT.get(robot_name, {})
    if robot_name == "soma23":
        return dict(_SOMA23)
    table = _derive_generic(robot_config)
    table.update(explicit)
    return table


__all__ = ["battle_table_kwargs"]
