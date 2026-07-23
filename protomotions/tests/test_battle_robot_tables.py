# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-robot battle tables must resolve against the robot's real skeleton."""

import pytest

from protomotions.envs.battle.hit_state import resolve_body_ids
from protomotions.envs.battle.robot_tables import battle_table_kwargs


def _robot(name):
    from protomotions.robot_configs.factory import robot_config

    try:
        return robot_config(name)
    except Exception as exc:  # asset files may be absent on some checkouts
        pytest.skip(f"robot config {name} unavailable: {exc}")


@pytest.mark.parametrize("name", ["soma23", "t800", "atlas"])
def test_tables_resolve_against_skeleton(name):
    cfg = _robot(name)
    body_names = list(cfg.kinematic_info.body_names)
    table = battle_table_kwargs(cfg, name)

    resolve_body_ids(table["strike_body_names"], body_names)
    resolve_body_ids(table["damage_body_names"], body_names)
    resolve_body_ids(table["key_body_names"], body_names)
    resolve_body_ids([table["head_body_name"]], body_names)
    resolve_body_ids([table["facing_target_body_name"]], body_names)

    # Aligned per-damage-body vectors.
    assert len(table["damage_multipliers"]) == len(table["damage_body_names"])
    assert len(table["stun_region_weights"]) == len(table["damage_body_names"])
    # Obs contract: exactly 5 key bodies (head, hands, feet) league-wide.
    assert len(table["key_body_names"]) == 5
    # Both strike groups populated, and groups cover the strike list.
    groups = table["strike_body_group_names"]
    assert groups["hands"] and groups["legs"]
    assert set(groups["hands"]) | set(groups["legs"]) == set(
        table["strike_body_names"]
    )


def test_soma23_tables_are_pinned():
    """The SOMA tables must never drift — v4/v5 checkpoints depend on them."""
    cfg = _robot("soma23")
    t = battle_table_kwargs(cfg, "soma23")
    assert t["damage_body_names"] == ["Head", "Chest", "Spine2", "Spine1", "Hips"]
    assert t["damage_multipliers"] == [2.0, 1.0, 1.25, 1.25, 0.5]
    assert t["head_body_name"] == "Head"
    assert t["facing_target_body_name"] == "Chest"
    assert t["gaze_forward_axis"] == (0.0, -1.0, 0.0)
    assert len(t["strike_body_names"]) == 12
