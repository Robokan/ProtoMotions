# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-character battle task: paired envs, hit scoring, and match accounting.

See SOMA_GPC_COMBAT_PLAN.md (Phase 5) for the design and IsaacLabASE's battle
system for the mechanics this ports.

Imports are lazy: ``context_views`` imports ``battle.context`` (for the
``EnvContext.battle`` view) while ``battle.env`` imports ``base_env.env``,
which imports ``context_views`` — eager package imports here would close that
cycle.
"""

_EXPORTS = {
    "BattleContext": "protomotions.envs.battle.context",
    "BattleControl": "protomotions.envs.battle.control",
    "BattleControlConfig": "protomotions.envs.battle.control",
    "BattleEnv": "protomotions.envs.battle.env",
    "BattleHitState": "protomotions.envs.battle.hit_state",
    "HitStateConfig": "protomotions.envs.battle.hit_state",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib

        module = importlib.import_module(_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
