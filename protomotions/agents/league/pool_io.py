# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared-pool snapshot I/O hygiene (MULTI_ROBOT_LEAGUE_PLAN Phase 0).

Concurrent league trainers publishing into one directory need:
atomic writes (no torn reads), collision-proof names (per-run ids, not
``len(dir)`` counters), provenance metadata (who trained this, on which
robot/architecture/rules era, with what weight shapes), and loud
validation before a snapshot is hosted in an opponent lane.

Used by both league homes — ``full_model_league.FullModelLeagueMixin``
and the PEFT league ``agent.LeagueDiscretePriorPEFTRLFTAgent``.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch

# The current battle rules era. Bump when damage/stun/win rules change in a
# way that makes old snapshots a different game (plan: "rules-era stamping").
# Era components: KE-based damage, strike_min_speed=1.5, stun-gated KO,
# win-500 terminal reward, kick bonus.
GAME_RULES_VERSION = "ke1.5-stunko-win500-kick"

# Bump when the snapshot payload layout changes incompatibly.
SNAPSHOT_SCHEMA_VERSION = 2


def new_run_id() -> str:
    """Collision-proof short run id: launch time + pid entropy."""
    return f"{time.strftime('%y%m%d%H%M%S')}{os.getpid() % 10000:04d}"


def atomic_save(payload: dict, path: Path) -> None:
    """torch.save via a temp file + os.replace — readers never see a torn file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def state_fingerprint(state_dict: Dict[str, torch.Tensor]) -> str:
    """Architecture fingerprint: hash of the sorted (key, shape) signature.

    Two snapshots are lane-compatible for full-weight hosting iff their
    fingerprints match (same keys, same shapes). Weight VALUES are
    deliberately excluded — this identifies the architecture, not the policy.
    """
    sig = ";".join(
        f"{k}:{tuple(v.shape)}" for k, v in sorted(state_dict.items())
    )
    return hashlib.sha1(sig.encode()).hexdigest()[:12]


def snapshot_run_id(path: Path) -> Optional[str]:
    """Parse the run id out of ``policy_{run_id}_{counter}.ckpt``.

    Legacy names (``policy_{counter}.ckpt``) return None — they predate
    run ids and are treated as own-family by the restoring run.
    """
    parts = path.stem.split("_")
    if len(parts) == 3 and parts[0] == "policy":
        return parts[1]
    return None


def family_key(meta: dict) -> str:
    """Family identity of a snapshot: robot + architecture + weight shapes.

    PFSP gating and per-family quotas group by this, NOT by run id — three
    seeds of the same config are three runs but one family only if you want
    run-level grouping; the plan gates on own-family = own-run, so run_id
    is included when present.
    """
    return "/".join(
        str(meta.get(k, "unknown"))
        for k in ("robot", "architecture", "fingerprint", "run_id")
    )


def load_snapshot_meta(path: Path) -> Optional[dict]:
    """Load a snapshot's metadata (full payload; snapshots are small)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - concurrent writer may be mid-replace
        return None


def build_snapshot_meta(
    *,
    run_id: str,
    robot: str,
    architecture: str,
    fingerprint: str,
    action_dim: Optional[int],
    epoch: int,
    rating: float,
    reason: str,
    extra: Optional[dict] = None,
) -> dict:
    meta = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "run_id": run_id,
        "robot": robot,
        "architecture": architecture,
        "fingerprint": fingerprint,
        "action_dim": action_dim,
        "game_rules_version": GAME_RULES_VERSION,
        "epoch": epoch,
        "rating": rating,
        "reason": reason,
        "time": time.time(),
    }
    if extra:
        meta.update(extra)
    return meta


class SnapshotIncompatible(RuntimeError):
    """A snapshot cannot be hosted by this run's lanes."""


def check_compatible(
    meta: dict,
    *,
    robot: str,
    architecture: str,
    fingerprint: str,
    accept_foreign_rules_era: bool = False,
    path: str = "?",
) -> None:
    """Raise SnapshotIncompatible (loudly) on any provenance mismatch.

    Only validates keys the snapshot actually carries — legacy snapshots
    without provenance pass (they were written by this run's own lineage).
    """
    for key, own in (("robot", robot), ("architecture", architecture),
                     ("fingerprint", fingerprint)):
        theirs = meta.get(key)
        if theirs is not None and theirs != own:
            raise SnapshotIncompatible(
                f"snapshot {path}: {key}={theirs!r} does not match hosting "
                f"lane ({key}={own!r}) — refusing to load silently-garbage "
                f"weights (MULTI_ROBOT_LEAGUE_PLAN Phase 0 validation)"
            )
    era = meta.get("game_rules_version")
    if era is not None and era != GAME_RULES_VERSION and not accept_foreign_rules_era:
        raise SnapshotIncompatible(
            f"snapshot {path}: game_rules_version={era!r} != current "
            f"{GAME_RULES_VERSION!r} (different game; set league."
            f"accept_foreign_rules_era to host anyway)"
        )


__all__ = [
    "GAME_RULES_VERSION",
    "SNAPSHOT_SCHEMA_VERSION",
    "new_run_id",
    "atomic_save",
    "state_fingerprint",
    "snapshot_run_id",
    "family_key",
    "load_snapshot_meta",
    "build_snapshot_meta",
    "SnapshotIncompatible",
    "check_compatible",
]
