# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the combat motion-weighting tool (SOMA_GPC_COMBAT_PLAN Phase 3)."""

import importlib.util
from pathlib import Path

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "data/scripts/weight_combat_motions.py"
)
_spec = importlib.util.spec_from_file_location("weight_combat_motions", _MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
apply_combat_weights = _module.apply_combat_weights


def _lib(num_seed=8, num_combat=2, seed_weight=1.0, combat_weight=1.0):
    files = [f"motions/seed/clip_{i}.motion" for i in range(num_seed)]
    files += [f"motions/combat/strike_{i}.motion" for i in range(num_combat)]
    weights = torch.tensor(
        [seed_weight] * num_seed + [combat_weight] * num_combat
    )
    return {"motion_files": tuple(files), "motion_weights": weights}


def test_combat_fraction_sets_exact_sampling_share():
    data = _lib(num_seed=98, num_combat=2)
    stats = apply_combat_weights(data, "/combat/", combat_fraction=0.5)
    assert stats["combat_sampling_share"] == pytest.approx(0.5, abs=1e-5)
    # Sampling shares survive multinomial normalization by construction
    w = data["motion_weights"]
    assert w.sum() == pytest.approx(100.0, rel=1e-5)
    assert (w > 0).all()


def test_combat_fraction_is_independent_of_clip_counts():
    """The point of the tool: 142K seed clips must not drown 200 combat clips."""
    small = _lib(num_seed=10, num_combat=5)
    large = _lib(num_seed=10000, num_combat=5)
    s1 = apply_combat_weights(small, "/combat/", combat_fraction=0.4)
    s2 = apply_combat_weights(large, "/combat/", combat_fraction=0.4)
    assert s1["combat_sampling_share"] == pytest.approx(0.4, abs=1e-5)
    assert s2["combat_sampling_share"] == pytest.approx(0.4, abs=1e-5)


def test_boost_multiplies_combat_weights():
    data = _lib(num_seed=2, num_combat=2)
    apply_combat_weights(data, "/combat/", boost=3.0)
    w = data["motion_weights"]
    # combat clips end up 3x the seed clips after normalization
    assert w[2] / w[0] == pytest.approx(3.0, rel=1e-5)


def test_respects_existing_per_clip_weights():
    data = _lib(num_seed=2, num_combat=2, combat_weight=2.0)
    data["motion_weights"][2] = 3.0  # one combat clip weighted higher
    apply_combat_weights(data, "/combat/", combat_fraction=0.5)
    w = data["motion_weights"]
    # Relative weighting inside the combat group is preserved (3 : 2)
    assert w[2] / w[3] == pytest.approx(1.5, rel=1e-5)


def test_rejects_pattern_matching_nothing_or_everything():
    with pytest.raises(ValueError, match="No clips match"):
        apply_combat_weights(_lib(), "/nonexistent/", combat_fraction=0.5)
    with pytest.raises(ValueError, match="Every clip matches"):
        apply_combat_weights(_lib(), "motions/", combat_fraction=0.5)


def test_rejects_ambiguous_or_missing_mode():
    with pytest.raises(ValueError, match="exactly one"):
        apply_combat_weights(_lib(), "/combat/", combat_fraction=0.5, boost=2.0)
    with pytest.raises(ValueError, match="exactly one"):
        apply_combat_weights(_lib(), "/combat/")
    with pytest.raises(ValueError, match="must be in"):
        apply_combat_weights(_lib(), "/combat/", combat_fraction=1.5)
