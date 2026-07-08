# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-body hit-energy integrator for battle tasks.

Port of IsaacLabASE's ``BodyHitState`` (amp_getup_env.py) with one structural
change: attribution. IsaacLabASE reads each defender's *filtered* contact
sensors (two articulations in one env prim), so any sensed force is known to
come from the opponent. In ProtoMotions' paired-env layout the sensors report
*net* world contact forces (ground included), so a force on a damage body only
counts as a hit when an opponent strike body is within ``proximity_radius`` of
it. The tuned FSM constants (hysteresis thresholds, cooldown, closing-velocity
gating, log-normalized energy) are carried over unchanged.

All tensors are laid out per-env over the full doubled batch (2N envs); the
caller supplies the opponent permutation.
"""

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor


@dataclass
class HitStateConfig:
    """Constants for the hit-energy FSM (defaults match IsaacLabASE)."""

    force_on: float = 20.0  # N, bout-start threshold
    force_off: float = 10.0  # N, bout-end threshold (hysteresis)
    cooldown_time: float = 0.15  # s per-body cooldown between bouts
    v_gate: float = 0.001  # m/s minimum closing speed (live ASE value)
    energy_gain: float = 50.0  # dE = F * v_rel * dt * energy_gain
    e0_percentile: float = 0.95  # percentile for the global energy scale
    e0_ema: float = 0.99  # EMA coefficient for the global energy scale
    proximity_radius: float = 0.35  # m, strike body must be this close to count
    warmup_steps: int = 10  # zero hit rewards for the first N steps of an episode


class BattleHitState:
    """Vectorized per-(env, damage-body) hit-energy state machine.

    Produces a per-step, per-env "hit energy taken" scalar that is
    log-normalized against a running global scale, exactly like the reference
    implementation, so reward magnitudes stay stable as fights evolve.
    """

    def __init__(
        self,
        num_envs: int,
        damage_body_ids: Tensor,
        strike_body_ids: Tensor,
        damage_multipliers: Tensor,
        config: HitStateConfig,
        dt: float,
        device: torch.device,
    ):
        """
        Args:
            num_envs: Total env count (2N, both sides of every match).
            damage_body_ids: Indices of bodies that can take damage [D].
            strike_body_ids: Indices of bodies that can deal damage [S].
            damage_multipliers: Per-damage-body region multiplier [D]
                (e.g. head 2.0, torso 1.0, pelvis 0.5).
            config: FSM constants.
            dt: Control step in seconds.
            device: Torch device.
        """
        self.config = config
        self.dt = dt
        self.device = device
        self.damage_body_ids = damage_body_ids
        self.strike_body_ids = strike_body_ids
        self.damage_multipliers = damage_multipliers.to(device)

        num_damage = len(damage_body_ids)
        self._active = torch.zeros(num_envs, num_damage, dtype=torch.bool, device=device)
        self._e_accum = torch.zeros(num_envs, num_damage, device=device)
        self._e_prev = torch.zeros(num_envs, num_damage, device=device)
        self._cooldown = torch.zeros(num_envs, num_damage, device=device)
        self._e0 = 1.0  # global log-normalization scale (python float EMA)
        self._steps_cool = max(1, int(round(config.cooldown_time / dt)))

    def reset(self, env_ids: Tensor) -> None:
        self._active[env_ids] = False
        self._e_accum[env_ids] = 0.0
        self._e_prev[env_ids] = 0.0
        self._cooldown[env_ids] = 0.0

    @torch.no_grad()
    def step(
        self,
        contact_forces: Tensor,
        body_pos: Tensor,
        body_vel: Tensor,
        opp_body_pos: Tensor,
        opp_body_vel: Tensor,
        progress: Tensor,
    ) -> Tensor:
        """Advance the FSM one control step and return per-env hit energy taken.

        Args:
            contact_forces: Net contact forces on ALL bodies [2N, B, 3].
            body_pos: Own body positions [2N, B, 3].
            body_vel: Own body velocities [2N, B, 3].
            opp_body_pos: Opponent body positions, partner-permuted [2N, B, 3].
            opp_body_vel: Opponent body velocities, partner-permuted [2N, B, 3].
            progress: Episode progress counters [2N].

        Returns:
            Per-env log-normalized hit energy taken this step [2N]. The
            energy *dealt* by env i is the energy taken by its partner; the
            caller permutes.
        """
        cfg = self.config
        d_ids = self.damage_body_ids
        s_ids = self.strike_body_ids

        # Force magnitude and unit normal on each damage body
        force = contact_forces[:, d_ids, :]  # [2N, D, 3]
        f_mag = torch.norm(force, dim=-1)  # [2N, D]
        n_hat = force / f_mag.clamp_min(1e-8).unsqueeze(-1)

        d_pos = body_pos[:, d_ids, :]  # [2N, D, 3]
        d_vel = body_vel[:, d_ids, :]
        s_pos = opp_body_pos[:, s_ids, :]  # [2N, S, 3]
        s_vel = opp_body_vel[:, s_ids, :]

        # Attribution: nearest opponent strike body per damage body
        # dist[e, d, s] = ||d_pos[e,d] - s_pos[e,s]||
        dist = torch.cdist(d_pos, s_pos)  # [2N, D, S]
        min_dist, nearest_s = dist.min(dim=-1)  # [2N, D]
        include = min_dist <= cfg.proximity_radius  # [2N, D]

        # Closing speed along the contact normal, using the attributed striker
        striker_vel = torch.gather(
            s_vel, 1, nearest_s.unsqueeze(-1).expand(-1, -1, 3)
        )  # [2N, D, 3]
        v_rel = ((striker_vel - d_vel) * n_hat).sum(dim=-1).clamp_min(0.0)  # [2N, D]
        v_ok = v_rel >= cfg.v_gate

        d_energy = f_mag * v_rel * self.dt * cfg.energy_gain * (include & v_ok)

        # Bout FSM with hysteresis (force_on/force_off) + cooldown
        can_start = self._cooldown <= 0.5
        start = (f_mag > cfg.force_on) & ~self._active & can_start & include
        end = ((f_mag < cfg.force_off) | ~include) & self._active
        self._active = (self._active & ~end & (f_mag >= cfg.force_off)) | start

        self._e_accum = self._e_accum + d_energy * self._active

        # Global scale: EMA of the batch percentile of accumulated energy
        flat = self._e_accum.flatten()
        if flat.numel() > 0:
            e_cap = torch.quantile(flat, self.config.e0_percentile).item()
            if e_cap > 0.0:
                self._e0 = cfg.e0_ema * self._e0 + (1.0 - cfg.e0_ema) * e_cap

        # Per-step reward per body: positive delta of log1p-normalized energy
        phi_now = torch.log1p(self._e_accum / max(self._e0, 1e-6))
        phi_prev = torch.log1p(self._e_prev / max(self._e0, 1e-6))
        r_per_body = (phi_now - phi_prev).clamp_min(0.0) * include
        self._e_prev = self._e_accum.clone()

        # Bout end: arm cooldown, zero accumulators
        self._cooldown = torch.where(
            end, torch.full_like(self._cooldown, float(self._steps_cool)), self._cooldown
        )
        self._e_accum = torch.where(end, torch.zeros_like(self._e_accum), self._e_accum)
        self._e_prev = torch.where(end, torch.zeros_like(self._e_prev), self._e_prev)
        self._cooldown = (self._cooldown - 1.0).clamp_min(0.0)

        # Region multipliers, warm-up gating, reduce over bodies
        r_taken = (r_per_body * self.damage_multipliers.unsqueeze(0)).sum(dim=-1)
        r_taken = torch.where(
            progress < self.config.warmup_steps, torch.zeros_like(r_taken), r_taken
        )
        return r_taken


def resolve_body_ids(body_names: List[str], all_body_names: List[str]) -> Tensor:
    """Map body names to indices, raising on unknown names."""
    ids = []
    for name in body_names:
        if name not in all_body_names:
            raise ValueError(
                f"Battle body '{name}' not found in robot bodies: {all_body_names}"
            )
        ids.append(all_body_names.index(name))
    return torch.tensor(ids, dtype=torch.long)


__all__ = ["HitStateConfig", "BattleHitState", "resolve_body_ids"]
