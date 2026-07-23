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
    # HEALTH-damage model (independent of the log-normalized REWARD stream):
    # kinetic energy of the striking limb, KE = 0.5 * m_limb * v_impact^2,
    # deposited ONCE per contact event at its onset (the FSM rising edge).
    # Contact-solver forces are impulse artifacts and play no part; mass and
    # velocity come from ground-truth sim state. A contact only qualifies if
    # the striking limb's closing speed at onset is >= strike_min_speed — a
    # push, lean, grind, or standing touch arrives well under 1 m/s and scores
    # exactly zero, while a committed swing arrives at several m/s. Lingering
    # contact never re-scores: one hit, one deposit.
    strike_min_speed: float = 2.5  # m/s, impact-speed gate (calibrated on data)
    # Legacy field from the abandoned force-energy damage model; unused.
    min_hit_energy: float = 50.0
    # Reference energy (J) for the KE-mode REWARD: r = log1p(KE/ref) per
    # contact event, continuous and UNGATED — a tap still earns a small
    # positive guide; only health/wins apply strike_min_speed. ~70 J = a
    # solid strike (hand@11 m/s or shin@7 m/s).
    ke_reward_ref: float = 70.0
    # Flat bonus added ONCE per env on any contact-onset event (KE-reward
    # mode only), on top of log1p(KE/ref). Makes "land a touch" compete with
    # dense facing while KE still ranks hardness. 0 = off (legacy).
    hit_flat: float = 0.0


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
        strike_body_groups: Tensor = None,
        num_strike_groups: int = 0,
        strike_multipliers: Tensor = None,
        strike_body_masses: Tensor = None,
        reward_from_event_ke: bool = False,
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
            strike_body_groups: Group id per strike body [S] (e.g. hands=0,
                legs=1) for per-group hit attribution; None disables.
            num_strike_groups: Number of distinct strike groups.
            strike_multipliers: Per-strike-body multiplier [S] applied to RAW
                strike energy before log-normalization (e.g. legs 2.0, hands
                1.0). None = uniform 1.0.
        """
        self.config = config
        self.dt = dt
        self.device = device
        self.damage_body_ids = damage_body_ids
        self.strike_body_ids = strike_body_ids
        self.damage_multipliers = damage_multipliers.to(device)
        self.strike_body_groups = (
            strike_body_groups.to(device) if strike_body_groups is not None else None
        )
        self.num_strike_groups = num_strike_groups
        self.strike_multipliers = (
            strike_multipliers.to(device) if strike_multipliers is not None else None
        )
        # Per-strike-body masses [S] (kg) for the kinetic-energy damage model.
        # None -> unit masses (pure speed^2); the caller may set this lazily
        # once the simulator exposes real masses (see set_strike_body_masses).
        self.strike_body_masses = (
            strike_body_masses.to(device) if strike_body_masses is not None else None
        )
        # True: reward = log1p(per-event KE / ke_reward_ref), continuous and
        # ungated, replacing the accumulated-F*v log-delta stream.
        self.reward_from_event_ke = reward_from_event_ke

        num_damage = len(damage_body_ids)
        self._active = torch.zeros(num_envs, num_damage, dtype=torch.bool, device=device)
        self._e_accum = torch.zeros(num_envs, num_damage, device=device)
        self._e_prev = torch.zeros(num_envs, num_damage, device=device)
        self._cooldown = torch.zeros(num_envs, num_damage, device=device)
        self._e0 = 1.0  # global log-normalization scale (python float EMA)
        self._steps_cool = max(1, int(round(config.cooldown_time / dt)))
        # Diagnostics for the last step's contact-onset events (pre-speed-gate),
        # used by the calibration probe: impact speed, kinetic energy, and
        # attributed striker index per (env, damage body); zero where no event
        # started this step.
        self.last_event_speed = torch.zeros(num_envs, num_damage, device=device)
        self.last_event_ke = torch.zeros(num_envs, num_damage, device=device)
        self.last_event_striker = torch.zeros(
            num_envs, num_damage, dtype=torch.long, device=device
        )

    def set_strike_body_masses(self, masses: Tensor) -> None:
        """Set per-strike-body masses [S] (kg) for KE damage (lazy wiring)."""
        self.strike_body_masses = masses.to(self.device)

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
            Tuple of (taken, taken_by_group, taken_per_body):
            - taken: per-env log-normalized hit energy taken this step [2N].
              The energy *dealt* by env i is the energy taken by its partner;
              the caller permutes.
            - taken_by_group: the same energy split by the attributed
              striker's group [2N, num_strike_groups] (zeros-shaped [2N, 0]
              when groups are disabled).
            - taken_per_body: region-weighted energy per damage body [2N, D]
              (drives the per-part hit-flash visualization).
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

        # Per-limb raw-energy weighting (e.g. legs > hands), attributed via the
        # nearest striker. Applied HERE — before accumulation and the log-
        # normalization below — so a kick's boosted energy isn't compressed
        # away by the global log scale (see BattleControlConfig
        # .strike_group_multipliers).
        if self.strike_multipliers is not None:
            d_energy = d_energy * self.strike_multipliers[nearest_s]

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

        warmup = progress < self.config.warmup_steps

        # --- Kinetic-energy hit events -----------------------------------
        # One deposit per contact event, at its onset (`start` rising edge):
        # KE = 0.5 * m_striker * v_impact^2, from ground-truth limb mass and
        # closing speed — contact-solver force magnitudes play no part (they
        # are impulse artifacts whose per-step peaks are discretization
        # noise). A sustained push/lean/grind arrives at ~0 m/s and scores
        # ~nothing; lingering contact after a hit never re-scores.
        if self.strike_body_masses is not None:
            striker_mass = self.strike_body_masses[nearest_s]  # [2N, D]
        else:
            striker_mass = torch.ones_like(v_rel)
        ke = 0.5 * striker_mass * v_rel * v_rel  # [2N, D] joules
        event = start  # rising edge already requires force_on & include
        # Diagnostics (pre-speed-gate) for calibration probes.
        self.last_event_speed = v_rel * event
        self.last_event_ke = ke * event
        self.last_event_striker = nearest_s
        # HEALTH damage: speed-GATED KE (taps deal zero; commitment wins).
        ke_event = ke * (event & (v_rel >= cfg.strike_min_speed))
        ke_event = torch.where(
            warmup.unsqueeze(-1), torch.zeros_like(ke_event), ke_event
        )

        # REWARD: in KE mode, a continuous UNGATED function of the same
        # per-event energy — log1p(KE/ref) — so even a light tap earns a
        # small positive guide and the signal grows monotonically with how
        # hard the hit lands. Optional hit_flat adds a once-per-onset bonus
        # so sparse contacts can compete with dense facing. The speed gate
        # applies only to health/wins: taps teach, but they never score HP.
        if self.reward_from_event_ke:
            r_per_body = torch.log1p((ke * event) / max(cfg.ke_reward_ref, 1e-6))

        # Region multipliers, warm-up gating, reduce over bodies
        r_weighted = r_per_body * self.damage_multipliers.unsqueeze(0)  # [2N, D]
        r_taken = r_weighted.sum(dim=-1)
        if self.reward_from_event_ke:
            hit_flat = float(getattr(cfg, "hit_flat", 0.0) or 0.0)
            if hit_flat > 0.0:
                # One flat deposit per env per step that has any onset — not
                # per damage-body, so multi-region contact doesn't stack.
                onset = event.any(dim=-1).to(dtype=r_taken.dtype)
                r_taken = r_taken + hit_flat * onset
        r_taken = torch.where(warmup, torch.zeros_like(r_taken), r_taken)

        # Split by the attributed striker's group (hands vs legs for
        # kickboxing diversity accounting)
        if self.strike_body_groups is not None and self.num_strike_groups > 0:
            groups = self.strike_body_groups[nearest_s]  # [2N, D]
            taken_by_group = torch.zeros(
                r_weighted.shape[0], self.num_strike_groups, device=self.device
            )
            taken_by_group.scatter_add_(1, groups, r_weighted)
            taken_by_group = torch.where(
                warmup.unsqueeze(-1), torch.zeros_like(taken_by_group), taken_by_group
            )
        else:
            taken_by_group = torch.zeros(r_weighted.shape[0], 0, device=self.device)

        taken_per_body = torch.where(
            warmup.unsqueeze(-1), torch.zeros_like(r_weighted), r_weighted
        )

        return r_taken, taken_by_group, taken_per_body, ke_event


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
