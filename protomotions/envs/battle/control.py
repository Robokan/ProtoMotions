# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Battle control component: fight state for two-character matches.

Owns everything stateful about a fight — health, per-body hit integration,
knockdown timers, idle/stalling accounting, round timing, and win/lose/draw
determination — and exposes it to observation/reward/termination kernels via
``EnvContext.battle``.

Pairing: with ``2N`` envs, env ``i`` fights env ``(i + N) % 2N``. Match
``m`` (``m < N``) is the pair ``(m, m + N)``.

Match-end rules (per the SOMA GPC combat plan; constants from IsaacLabASE):
- Knockout: a fighter stays "down" (root below ``knockdown_height``) beyond
  ``knockdown_grace_seconds`` — the grace window is what makes get-up tokens
  tactically valuable — or its health reaches zero. The downed fighter loses.
- Out of bounds: leaving the arena loses immediately.
- Timeout: a points decision on remaining-health difference; a draw only when
  healths are within ``points_decision_eps``.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.battle.context import BattleContext
from protomotions.utils import rotations
from protomotions.envs.battle.hit_state import (
    BattleHitState,
    HitStateConfig,
    resolve_body_ids,
)
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig

log = logging.getLogger(__name__)
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class BattleControlConfig(ControlComponentConfig):
    """Configuration for the battle control component (defaults: soma23)."""

    _target_: str = "protomotions.envs.battle.control.BattleControl"

    # Body sets
    # Strike surfaces span the WHOLE limb (minus the shoulder girdle and toes):
    # arms = upper arm + forearm/elbow + hand; legs = thigh/knee + shin + foot.
    # This lets elbow, forearm, knee, thigh and shin strikes register, not just
    # hands and feet. The hit integrator's closing-velocity gate means a limb
    # resting against the opponent (clinch/block) scores ~nothing; only a limb
    # driven into them counts.
    strike_body_names: List[str] = field(
        default_factory=lambda: [
            "LeftArm", "LeftForeArm", "LeftHand",
            "RightArm", "RightForeArm", "RightHand",
            "LeftLeg", "LeftShin", "LeftFoot",
            "RightLeg", "RightShin", "RightFoot",
        ]
    )
    # Two strike groups for kickboxing diversity accounting (dealt hit energy is
    # tracked per group so the reward pays for the under-used group). Labels are
    # kept as "hands"/"legs" because telemetry and monitoring key off them, but
    # each now spans the whole upper / lower limb respectively.
    strike_body_group_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "hands": ["LeftArm", "LeftForeArm", "LeftHand",
                      "RightArm", "RightForeArm", "RightHand"],
            "legs": ["LeftLeg", "LeftShin", "LeftFoot",
                     "RightLeg", "RightShin", "RightFoot"],
        }
    )
    # No per-limb damage multiplier: kick-vs-punch hardness is left entirely to
    # the physics (energy = contact force x closing speed; legs hit harder via
    # their greater mass). Kept as a 1.0 no-op knob in case a mild, explicit
    # nudge is wanted later.
    strike_group_multipliers: Dict[str, float] = field(
        default_factory=lambda: {"hands": 1.0, "legs": 1.0}
    )
    damage_body_names: List[str] = field(
        default_factory=lambda: ["Head", "Chest", "Spine2", "Spine1", "Hips"]
    )
    # Region multipliers, aligned with damage_body_names. Spine2/Spine1 are the
    # mid-torso (upper/lower abdomen — solar plexus / liver), previously not
    # damageable so body shots to the stomach scored nothing.
    # head > stomach > chest > pelvis (a clean liver/solar-plexus shot lands hard)
    damage_multipliers: List[float] = field(
        default_factory=lambda: [2.0, 1.0, 1.25, 1.25, 0.5]
    )
    # Key bodies exposed in opponent observations
    key_body_names: List[str] = field(
        default_factory=lambda: ["Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot"]
    )
    # Head body for the gaze-based facing reward
    head_body_name: str = "Head"

    # Kick-attempt shaping (Eric, 2026-07-26): pay a small bonus each time a
    # foot rises above kick_bonus_height (a real kick-height lift), with
    # hysteresis (must drop below kick_bonus_rearm_height to re-arm) and a
    # per-foot per-episode cap — teaches the league to TRY kicks, which the
    # early punch meta otherwise prunes. Weighted via the dense reward
    # component, so it anneals with --dense-reward-scale.
    kick_bonus_left_foot_body: str = "LeftFoot"
    kick_bonus_right_foot_body: str = "RightFoot"
    kick_bonus_height: float = 0.75
    kick_bonus_rearm_height: float = 0.40
    kick_bonus_max_per_foot: int = 3
    # Where the gaze should POINT: boxers watch the opponent's upper chest /
    # shoulder line (torso telegraphs strikes; hands are too fast to track,
    # eyes deceive), not the face. The chest is also a steadier target that
    # tolerates bladed stances and slipping better than the head.
    facing_target_body_name: str = "Chest"
    # Gaze direction in the head's local frame. SOMA (SMPL-family) faces
    # body-frame -y; +x (the calc_heading convention) points out the ear.
    gaze_forward_axis: Tuple[float, float, float] = (0.0, -1.0, 0.0)

    # Arena geometry (IsaacLabASE: borderline_space = 7.0 m square)
    arena_size: float = 7.0  # side length in meters
    arena_spacing: float = 16.0  # distance between arena centers (>= 2x arena_size)
    min_spawn_center_distance: float = 1.5  # rejection-sample away from center
    min_spawn_partner_distance: float = 1.5  # and away from the opponent
    spawn_max_fraction: float = 0.8  # spawn within this fraction of the arena

    # Fight rules
    initial_health: float = 1.0
    # Health lost per unit of hit energy taken (region-weighted). The energy
    # source depends on raw_health_damage below: log-normalized (False) or raw
    # thresholded (True) — the two scales are NOT comparable (log ~O(1)/step,
    # raw ~O(100) for a hard strike), so this value must match the mode.
    damage_to_health: float = 0.05
    # False (default): health drains from the LOG-NORMALIZED hit energy — the
    # original model every existing checkpoint/frozen config was built with.
    # True: health drains from the KINETIC ENERGY of qualifying strikes —
    # 0.5 * m_limb * v_impact^2, one deposit per contact event, zero below the
    # hit_state.strike_min_speed impact-speed gate (pushes/leans/grinds score
    # nothing; see HitStateConfig). damage_to_health is then HP-per-joule
    # (e.g. ~0.002: a 70 J head strike x2 region mult removes ~28%). STUN, if
    # gated, also uses the same per-hit KE (divided by stun_raw_energy_ref) so
    # only genuine strikes concuss. False (default): the original
    # log-normalized damage model every existing frozen config was built with.
    raw_health_damage: bool = False
    # KE (joules) counting as "one full unit" of stun input in KE mode — set
    # near a solid strike's energy so one clean head hit concusses briefly and
    # body hits barely register.
    stun_raw_energy_ref: float = 70.0
    # Hard per-hit ceiling on HP removed by a single contact event in KE mode.
    # Backstop against any residual state glitch: no single touch can wipe a
    # health bar regardless of measured energy.
    max_hp_per_hit: float = 0.25
    knockdown_height: float = 0.2  # m, root below this counts as "down"
    knockdown_grace_seconds: float = 2.0  # get-up window before KO
    # Referee's count: down for ANY reason this long loses the bout,
    # regardless of stun (the concussion gate covers real KOs; this is the
    # backstop so a fighter can't camp on the canvas).
    count_out_seconds: float = 5.0
    points_decision_eps: float = 0.02  # health diff below this at timeout = draw

    # Stun / concussion model. A hit deposits stun scaled by its (region-
    # weighted) energy; stun decays at a FIXED rate, so both the peak and the
    # duration grow with hit hardness from one accumulator. A down fighter is a
    # KO only while still stunned (stun > stun_ko_threshold): a trip or self-
    # fall deposits no stun and can never be a knockout, and a hard enough hit
    # keeps stun above threshold past the get-up window, guaranteeing a KO.
    # stun_region_weights is aligned to damage_body_names — head dominates
    # (a head shot scrambles you; body shots barely stun). (Stage 2 will also
    # drive IMU/proprioception disorientation noise from this same stun value.)
    stun_gain: float = 3.0
    stun_decay_per_sec: float = 0.5
    stun_ko_threshold: float = 0.4
    stun_region_weights: List[float] = field(
        default_factory=lambda: [1.0, 0.1, 0.15, 0.1, 0.05]
    )
    # When True a knockout requires being down AND still stunned (the new
    # concussion model); when False the KO reverts to the original "down past
    # the grace window" rule regardless of stun. Off by default so the striking
    # changes can be validated in isolation before the KO mechanic is enabled.
    stun_gates_ko: bool = False
    # Outcome signal for drawn matches (both fighters). Slightly negative so
    # running out the clock is never the safe harbor — engaging (points win
    # +1 / loss -1, symmetric zero EV) strictly dominates mutual passivity.
    draw_signal: float = -0.25
    # Decisive wins/losses scale by (1 + this * time_left_frac): finishing
    # early pays more, losing early costs more. 0 disables. See the
    # early-finish block in _update_match_state.
    early_finish_win_scale: float = 1.0
    # Out of bounds ends the match with a POINTS DECISION (like timeout)
    # rather than an instant loss: shoving the opponent out only "wins" if
    # you were already ahead on damage, so ring-outs stop being a strategy.
    # Set True to restore instant-loss ring-outs.
    out_of_bounds_loses: bool = False

    # Anti-stalling (IsaacLabASE: +0.005/step below 1.0 rad/s max joint speed)
    idle_joint_speed: float = 1.0
    idle_time_increment: float = 0.005

    # Fall-state initialization curriculum (AmpGetupEnv lineage)
    fall_init_prob: float = 0.1
    recovery_seconds: float = 2.0  # termination suppression after a fall init

    # Viewer: markers per arena side outlining the ring (0 disables)
    arena_ring_markers_per_side: int = 8
    arena_ring_marker_scale: float = 0.02

    # Hit FSM constants
    hit_state: HitStateConfig = field(default_factory=HitStateConfig)

    # Cross-morphology (MULTI_ROBOT_LEAGUE_PLAN Phase 3): body tables for the
    # OPPONENT block (envs [N..2N)) when it hosts a different robot. A dict of
    # battle_table_kwargs(opp_robot) entries PLUS "body_names" (the opponent
    # robot's kinematic body list) and optionally per-stature scalar
    # overrides: "knockdown_height", "kick_bonus_height",
    # "kick_bonus_rearm_height", "default_root_height". None = both sides use
    # the primary tables (single-morphology, exact legacy behavior).
    opponent_tables: Dict = None


class BattleControl(ControlComponent):
    """Stateful fight manager for paired-env battles."""

    def __init__(self, config: BattleControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self.config: BattleControlConfig = config

        num_envs = env.num_envs
        if num_envs % 2 != 0:
            raise ValueError(
                f"Battle environments must come in pairs; got num_envs={num_envs}"
            )
        self.num_matches = num_envs // 2
        device = env.device

        # partner[i] = the env index of i's opponent
        self.partner = (
            torch.arange(num_envs, device=device, dtype=torch.long) + self.num_matches
        ) % num_envs

        # Arena centers: one per match, laid out on a square grid, shared by
        # both sides of the pair.
        self.arena_centers = self._build_arena_centers()  # [2N, 2]

        # ---- Per-side body tables (MULTI_ROBOT_LEAGUE_PLAN Phase 3) -------
        # Side A = ego block [0..N), side B = opponent block [N..2N). With
        # config.opponent_tables unset, side B duplicates side A and every
        # per-env tensor is a constant row — numerically identical to the
        # legacy single-table path (gather == index).
        side_a = self._resolve_side_tables(
            {
                "strike_body_names": config.strike_body_names,
                "strike_body_group_names": config.strike_body_group_names,
                "strike_group_multipliers": config.strike_group_multipliers,
                "damage_body_names": config.damage_body_names,
                "damage_multipliers": config.damage_multipliers,
                "stun_region_weights": config.stun_region_weights,
                "key_body_names": config.key_body_names,
                "head_body_name": config.head_body_name,
                "facing_target_body_name": config.facing_target_body_name,
                "gaze_forward_axis": config.gaze_forward_axis,
                "kick_bonus_left_foot_body": config.kick_bonus_left_foot_body,
                "kick_bonus_right_foot_body": config.kick_bonus_right_foot_body,
                "knockdown_height": config.knockdown_height,
                "kick_bonus_height": config.kick_bonus_height,
                "kick_bonus_rearm_height": config.kick_bonus_rearm_height,
                "default_root_height": env.robot_config.default_root_height,
            },
            env.robot_config.kinematic_info.body_names,
        )
        opp_tables = getattr(config, "opponent_tables", None)
        self._cross_morph = opp_tables is not None
        if self._cross_morph:
            merged = dict(side_a["raw"])
            merged.update({k: v for k, v in opp_tables.items() if k != "body_names"})
            side_b = self._resolve_side_tables(merged, opp_tables["body_names"])
            if side_a["group_labels"] != side_b["group_labels"]:
                raise ValueError(
                    "Strike group labels must match across sides "
                    f"({side_a['group_labels']} vs {side_b['group_labels']})"
                )
            if len(side_a["key_ids"]) != len(side_b["key_ids"]):
                raise ValueError(
                    "Key body count (K) must match across sides — the "
                    "opponent obs kernel width is 20 + 6K league-wide"
                )
        else:
            side_b = side_a
        self.strike_group_labels = side_a["group_labels"]
        # Family table bank (rung 4): 0 = ego robot, 1 = the arena's foreign
        # body. Every env boots family 0 (pure self-play) and flips per
        # match via set_opponent_family when the league serves a foreign
        # snapshot.
        self._family_bank = [side_a, side_b]
        self.opponent_family = torch.zeros(
            self.num_matches, dtype=torch.long, device=device
        )

        half = self.num_matches
        widths = {
            "strike": max(len(f["strike_ids"]) for f in self._family_bank),
            "damage": max(len(f["damage_ids"]) for f in self._family_bank),
        }
        self._table_widths = widths

        def full_env(vec, width=None, dtype=torch.long, pad=0):
            """[2N, width] all rows = vec (family-0 boot state) + mask."""
            width = width or len(vec)
            out = torch.full((num_envs, width), pad, dtype=dtype, device=device)
            mask = torch.zeros(num_envs, width, dtype=torch.bool, device=device)
            t = torch.as_tensor(vec, dtype=dtype, device=device)
            out[:, : len(vec)] = t
            mask[:, : len(vec)] = True
            return out, mask

        # Own-side per-env tables (family-0 everywhere at boot)
        self.key_body_ids, _ = full_env(side_a["key_ids"])
        self.head_body_id, _ = full_env([side_a["head_id"]])
        self.head_body_id = self.head_body_id.squeeze(-1)
        self.facing_target_body_id, _ = full_env([side_a["facing_id"]])
        self.facing_target_body_id = self.facing_target_body_id.squeeze(-1)
        self.kick_foot_body_ids, _ = full_env(side_a["kick_ids"])
        self.damage_body_ids, self._damage_mask = full_env(
            side_a["damage_ids"], widths["damage"]
        )
        # Partner-side strike tables: row i indexes the OPPONENT's bodies.
        self.strike_body_ids, self._strike_mask = full_env(
            side_a["strike_ids"], widths["strike"]
        )
        strike_groups_pe, _ = full_env(side_a["groups"], widths["strike"])
        strike_mults_pe, _ = full_env(
            side_a["strike_mults"], widths["strike"], dtype=torch.float
        )
        damage_mults_pe, _ = full_env(
            side_a["damage_mults"], widths["damage"], dtype=torch.float
        )
        self._stun_region_weights, _ = full_env(
            side_a["stun_weights"], widths["damage"], dtype=torch.float
        )
        def scalar_pe(a):
            return torch.full((num_envs,), float(a), device=device)
        self.knockdown_height_pe = scalar_pe(side_a["knockdown_height"])
        self.kick_bonus_height_pe = scalar_pe(side_a["kick_bonus_height"])
        self.kick_bonus_rearm_pe = scalar_pe(side_a["kick_bonus_rearm_height"])
        self.default_root_height_pe = scalar_pe(side_a["default_root_height"])
        self._gaze_axis_pe = torch.tensor(
            side_a["gaze_axis"], device=device, dtype=torch.float
        ).expand(num_envs, 3).contiguous()
        self._side_strike_ids = (side_a["strike_ids"], side_b["strike_ids"])
        self._family_strike_masses = [None] * len(self._family_bank)

        # Per-family PD action scaling (rung 4): the env action config may
        # carry family-PAIR offset/scale tensors [2, D] (row 0 = ego family,
        # row 1 = the arena's foreign body) or, from pre-rung-4 frozen
        # configs, per-env [2N_train, D] block tensors. Either way, adopt
        # them into LIVE per-env tensors sized to THIS run's env count,
        # boot-filled with the ego family's rows, and re-stamped per match
        # in set_opponent_family. The action function reads the config dict
        # every step, so in-place updates propagate.
        self._pd_pairs = {}
        action_cfg = getattr(env.config, "action_config", None)
        if self._cross_morph and isinstance(action_cfg, dict):
            for key in ("pd_action_offset", "pd_action_scale"):
                val = action_cfg.get(key)
                if not (torch.is_tensor(val) and val.dim() == 2):
                    continue
                if val.shape[0] == 2:
                    pairs = val.to(device)
                elif val.shape[0] >= 2:
                    # legacy per-env block layout: rows [0, N_train]
                    pairs = val[[0, val.shape[0] // 2]].to(device)
                else:
                    continue
                self._pd_pairs[key] = pairs
                live = pairs[0].unsqueeze(0).expand(num_envs, -1).contiguous()
                action_cfg[key] = live

        self.hit_state = BattleHitState(
            num_envs=num_envs,
            damage_body_ids=self.damage_body_ids,
            strike_body_ids=self.strike_body_ids,
            damage_multipliers=damage_mults_pe * self._damage_mask,
            config=config.hit_state,
            dt=env.dt,
            device=device,
            strike_body_groups=strike_groups_pe,
            num_strike_groups=len(self.strike_group_labels),
            strike_multipliers=strike_mults_pe,
            # KE mode uses the same per-event physics for the dense reward
            # (continuous, ungated log1p(KE/ref)) as for health (speed-gated).
            reward_from_event_ke=config.raw_health_damage,
            damage_mask=self._damage_mask,
            strike_mask=self._strike_mask,
        )
        self._stun_region_weights = self._stun_region_weights * self._damage_mask

        # Fight state
        self.health = torch.full((num_envs,), config.initial_health, device=device)
        self.down_timer = torch.zeros(num_envs, device=device)
        self.stun = torch.zeros(num_envs, device=device)
        self.idle_time = torch.zeros(num_envs, device=device)
        self.recovery_steps_left = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.hit_energy_taken = torch.zeros(num_envs, device=device)
        self.hit_energy_dealt = torch.zeros(num_envs, device=device)
        # Cumulative dealt energy per strike group this episode + the
        # per-step diversity bonus (growth of the lesser group's cumulative)
        num_groups = len(self.strike_group_labels)
        self.dealt_by_group_cum = torch.zeros(num_envs, num_groups, device=device)
        self.strike_diversity_bonus = torch.zeros(num_envs, device=device)
        # Kick-attempt shaping state: per-foot event counters + hysteresis arm
        self.kick_counts = torch.zeros(
            num_envs, 2, dtype=torch.long, device=device)
        self.kick_armed = torch.ones(
            num_envs, 2, dtype=torch.bool, device=device)
        self.kick_attempt_bonus = torch.zeros(num_envs, device=device)

        # Outcome buffers, stamped on the step the match ends
        self.win_signal = torch.zeros(num_envs, device=device)
        self.match_ended = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._terminate = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.end_cause_ko = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.end_cause_oob = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.end_cause_points = torch.zeros(num_envs, dtype=torch.bool, device=device)

        self._knockdown_grace_steps = max(
            1, int(round(config.knockdown_grace_seconds / env.dt))
        )
        self._recovery_steps = max(1, int(round(config.recovery_seconds / env.dt)))

        # Ring outline offsets around the arena center [P, 2] (viewer only)
        self._ring_offsets = self._build_ring_offsets()
        # Sent once — the ring is static, so re-sending every step would just
        # be wasted transform writes.
        self._ring_sent = False
        self._draw_ring = self._ring_offsets.shape[0] > 0

        # Gaze/facing state for the potential-based facing reward.
        # prev < 0 marks "just reset": the first post-reset delta is zero.
        self.facing = torch.zeros(num_envs, device=device)
        self.facing_delta = torch.zeros(num_envs, device=device)
        self._prev_facing = torch.full((num_envs,), -1.0, device=device)

    def set_opponent_family(self, match_ids: Tensor, families: Tensor) -> None:
        """Re-point per-env tables for matches whose opponent family changed.

        Row j = N + match carries the OPPONENT's own-side tables; row i =
        match carries the ego's partner-side (strike) tables — both flip
        with the opponent's family. The ego row's own tables and the
        opponent row's strike tables (its partner is always the ego robot)
        never change."""
        changed = self.opponent_family[match_ids] != families
        if not changed.any():
            return
        match_ids = match_ids[changed]
        families = families[changed]
        self.opponent_family[match_ids] = families
        device = self.env.device
        hs = self.hit_state
        for fam in families.unique().tolist():
            bank = self._family_bank[fam]
            rows_j = (match_ids[families == fam] + self.num_matches).to(device)
            rows_i = match_ids[families == fam].to(device)

            def fill(dst, vec, rows, mask=None, dtype=None):
                t = torch.as_tensor(vec, device=device,
                                    dtype=dtype or dst.dtype)
                dst[rows] = 0 if dst.dtype != torch.bool else False
                dst[rows.unsqueeze(-1),
                    torch.arange(len(vec), device=device)] = t
                if mask is not None:
                    mask[rows] = False
                    mask[rows.unsqueeze(-1),
                         torch.arange(len(vec), device=device)] = True

            # Opponent row: its own body tables
            fill(self.key_body_ids, bank["key_ids"], rows_j)
            self.head_body_id[rows_j] = int(bank["head_id"])
            self.facing_target_body_id[rows_j] = int(bank["facing_id"])
            fill(self.kick_foot_body_ids, bank["kick_ids"], rows_j)
            fill(self.damage_body_ids, bank["damage_ids"], rows_j,
                 self._damage_mask)
            fill(hs.damage_body_ids, bank["damage_ids"], rows_j)
            hs.damage_mask[rows_j] = self._damage_mask[rows_j]
            fill(hs.damage_multipliers, bank["damage_mults"], rows_j)
            fill(self._stun_region_weights, bank["stun_weights"], rows_j)
            self.knockdown_height_pe[rows_j] = bank["knockdown_height"]
            self.kick_bonus_height_pe[rows_j] = bank["kick_bonus_height"]
            self.kick_bonus_rearm_pe[rows_j] = bank["kick_bonus_rearm_height"]
            self.default_root_height_pe[rows_j] = bank["default_root_height"]
            self._gaze_axis_pe[rows_j] = torch.tensor(
                bank["gaze_axis"], device=device, dtype=torch.float
            )
            # Opponent row: its own family's PD action scaling
            for key, pairs in self._pd_pairs.items():
                self.env.config.action_config[key][rows_j] = pairs[fam]
            # Ego row: partner-side strike tables point at the new opponent
            fill(self.strike_body_ids, bank["strike_ids"], rows_i,
                 self._strike_mask)
            fill(hs.strike_body_ids, bank["strike_ids"], rows_i)
            hs.strike_mask[rows_i] = self._strike_mask[rows_i]
            if hs.strike_body_groups is not None:
                fill(hs.strike_body_groups, bank["groups"], rows_i)
            if hs.strike_multipliers is not None:
                fill(hs.strike_multipliers, bank["strike_mults"], rows_i)
            if hs.strike_body_masses is not None:
                masses = self._family_strike_mass(fam)
                if masses is not None:
                    fill(hs.strike_body_masses, masses, rows_i)

    def _family_strike_mass(self, fam: int):
        """Strike-limb masses (kg) for one family, from the simulator."""
        if self._family_strike_masses[fam] is None:
            get = getattr(self.env.simulator, "get_body_masses_for", None)
            if get is not None:
                body_masses = get(family_b=(fam == 1))
                ids = torch.as_tensor(
                    self._family_bank[fam]["strike_ids"],
                    device=body_masses.device,
                )
                self._family_strike_masses[fam] = body_masses[ids]
        return self._family_strike_masses[fam]

    @staticmethod
    def _gather_body(t: Tensor, ids: Tensor) -> Tensor:
        """Per-env single-body gather: t [2N, B, C] + ids [2N] -> [2N, C]."""
        idx = ids.view(-1, 1, 1).expand(-1, 1, t.shape[-1])
        return t.gather(1, idx).squeeze(1)

    @staticmethod
    def _gather_bodies(t: Tensor, ids: Tensor) -> Tensor:
        """Per-env multi-body gather: t [2N, B, C] + ids [2N, K] -> [2N, K, C]."""
        idx = ids.unsqueeze(-1).expand(-1, -1, t.shape[-1])
        return t.gather(1, idx)

    @staticmethod
    def _resolve_side_tables(raw: dict, body_names) -> dict:
        """Resolve one side's name tables into id vectors + scalars."""
        group_labels = list(raw["strike_body_group_names"].keys())
        group_of = {}
        for group_idx, (_label, names) in enumerate(
            raw["strike_body_group_names"].items()
        ):
            for name in names:
                group_of[name] = group_idx
        return {
            "raw": raw,
            "group_labels": group_labels,
            "strike_ids": resolve_body_ids(raw["strike_body_names"], body_names),
            "damage_ids": resolve_body_ids(raw["damage_body_names"], body_names),
            "key_ids": resolve_body_ids(raw["key_body_names"], body_names),
            "head_id": int(
                resolve_body_ids([raw["head_body_name"]], body_names)[0]
            ),
            "facing_id": int(
                resolve_body_ids([raw["facing_target_body_name"]], body_names)[0]
            ),
            "kick_ids": resolve_body_ids(
                [raw["kick_bonus_left_foot_body"],
                 raw["kick_bonus_right_foot_body"]],
                body_names,
            ),
            "groups": [
                group_of.get(name, 0) for name in raw["strike_body_names"]
            ],
            "strike_mults": [
                raw["strike_group_multipliers"].get(
                    group_labels[group_of.get(name, 0)], 1.0
                )
                for name in raw["strike_body_names"]
            ],
            "damage_mults": list(raw["damage_multipliers"]),
            "stun_weights": list(raw["stun_region_weights"]),
            "gaze_axis": tuple(raw["gaze_forward_axis"]),
            "knockdown_height": raw["knockdown_height"],
            "kick_bonus_height": raw["kick_bonus_height"],
            "kick_bonus_rearm_height": raw["kick_bonus_rearm_height"],
            "default_root_height": raw["default_root_height"],
        }

    def _build_ring_offsets(self) -> Tensor:
        """Evenly spaced XY offsets tracing the square arena boundary."""
        per_side = self.config.arena_ring_markers_per_side
        if per_side <= 0:
            return torch.zeros(0, 2, device=self.env.device)
        half = self.config.arena_size / 2.0
        t = torch.linspace(-half, half, per_side + 1, device=self.env.device)[:-1]
        ones = torch.full_like(t, half)
        # Four sides, corners included once each
        pts = torch.cat(
            [
                torch.stack([t, ones], dim=-1),  # top: left -> right
                torch.stack([ones, -t], dim=-1),  # right: top -> bottom
                torch.stack([-t, -ones], dim=-1),  # bottom: right -> left
                torch.stack([-ones, t], dim=-1),  # left: bottom -> top
            ],
            dim=0,
        )
        return pts  # [4 * per_side, 2]

    # ------------------------------------------------------------------
    # Arena layout
    # ------------------------------------------------------------------
    def _build_arena_centers(self) -> Tensor:
        """Arena centers on a square grid, per env (both partners share one).

        The grid starts past the terrain border (border cells are invalid
        spawn area) and must fit inside the generated terrain extent.
        """
        cfg = self.config
        grid = math.ceil(math.sqrt(self.num_matches))

        origin = 0.0
        terrain_cfg = getattr(self.env.terrain, "config", None)
        if terrain_cfg is not None:
            origin = float(getattr(terrain_cfg, "border_size", 0.0))
            extent_x = terrain_cfg.map_length * terrain_cfg.num_levels
            extent_y = terrain_cfg.map_width * terrain_cfg.num_terrains
            required = grid * cfg.arena_spacing
            if required > min(extent_x, extent_y):
                raise ValueError(
                    f"Arena grid needs {required:.0f}m but the terrain interior "
                    f"is only {extent_x:.0f}x{extent_y:.0f}m. Increase "
                    "terrain map_length/map_width (or num_levels/num_terrains) "
                    f"to fit {self.num_matches} matches at "
                    f"arena_spacing={cfg.arena_spacing}."
                )

        centers = torch.zeros(self.num_matches, 2, device=self.env.device)
        for m in range(self.num_matches):
            row, col = divmod(m, grid)
            centers[m, 0] = origin + (col + 0.5) * cfg.arena_spacing
            centers[m, 1] = origin + (row + 0.5) * cfg.arena_spacing
        return centers.repeat(2, 1)  # env i and i+N share centers[i % N]

    def sample_spawn_positions(self, env_ids: Tensor) -> Tensor:
        """Rejection-sample spawn XY within the arena for the given envs.

        Positions are at least ``min_spawn_center_distance`` from the arena
        center; partner separation is enforced by
        :meth:`enforce_partner_separation` once both sides are placed.
        """
        cfg = self.config
        n = len(env_ids)
        device = self.env.device
        centers = self.arena_centers[env_ids]
        max_extent = cfg.arena_size * cfg.spawn_max_fraction

        pos = centers + (torch.rand(n, 2, device=device) - 0.5) * max_extent
        for _ in range(100):
            too_close = (
                torch.norm(pos - centers, dim=-1) < cfg.min_spawn_center_distance
            )
            if not too_close.any():
                break
            resample = centers[too_close] + (
                torch.rand(int(too_close.sum()), 2, device=device) - 0.5
            ) * max_extent
            pos[too_close] = resample
        return pos

    def enforce_partner_separation(self, env_ids: Tensor, spawn_xy: Tensor) -> Tensor:
        """Push apart partners that were sampled closer than the minimum."""
        cfg = self.config
        pos_map: Dict[int, int] = {
            int(e): i for i, e in enumerate(env_ids.tolist())
        }
        for i, e in enumerate(env_ids.tolist()):
            p = int(self.partner[e])
            j = pos_map.get(p)
            if j is None or j <= i:
                continue
            delta = spawn_xy[j] - spawn_xy[i]
            dist = torch.norm(delta)
            if dist < cfg.min_spawn_partner_distance:
                direction = (
                    delta / dist
                    if dist > 1e-6
                    else torch.tensor([1.0, 0.0], device=spawn_xy.device)
                )
                push = (cfg.min_spawn_partner_distance - dist) * 0.5 + 1e-3
                spawn_xy[i] = spawn_xy[i] - direction * push
                spawn_xy[j] = spawn_xy[j] + direction * push
        return spawn_xy

    # ------------------------------------------------------------------
    # ControlComponent API
    # ------------------------------------------------------------------
    def reset(self, env_ids: Tensor):
        if len(env_ids) == 0:
            return
        cfg = self.config
        self.health[env_ids] = cfg.initial_health
        self.down_timer[env_ids] = 0.0
        self.stun[env_ids] = 0.0
        self.idle_time[env_ids] = 0.0
        self.hit_energy_taken[env_ids] = 0.0
        self.hit_energy_dealt[env_ids] = 0.0
        self.dealt_by_group_cum[env_ids] = 0.0
        self.strike_diversity_bonus[env_ids] = 0.0
        self.kick_counts[env_ids] = 0
        self.kick_armed[env_ids] = True
        self.kick_attempt_bonus[env_ids] = 0.0
        self.facing[env_ids] = 0.0
        self.facing_delta[env_ids] = 0.0
        self._prev_facing[env_ids] = -1.0
        self.win_signal[env_ids] = 0.0
        self.match_ended[env_ids] = False
        self._terminate[env_ids] = False
        self.hit_state.reset(env_ids)

        # Fall-init curriculum: recently fall-initialized envs get a recovery
        # window during which knockout cannot fire (the fighter is expected to
        # be down; it must learn to get up).
        fall_mask = getattr(self.env, "battle_fall_init_mask", None)
        if fall_mask is not None:
            recover = env_ids[fall_mask[env_ids]]
            self.recovery_steps_left[env_ids] = 0
            self.recovery_steps_left[recover] = self._recovery_steps
        else:
            self.recovery_steps_left[env_ids] = 0

    def step(self):
        """Advance fight state one control step and stamp match outcomes."""
        cfg = self.config
        env = self.env
        state = env.simulator.get_robot_state()
        partner = self.partner

        body_pos = state.rigid_body_pos
        body_vel = state.rigid_body_vel
        contact_forces = state.rigid_body_contact_forces

        # Hit integration (energy TAKEN per env; dealt = partner's taken)
        # Lazy one-time wiring of real limb masses for the KE damage model
        # (masses are only available once the sim articulation is initialized).
        if cfg.raw_health_damage and self.hit_state.strike_body_masses is None:
            try:
                # Row i's strike columns index the PARTNER robot's bodies:
                # per-family mass vectors, laid out by the current opponent
                # assignment (all-ego at boot; set_opponent_family refreshes
                # rows when a foreign opponent moves in).
                half = self.num_matches
                width = self.strike_body_ids.shape[1]
                if hasattr(env.simulator, "get_body_masses_for"):
                    mass_ego = self._family_strike_mass(0)
                else:
                    masses = env.simulator.get_body_masses()
                    ids = torch.as_tensor(
                        self._family_bank[0]["strike_ids"], device=masses.device
                    )
                    mass_ego = masses.mean(dim=0)[ids]
                    self._family_strike_masses[0] = mass_ego
                mass_pe = torch.zeros(env.num_envs, width, device=env.device)
                mass_pe[:, : len(mass_ego)] = mass_ego.to(env.device)
                self.hit_state.set_strike_body_masses(mass_pe)
                # Re-stamp any matches already hosting a foreign family.
                foreign = (self.opponent_family != 0).nonzero(as_tuple=False).flatten()
                if len(foreign) > 0:
                    fams = self.opponent_family[foreign].clone()
                    self.opponent_family[foreign] = 0  # force refresh
                    self.set_opponent_family(foreign, fams)
                log.info(
                    "KE damage: ego strike-limb masses (kg) = %s",
                    [round(float(m), 2) for m in mass_ego],
                )
            except NotImplementedError:
                log.warning(
                    "Simulator exposes no body masses; KE damage uses unit "
                    "masses (pure speed^2)."
                )
                self.hit_state.set_strike_body_masses(
                    torch.ones_like(self.strike_body_ids, dtype=torch.float)
                )

        taken, taken_by_group, taken_per_body, ke_per_body = self.hit_state.step(
            contact_forces=contact_forces,
            body_pos=body_pos,
            body_vel=body_vel,
            opp_body_pos=body_pos[partner],
            opp_body_vel=body_vel[partner],
            progress=env.progress_buf,
        )
        # REWARD stream: log-normalized hit energy (stable magnitudes).
        self.hit_energy_taken = taken
        self.hit_energy_dealt = taken[partner]
        # HEALTH: in KE mode, one deposit per qualifying strike —
        # damage_to_health (HP/joule) x KE x region multiplier, each hit capped
        # at max_hp_per_hit. Otherwise the original log-normalized model.
        if cfg.raw_health_damage:
            hp_per_hit = (
                cfg.damage_to_health
                * ke_per_body
                * self.hit_state.damage_multipliers
            ).clamp_max(cfg.max_hp_per_hit)
            hp_loss = hp_per_hit.sum(dim=-1)
            self.health = (self.health - hp_loss).clamp_min(0.0)
        else:
            self.health = (
                self.health - cfg.damage_to_health * taken
            ).clamp_min(0.0)

        # Stun accumulator: deposit region-weighted (head-heavy) hit energy,
        # decay at a fixed rate. Both peak and duration scale with hit hardness.
        # KE mode: stun comes from the same per-hit kinetic energy as health
        # (normalized by stun_raw_energy_ref), so pushes and taps deposit zero
        # stun and the stun_gates_ko concussion gate means what it says.
        if cfg.raw_health_damage:
            stun_input = (
                (ke_per_body / max(cfg.stun_raw_energy_ref, 1e-6))
                * self._stun_region_weights
            ).sum(dim=-1)
        else:
            stun_input = (taken_per_body * self._stun_region_weights).sum(dim=-1)
        self.stun = (
            self.stun + cfg.stun_gain * stun_input - cfg.stun_decay_per_sec * env.dt
        ).clamp_min(0.0)

        # Kickboxing diversity: reward growth of the LESSER cumulative
        # dealt-energy group (hands vs legs) — specialization in one limb
        # group stops earning this stream.
        dealt_by_group = taken_by_group[partner]  # [2N, G]
        prev_min = self.dealt_by_group_cum.min(dim=-1).values
        self.dealt_by_group_cum = self.dealt_by_group_cum + dealt_by_group
        new_min = self.dealt_by_group_cum.min(dim=-1).values
        self.strike_diversity_bonus = (new_min - prev_min).clamp_min(0.0)

        # Kick-attempt shaping: one bonus unit per armed foot crossing
        # kick_bonus_height, capped per foot per episode; foot re-arms
        # after dropping below kick_bonus_rearm_height (no leg-hold farming).
        foot_z = body_pos[..., 2].gather(1, self.kick_foot_body_ids)  # [2N, 2]
        up = foot_z > self.kick_bonus_height_pe.unsqueeze(-1)
        fired = self.kick_armed & up & (
            self.kick_counts < cfg.kick_bonus_max_per_foot)
        self.kick_counts = self.kick_counts + fired.long()
        self.kick_armed = torch.where(
            fired, torch.zeros_like(self.kick_armed), self.kick_armed)
        self.kick_armed = self.kick_armed | (
            foot_z < self.kick_bonus_rearm_pe.unsqueeze(-1))
        self.kick_attempt_bonus = fired.float().sum(dim=-1)

        # Knockdown timer
        root_height = body_pos[:, 0, 2]
        down = root_height < self.knockdown_height_pe
        self.down_timer = torch.where(
            down, self.down_timer + env.dt, torch.zeros_like(self.down_timer)
        )
        self.recovery_steps_left = (self.recovery_steps_left - 1).clamp_min(0)

        # Idle/stalling accounting
        max_joint_speed = state.dof_vel.abs().max(dim=-1).values
        self.idle_time = torch.where(
            max_joint_speed >= cfg.idle_joint_speed,
            torch.zeros_like(self.idle_time),
            self.idle_time + cfg.idle_time_increment,
        )

        # Gaze quality + potential-based delta (SOMA faces body-frame -y).
        # Gaze originates at own head; the TARGET is the opponent's chest —
        # boxer's soft focus on the shoulder line, not eye contact.
        head_pos = self._gather_body(body_pos, self.head_body_id)
        head_rot = self._gather_body(state.rigid_body_rot, self.head_body_id)
        facing_target = self._gather_body(body_pos, self.facing_target_body_id)
        to_opp = torch.nn.functional.normalize(
            facing_target[partner] - head_pos, dim=-1
        )
        gaze = torch.nn.functional.normalize(
            rotations.quat_rotate(head_rot, self._gaze_axis_pe, True),
            dim=-1,
        )
        facing_now = ((gaze * to_opp).sum(dim=-1) + 1.0) * 0.5
        just_reset = self._prev_facing < 0.0
        self.facing_delta = torch.where(
            just_reset,
            torch.zeros_like(facing_now),
            facing_now - self._prev_facing,
        )
        self.facing = facing_now
        self._prev_facing = facing_now

        # ---- Match-end determination -------------------------------------
        in_recovery = self.recovery_steps_left > 0
        # A down fighter is knocked out if it's past the get-up window. With
        # stun_gates_ko, that ALSO requires still being stunned from a hard hit
        # (so a trip / self-fall can't KO — the concussion model); otherwise it
        # reverts to the original down-past-grace rule. health<=0 is a separate
        # accumulated-damage TKO either way.
        down_ko = self.down_timer > cfg.knockdown_grace_seconds
        if cfg.stun_gates_ko:
            down_ko = down_ko & (self.stun > cfg.stun_ko_threshold)
        # Referee's count-out: down past count_out_seconds loses regardless
        # of stun — canvas-camping is never safe.
        counted_out = self.down_timer > cfg.count_out_seconds
        knocked_out = (down_ko | counted_out | (self.health <= 0.0)) & ~in_recovery

        root_xy = body_pos[:, 0, :2]
        half = cfg.arena_size / 2.0
        oob = (root_xy - self.arena_centers).abs().max(dim=-1).values > half
        loses_now = knocked_out | (oob if cfg.out_of_bounds_loses else torch.zeros_like(oob))

        timeout = env.progress_buf >= env.max_episode_length - 1
        # Out of bounds (when not an instant loss) ends the match like a
        # timeout: points decision on health, so ring-outs aren't a strategy.
        ends_on_points = timeout | timeout[partner]
        if not cfg.out_of_bounds_loses:
            ends_on_points = ends_on_points | oob | oob[partner]

        ends = loses_now | loses_now[partner] | ends_on_points

        win = torch.zeros_like(self.win_signal)
        # Decisive: I win if my opponent loses and I don't (simultaneous = draw)
        win = torch.where(loses_now[partner] & ~loses_now, torch.ones_like(win), win)
        win = torch.where(loses_now & ~loses_now[partner], -torch.ones_like(win), win)
        # Points decision on health difference for non-decisive ends
        health_diff = self.health - self.health[partner]
        points = torch.where(
            health_diff > cfg.points_decision_eps,
            torch.ones_like(win),
            torch.where(
                health_diff < -cfg.points_decision_eps,
                -torch.ones_like(win),
                torch.zeros_like(win),
            ),
        )
        points_only = ends & ~loses_now & ~loses_now[partner]
        win = torch.where(points_only, points, win)

        # Drawn matches (no decisive result, healths within eps — including
        # simultaneous losses) pay draw_signal to BOTH sides so running out
        # the clock is never the safe play.
        drawn = ends & (win.abs() <= 0.5)
        win = torch.where(
            drawn, torch.full_like(win, cfg.draw_signal), win
        )

        # Early-finish bonus: decisive wins/losses scale with time remaining
        # (factor 1 + early_finish_win_scale * time_left_frac, so a first-
        # second KO pays up to 2x a timeout points win at scale 1.0). Without
        # it, ending the fight early is economically irrational — the dense
        # streams pay every step the fight continues, so a knockout forfeits
        # more income than the win reward returns. Draws are never scaled.
        # League bookkeeping is unaffected (env classifies by |win| vs 0.5).
        if cfg.early_finish_win_scale > 0.0:
            time_left = (
                1.0 - env.progress_buf.float() / max(env.max_episode_length, 1)
            ).clamp(0.0, 1.0)
            factor = 1.0 + cfg.early_finish_win_scale * time_left
            # Asymmetric: only WINS earn the early-finish multiplier. When
            # losses were amplified too (up to -2x), risk-taking became
            # strictly dominated - the league converged to guard-crouch
            # chip-and-turtle (2026-07-24 telemetry: KOs 0.0004 -> 0.0000,
            # contact energy falling, points-decision meta).
            win = torch.where(drawn | (win < 0), win, win * factor)

        self.match_ended = ends
        self.win_signal = torch.where(ends, win, torch.zeros_like(win))
        # Decisive ends are true terminations (no value bootstrap); points
        # ends (timeout / ring-out) are resets (bootstrap allowed).
        self._terminate = ends & (loses_now | loses_now[partner])

        # Outcome-cause telemetry: how matches end is the leading indicator
        # of degenerate metas (all-ring-out, all-timeout stalling, ...)
        self.end_cause_ko = ends & (knocked_out | knocked_out[partner])
        self.end_cause_oob = ends & (oob | oob[partner]) & ~self.end_cause_ko
        self.end_cause_points = points_only & ~self.end_cause_oob

    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        reset = self.match_ended.clone()
        terminate = self._terminate.clone()
        return reset, terminate

    def populate_context(self, ctx) -> None:
        env = self.env
        state = env.simulator.get_robot_state()
        partner = self.partner
        cfg = self.config

        body_pos = state.rigid_body_pos
        body_vel = state.rigid_body_vel

        downed_norm = (
            self.down_timer / cfg.knockdown_grace_seconds
        ).clamp(0.0, 1.0)
        time_left = (
            1.0 - env.progress_buf.float() / max(env.max_episode_length, 1)
        ).clamp(0.0, 1.0)

        ctx.battle = BattleContext(
            opp_root_pos=state.root_pos[partner],
            opp_root_rot=state.root_rot[partner],
            opp_root_vel=state.root_vel[partner],
            opp_root_ang_vel=state.root_ang_vel[partner],
            # Key/head positions gathered with each env's OWN ids, then
            # partner-permuted — so an atlas ego reads the t800's key bodies
            # through the t800's table, not its own indices.
            opp_key_body_pos=self._gather_bodies(body_pos, self.key_body_ids)[partner],
            opp_key_body_vel=self._gather_bodies(body_vel, self.key_body_ids)[partner],
            head_pos=self._gather_body(body_pos, self.head_body_id),
            head_rot=self._gather_body(state.rigid_body_rot, self.head_body_id),
            opp_head_pos=self._gather_body(body_pos, self.head_body_id)[partner],
            health=self.health,
            opp_health=self.health[partner],
            downed=downed_norm,
            opp_downed=downed_norm[partner],
            round_time_left=time_left,
            idle_time=self.idle_time,
            hit_energy_dealt=self.hit_energy_dealt,
            hit_energy_taken=self.hit_energy_taken,
            strike_diversity_bonus=self.strike_diversity_bonus,
            kick_attempt_bonus=self.kick_attempt_bonus,
            facing=self.facing,
            facing_delta=self.facing_delta,
            win_signal=self.win_signal,
            match_ended=self.match_ended,
            arena_center=self.arena_centers,
            arena_half_size=cfg.arena_size / 2.0,
        )

    # ------------------------------------------------------------------
    # Ring visualization (viewer only)
    # ------------------------------------------------------------------
    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, VisualizationMarkerConfig]:
        if headless or not self._draw_ring:
            return {}
        cfg = self.config
        num_points = self._ring_offsets.shape[0]
        return {
            "arena_ring": VisualizationMarkerConfig(
                type="sphere",
                color=(0.9, 0.15, 0.1),
                markers=[MarkerConfig(scale=cfg.arena_ring_marker_scale)] * num_points,
            )
        }

    def get_markers_state(self) -> Dict[str, MarkerState]:
        # Ring markers persist once set — only send them the first frame so
        # the viewer isn't re-writing transforms every step.
        if self.env.simulator.headless or self._ring_sent or not self._draw_ring:
            return {}
        self._ring_sent = True
        num_points = self._ring_offsets.shape[0]
        device = self.env.device
        pos = torch.zeros(self.env.num_envs, num_points, 3, device=device)
        pos[..., :2] = self.arena_centers.unsqueeze(1) + self._ring_offsets.unsqueeze(0)
        pos[..., 2] = 0.05
        rot = torch.zeros(self.env.num_envs, num_points, 4, device=device)
        rot[..., 3] = 1.0
        return {"arena_ring": MarkerState(translation=pos, orientation=rot)}


__all__ = ["BattleControlConfig", "BattleControl"]
