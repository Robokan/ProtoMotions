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
    strike_body_names: List[str] = field(
        default_factory=lambda: [
            "LeftHand",
            "RightHand",
            "LeftFoot",
            "RightFoot",
            "LeftShin",
            "RightShin",
        ]
    )
    # Strike groups for kickboxing diversity accounting: dealt hit energy is
    # tracked per group so the reward can pay extra for the under-used group
    # (a pure puncher or pure kicker leaves reward on the table).
    strike_body_group_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "hands": ["LeftHand", "RightHand"],
            "legs": ["LeftFoot", "RightFoot", "LeftShin", "RightShin"],
        }
    )
    damage_body_names: List[str] = field(
        default_factory=lambda: ["Head", "Chest", "Hips"]
    )
    # Region multipliers, aligned with damage_body_names (head > torso > pelvis)
    damage_multipliers: List[float] = field(default_factory=lambda: [2.0, 1.0, 0.5])
    # Key bodies exposed in opponent observations
    key_body_names: List[str] = field(
        default_factory=lambda: ["Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot"]
    )
    # Head body for the gaze-based facing reward
    head_body_name: str = "Head"
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
    # Health lost per unit of log-normalized hit energy taken (region-weighted).
    damage_to_health: float = 0.05
    knockdown_height: float = 0.2  # m, root below this counts as "down"
    knockdown_grace_seconds: float = 2.0  # get-up window before KO
    points_decision_eps: float = 0.02  # health diff below this at timeout = draw
    # Outcome signal for drawn matches (both fighters). Slightly negative so
    # running out the clock is never the safe harbor — engaging (points win
    # +1 / loss -1, symmetric zero EV) strictly dominates mutual passivity.
    draw_signal: float = -0.25
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
    # Viewer: flash damaged body parts red for this long after a scoring hit
    hit_flash_seconds: float = 0.35
    hit_flash_marker_scale: float = 0.08

    # Hit FSM constants
    hit_state: HitStateConfig = field(default_factory=HitStateConfig)


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

        body_names = env.robot_config.kinematic_info.body_names
        self.strike_body_ids = resolve_body_ids(
            config.strike_body_names, body_names
        ).to(device)
        self.damage_body_ids = resolve_body_ids(
            config.damage_body_names, body_names
        ).to(device)
        self.key_body_ids = resolve_body_ids(config.key_body_names, body_names).to(
            device
        )
        self.head_body_id = int(
            resolve_body_ids([config.head_body_name], body_names)[0]
        )

        # Map each strike body to its group id (declaration order of
        # strike_body_group_names). Ungrouped strike bodies go to group 0.
        self.strike_group_labels = list(config.strike_body_group_names.keys())
        group_of = {}
        for group_idx, (_label, names) in enumerate(
            config.strike_body_group_names.items()
        ):
            for name in names:
                group_of[name] = group_idx
        strike_groups = torch.tensor(
            [group_of.get(name, 0) for name in config.strike_body_names],
            dtype=torch.long,
        )

        self.hit_state = BattleHitState(
            num_envs=num_envs,
            damage_body_ids=self.damage_body_ids,
            strike_body_ids=self.strike_body_ids,
            damage_multipliers=torch.tensor(
                config.damage_multipliers, dtype=torch.float
            ),
            config=config.hit_state,
            dt=env.dt,
            device=device,
            strike_body_groups=strike_groups,
            num_strike_groups=len(self.strike_group_labels),
        )

        # Fight state
        self.health = torch.full((num_envs,), config.initial_health, device=device)
        self.down_timer = torch.zeros(num_envs, device=device)
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

        # Hit-flash timers per damage body (viewer only): counts down control
        # steps during which the struck part is highlighted red
        self._hit_flash_steps = max(1, int(round(config.hit_flash_seconds / env.dt)))
        self.hit_flash_timer = torch.zeros(
            num_envs, len(self.damage_body_ids), device=device
        )
        # Body-recolor highlighter (viewer only, built lazily) + one-shot ring
        self._highlighter = None
        self._ring_sent = False

        # Gaze/facing state for the potential-based facing reward.
        # prev < 0 marks "just reset": the first post-reset delta is zero.
        self.facing = torch.zeros(num_envs, device=device)
        self.facing_delta = torch.zeros(num_envs, device=device)
        self._prev_facing = torch.full((num_envs,), -1.0, device=device)
        self._gaze_axis = torch.tensor(
            config.gaze_forward_axis, dtype=torch.float, device=device
        )

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
        self.idle_time[env_ids] = 0.0
        self.hit_energy_taken[env_ids] = 0.0
        self.hit_energy_dealt[env_ids] = 0.0
        self.dealt_by_group_cum[env_ids] = 0.0
        self.strike_diversity_bonus[env_ids] = 0.0
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
        taken, taken_by_group, taken_per_body = self.hit_state.step(
            contact_forces=contact_forces,
            body_pos=body_pos,
            body_vel=body_vel,
            opp_body_pos=body_pos[partner],
            opp_body_vel=body_vel[partner],
            progress=env.progress_buf,
        )
        self.hit_energy_taken = taken
        self.hit_energy_dealt = taken[partner]
        self.health = (self.health - cfg.damage_to_health * taken).clamp_min(0.0)

        # Arm/decay the per-part hit flash (viewer only)
        self.hit_flash_timer = torch.where(
            taken_per_body > 1e-6,
            torch.full_like(self.hit_flash_timer, float(self._hit_flash_steps)),
            (self.hit_flash_timer - 1.0).clamp_min(0.0),
        )

        # Kickboxing diversity: reward growth of the LESSER cumulative
        # dealt-energy group (hands vs legs) — specialization in one limb
        # group stops earning this stream.
        dealt_by_group = taken_by_group[partner]  # [2N, G]
        prev_min = self.dealt_by_group_cum.min(dim=-1).values
        self.dealt_by_group_cum = self.dealt_by_group_cum + dealt_by_group
        new_min = self.dealt_by_group_cum.min(dim=-1).values
        self.strike_diversity_bonus = (new_min - prev_min).clamp_min(0.0)

        # Knockdown timer
        root_height = body_pos[:, 0, 2]
        down = root_height < cfg.knockdown_height
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

        # Gaze quality + potential-based delta (SOMA faces body-frame -y)
        head_pos = body_pos[:, self.head_body_id]
        head_rot = state.rigid_body_rot[:, self.head_body_id]
        to_opp = torch.nn.functional.normalize(
            head_pos[partner] - head_pos, dim=-1
        )
        gaze = torch.nn.functional.normalize(
            rotations.quat_rotate(
                head_rot, self._gaze_axis.expand_as(head_pos), True
            ),
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
        knocked_out = (
            (self.down_timer > cfg.knockdown_grace_seconds) | (self.health <= 0.0)
        ) & ~in_recovery

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
            opp_key_body_pos=body_pos[partner][:, self.key_body_ids],
            opp_key_body_vel=body_vel[partner][:, self.key_body_ids],
            head_pos=body_pos[:, self.head_body_id],
            head_rot=state.rigid_body_rot[:, self.head_body_id],
            opp_head_pos=body_pos[partner][:, self.head_body_id],
            health=self.health,
            opp_health=self.health[partner],
            downed=downed_norm,
            opp_downed=downed_norm[partner],
            round_time_left=time_left,
            idle_time=self.idle_time,
            hit_energy_dealt=self.hit_energy_dealt,
            hit_energy_taken=self.hit_energy_taken,
            strike_diversity_bonus=self.strike_diversity_bonus,
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
        # Ring only. Hit flashes recolor the body prims (see BodyHighlighter),
        # adding no per-frame geometry — spawning flash markers halved the
        # viewer frame rate.
        if headless:
            return {}
        cfg = self.config
        num_points = self._ring_offsets.shape[0]
        if num_points == 0:
            return {}
        return {
            "arena_ring": VisualizationMarkerConfig(
                type="sphere",
                color=(0.9, 0.15, 0.1),
                markers=[MarkerConfig(scale=cfg.arena_ring_marker_scale)] * num_points,
            )
        }

    def get_markers_state(self) -> Dict[str, MarkerState]:
        if self.env.simulator.headless:
            return {}

        # Recolor struck body prims red (transition-only USD writes; no
        # per-frame geometry). Lazily construct the highlighter on first call.
        if self.config.hit_flash_seconds > 0 and len(self.damage_body_ids) > 0:
            if self._highlighter is None:
                from protomotions.envs.battle.highlight import BodyHighlighter

                self._highlighter = BodyHighlighter(
                    num_envs=self.env.num_envs,
                    body_names=self.env.robot_config.kinematic_info.body_names,
                    damage_body_ids=self.damage_body_ids,
                )
            self._highlighter.update(self.hit_flash_timer)

        # Ring markers persist once set — only send them the first frame so the
        # viewer isn't re-writing 32 transforms every step.
        if self._ring_sent or self._ring_offsets.shape[0] == 0:
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
