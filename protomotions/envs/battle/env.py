# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-character battle environment.

Paired-env layout: with ``2N`` envs, env ``i`` and env ``i + N`` form match
``i``. Both fighters are ordinary single-character ProtoMotions envs — all
observation kernels, PD action processing, and the motion manager keep
operating on flat batches — while :class:`BattleControl` supplies the fight
semantics and this subclass supplies the pairing glue:

- match-atomic resets (resetting one side always resets the other),
- arena-based paired spawning,
- fall-state initialization curriculum,
- per-match win/lose/draw accounting in ``extras``.

Physics requirement: match partners are co-located in one arena, so cross-env
collisions must be enabled (``simulator.config.filter_env_collisions = False``
on the IsaacLab backend). Arenas are spaced far enough apart that fighters
from different matches can never touch; out-of-bounds ends the match well
before a fighter could reach a neighboring arena.

Action-timing note: both fighters' actions are part of one flat action tensor
applied in a single ``simulator.step`` call, so neither side observes the
other's action early — the paired-env layout gives IsaacLabASE's
"opponent actions applied first to eliminate timing bias" convention for free.
"""

from typing import Optional

import torch
from torch import Tensor

from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.battle.control import BattleControl
from protomotions.simulator.base_simulator.simulator_state import RobotState


class BattleEnv(BaseEnv):
    """BaseEnv with paired-match semantics for two-character battles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.num_envs % 2 != 0:
            raise ValueError(
                f"BattleEnv requires an even number of envs (got {self.num_envs})"
            )
        self.num_matches = self.num_envs // 2

        self.battle_control = self._find_battle_control()
        self.partner = self.battle_control.partner

        # Set by reset() before control components reset; consumed by
        # BattleControl.reset to grant a recovery window to fall-initialized envs.
        self.battle_fall_init_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _find_battle_control(self) -> BattleControl:
        for component in self.control_manager.components.values():
            if isinstance(component, BattleControl):
                return component
        raise ValueError(
            "BattleEnv requires a BattleControl entry in config.control_components"
        )

    # ------------------------------------------------------------------
    # Match-atomic resets
    # ------------------------------------------------------------------
    def _expand_to_match_pairs(self, env_ids: Tensor) -> Tensor:
        """Union env_ids with their partners so matches reset as a unit."""
        combined = torch.cat([env_ids, self.partner[env_ids]])
        return torch.unique(combined)

    def reset(self, env_ids=None, **kwargs):
        if env_ids is not None and len(env_ids) > 0:
            if isinstance(env_ids, list):
                env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            env_ids = self._expand_to_match_pairs(env_ids.to(self.device))

            # Fall-state initialization curriculum: a fraction of resets start
            # from a random tumbled pose rather than a reference clip, forcing
            # get-up skills to stay tactically live.
            fall_prob = self.battle_control.config.fall_init_prob
            self.battle_fall_init_mask[env_ids] = (
                torch.rand(len(env_ids), device=self.device) < fall_prob
            )
            if self.battle_fall_init_mask[env_ids].any():
                force_default = kwargs.get("force_default_mask")
                fall_subset = self.battle_fall_init_mask[env_ids]
                if force_default is None:
                    force_default = fall_subset.clone()
                else:
                    force_default = force_default | fall_subset
                kwargs["force_default_mask"] = force_default
        return super().reset(env_ids, **kwargs)

    def compute_default_reset_state(self, env_ids, sample_flat=False):
        """Default reset, with fall-initialized envs tumbled instead of standing."""
        new_states, new_object_states = super().compute_default_reset_state(
            env_ids, sample_flat
        )
        fall_subset = self.battle_fall_init_mask[env_ids]
        if fall_subset.any():
            n_fall = int(fall_subset.sum())
            # Random root orientation, dropped from above standing height so
            # the fighter settles into a fallen pose (AmpGetupEnv lineage:
            # random orientation at 2x init height, zero velocities).
            rand_quat = torch.nn.functional.normalize(
                torch.randn(n_fall, 4, device=self.device), dim=-1
            )
            new_states.root_rot[fall_subset] = rand_quat
            # Drop from above the (already respawn-offset) standing height so
            # the fighter settles into a fallen pose during the recovery window.
            new_states.root_pos[fall_subset, 2] = (
                new_states.root_pos[fall_subset, 2]
                + self.robot_config.default_root_height * 0.5
            )
            new_states.root_vel[fall_subset] = 0.0
            new_states.root_ang_vel[fall_subset] = 0.0
        return new_states, new_object_states

    # ------------------------------------------------------------------
    # Arena-based paired spawning
    # ------------------------------------------------------------------
    def update_respawn_root_offset_by_env_ids(
        self,
        env_ids,
        ref_state: Optional[RobotState] = None,
        sample_flat: bool = False,
    ) -> None:
        """Place resetting fighters inside their match's arena.

        Replaces free terrain sampling with rejection-sampled positions around
        the shared arena center (min distance from center and from the
        partner), then translates the reference pose there.
        """
        control = self.battle_control
        spawn_xy = control.sample_spawn_positions(env_ids)
        spawn_xy = control.enforce_partner_separation(env_ids, spawn_xy)

        respawn_offset = torch.zeros((len(env_ids), 3), device=self.device)
        if ref_state is None:
            ref_root = torch.zeros((len(env_ids), 2), device=self.device)
        else:
            ref_root = ref_state.root_pos[:, :2]
        respawn_offset[:, :2] = spawn_xy - ref_root

        if not self.skip_height_correction:
            if ref_state is not None:
                rigid_body_pos_spawned = ref_state.rigid_body_pos.clone() + (
                    respawn_offset.unsqueeze(1)
                )
            else:
                rigid_body_pos_spawned = respawn_offset.unsqueeze(1)
            terrain_heights = self.terrain.find_terrain_height_for_max_below_body(
                rigid_body_pos_spawned
            )
            respawn_offset[:, 2] = terrain_heights

        respawn_offset[:, 2] += self.config.ref_respawn_offset
        self.respawn_root_offset[env_ids] = respawn_offset

    # ------------------------------------------------------------------
    # Match accounting
    # ------------------------------------------------------------------
    def post_physics_step(self):
        super().post_physics_step()

        control = self.battle_control
        ended = control.match_ended
        win = control.win_signal

        # Per-env outcome flags (an agent slices the ego half [0, N))
        self.extras["battle/match_ended"] = ended
        self.extras["battle/win"] = (ended & (win > 0.5)).float()
        self.extras["battle/lose"] = (ended & (win < -0.5)).float()
        self.extras["battle/draw"] = (ended & (win.abs() <= 0.5)).float()
        self.extras["battle/health"] = control.health
        self.extras["battle/hit_energy_dealt"] = control.hit_energy_dealt
        # Gaze quality: the primary "are they squaring up" telemetry. Healthy
        # fights = high facing AND rising hit energy; high facing with zero
        # hits = wrong axis; low facing = reward too weak.
        self.extras["battle/facing"] = control.facing
        # Outcome-cause telemetry: the leading indicator of degenerate metas
        # (all-ring-out shoving, all-timeout stalling) is HOW matches end.
        self.extras["battle/end_ko"] = control.end_cause_ko.float()
        self.extras["battle/end_ringout"] = control.end_cause_oob.float()
        self.extras["battle/end_points"] = control.end_cause_points.float()
        self.extras["battle/dealt_hands"] = control.dealt_by_group_cum[:, 0]
        if control.dealt_by_group_cum.shape[1] > 1:
            self.extras["battle/dealt_legs"] = control.dealt_by_group_cum[:, 1]

        # Mirror resets across partners so a match always resets as a unit
        # (control-side conditions are already symmetric; this also covers
        # asymmetric sources like tracking-error terminations).
        self.reset_buf[:] = self.reset_buf | self.reset_buf[self.partner]
        self.terminate_buf[:] = self.terminate_buf | self.terminate_buf[self.partner]


__all__ = ["BattleEnv"]
