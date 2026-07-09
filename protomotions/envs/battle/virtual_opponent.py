# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Virtual-opponent control for combat SFT.

Combat SFT (plan Phase 4) supervises the PEFT adapter on combat clips while
conditioning on the same ``task_obs`` the battle RLFT will use. There is no
second character during SFT, so this component synthesizes a placeholder
opponent — a point held at strike-appropriate range off the character's
heading, re-jittered periodically (the combat analog of the stock SFT's
future-root-XY target jitter) — and publishes it through ``EnvContext.battle``
so ``battle_task_obs_factory`` produces identically-shaped observations in
both phases.

Key-body positions for the virtual opponent are a fixed humanoid template
(head/hands/feet offsets) around the virtual root; velocities are zero. Fight
scalars are neutral (full health, not downed, no hits).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.battle.context import BattleContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.utils import rotations

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class VirtualOpponentControlConfig(ControlComponentConfig):
    """Placeholder-opponent parameters for combat SFT."""

    _target_: str = (
        "protomotions.envs.battle.virtual_opponent.VirtualOpponentControl"
    )

    # Strike-appropriate engagement range (meters)
    range_min: float = 0.7
    range_max: float = 1.6
    # Bearing jitter around the character's heading (radians)
    bearing_jitter: float = 0.6
    # Re-sample the virtual opponent placement every [min, max] steps
    rejitter_steps_min: int = 30
    rejitter_steps_max: int = 90
    # Arena parameters mirrored from BattleControlConfig so the arena obs
    # terms match the battle env's statistics.
    arena_size: float = 7.0

    # Head body for the gaze-based facing reward
    head_body_name: str = "Head"

    # Humanoid key-body template around the virtual root, [K, 3] offsets in
    # the opponent's local frame (order must match the battle
    # key_body_names: Head, LeftHand, RightHand, LeftFoot, RightFoot —
    # index 0 doubles as the virtual opponent's head).
    key_body_template: List[List[float]] = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.65],  # Head (relative to root at ~0.95m)
            [0.05, 0.25, 0.05],  # LeftHand (guard-ish)
            [0.05, -0.25, 0.05],  # RightHand
            [0.0, 0.12, -0.9],  # LeftFoot
            [0.0, -0.12, -0.9],  # RightFoot
        ]
    )


class VirtualOpponentControl(ControlComponent):
    """Synthesizes a BattleContext-compatible placeholder opponent."""

    def __init__(self, config: VirtualOpponentControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self.config: VirtualOpponentControlConfig = config

        num_envs = env.num_envs
        device = env.device
        self._opp_pos = torch.zeros(num_envs, 3, device=device)
        self._opp_rot = torch.zeros(num_envs, 4, device=device)
        self._opp_rot[:, 3] = 1.0  # identity (w-last)
        self._rejitter_at = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._arena_center = torch.zeros(num_envs, 2, device=device)
        self._template = torch.tensor(
            config.key_body_template, dtype=torch.float, device=device
        )
        self._zeros = torch.zeros(num_envs, device=device)
        self._ones = torch.ones(num_envs, device=device)

        from protomotions.envs.battle.hit_state import resolve_body_ids

        self._head_body_id = int(
            resolve_body_ids(
                [config.head_body_name], env.robot_config.kinematic_info.body_names
            )[0]
        )

    def _place_opponent(self, env_ids: Tensor) -> None:
        cfg = self.config
        n = len(env_ids)
        device = self.env.device
        state = self.env.simulator.get_root_state()
        root_pos = state.root_pos[env_ids]
        root_rot = state.root_rot[env_ids]

        heading = rotations.calc_heading(root_rot, True)
        bearing = heading + (torch.rand(n, device=device) * 2.0 - 1.0) * cfg.bearing_jitter
        rng = cfg.range_min + torch.rand(n, device=device) * (
            cfg.range_max - cfg.range_min
        )

        opp_pos = root_pos.clone()
        opp_pos[:, 0] += torch.cos(bearing) * rng
        opp_pos[:, 1] += torch.sin(bearing) * rng
        self._opp_pos[env_ids] = opp_pos

        # Virtual opponent faces the character
        face_theta = bearing + torch.pi
        axis = torch.zeros(n, 3, device=device)
        axis[:, 2] = 1.0
        self._opp_rot[env_ids] = rotations.quat_from_angle_axis(
            face_theta, axis, True
        )

        steps = torch.randint(
            cfg.rejitter_steps_min,
            cfg.rejitter_steps_max + 1,
            (n,),
            device=device,
        )
        self._rejitter_at[env_ids] = self.env.progress_buf[env_ids] + steps

    def reset(self, env_ids: Tensor):
        if len(env_ids) == 0:
            return
        self._place_opponent(env_ids)
        root_pos = self.env.simulator.get_root_state().root_pos[env_ids]
        self._arena_center[env_ids] = root_pos[:, :2]

    def step(self):
        due = self.env.progress_buf >= self._rejitter_at
        env_ids = due.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._place_opponent(env_ids)

    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        device = self.env.device
        num_envs = self.env.num_envs
        zeros = torch.zeros(num_envs, dtype=torch.bool, device=device)
        return zeros, zeros.clone()

    def populate_context(self, ctx) -> None:
        num_envs = self.env.num_envs
        opp_rot = self._opp_rot

        # Key bodies: template offsets rotated by the opponent's yaw
        k = self._template.shape[0]
        rot_exp = opp_rot.unsqueeze(1).expand(-1, k, 4).reshape(-1, 4)
        template = self._template.unsqueeze(0).expand(num_envs, -1, -1).reshape(-1, 3)
        key_pos = rotations.quat_rotate(rot_exp, template, True).reshape(
            num_envs, k, 3
        ) + self._opp_pos.unsqueeze(1)

        time_left = (
            1.0 - self.env.progress_buf.float() / max(self.env.max_episode_length, 1)
        ).clamp(0.0, 1.0)

        robot_state = self.env.simulator.get_robot_state()

        ctx.battle = BattleContext(
            opp_root_pos=self._opp_pos,
            opp_root_rot=opp_rot,
            opp_root_vel=torch.zeros_like(self._opp_pos),
            opp_root_ang_vel=torch.zeros_like(self._opp_pos),
            opp_key_body_pos=key_pos,
            opp_key_body_vel=torch.zeros_like(key_pos),
            head_pos=robot_state.rigid_body_pos[:, self._head_body_id],
            head_rot=robot_state.rigid_body_rot[:, self._head_body_id],
            opp_head_pos=key_pos[:, 0],
            health=self._ones,
            opp_health=self._ones,
            downed=self._zeros,
            opp_downed=self._zeros,
            round_time_left=time_left,
            idle_time=self._zeros,
            hit_energy_dealt=self._zeros,
            hit_energy_taken=self._zeros,
            strike_diversity_bonus=self._zeros,
            win_signal=self._zeros,
            match_ended=torch.zeros(
                num_envs, dtype=torch.bool, device=self.env.device
            ),
            arena_center=self._arena_center,
            arena_half_size=self.config.arena_size / 2.0,
        )


__all__ = ["VirtualOpponentControlConfig", "VirtualOpponentControl"]
