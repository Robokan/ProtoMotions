# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Ego-half environment adapter for league self-play.

Hosts self-play on a framework that assumes one policy (the pattern from
IsaacLabASE's ``hrl_sp_agent.env_step``): the training agent sees only the
ego half of a :class:`BattleEnv` (matches ``0..N-1``), while this adapter
computes opponent actions through a policy callback and splices them into the
full ``2N`` action tensor of every step.

The agent-facing surface is a normal env: ``num_envs == N``,
``reset(env_ids)``/``step(action)`` take and return ego-half tensors, and any
attribute not defined here passes through to the wrapped env. Match-end
events (per-ego win/lose/draw) are reported through a callback so the league
agent can update PFSP statistics and resample opponents.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor


class SelfPlayEnvAdapter:
    """Presents the ego half of a paired BattleEnv as a standalone env."""

    def __init__(self, env):
        if env.num_envs % 2 != 0:
            raise ValueError("SelfPlayEnvAdapter requires an even inner env count")
        self._inner = env
        self.num_matches = env.num_envs // 2

        # Wired by the league agent before training starts.
        # opponent_policy(opp_obs: dict[str, Tensor]) -> Tensor [N, action_dim]
        self._opponent_policy: Optional[Callable[[Dict[str, Tensor]], Tensor]] = None
        # on_matches_ended(ego_ids, win, lose, draw) — all ego-half tensors
        self._on_matches_ended: Optional[
            Callable[[Tensor, Tensor, Tensor, Tensor], None]
        ] = None

        self._last_full_obs: Optional[Dict[str, Tensor]] = None

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def set_opponent_policy(self, policy: Callable) -> None:
        self._opponent_policy = policy

    def set_match_end_callback(self, callback: Callable) -> None:
        self._on_matches_ended = callback

    @property
    def inner_env(self):
        return self._inner

    # ------------------------------------------------------------------
    # Slicing helpers
    # ------------------------------------------------------------------
    def _ego(self, tensor: Tensor) -> Tensor:
        return tensor[: self.num_matches]

    def _opp(self, tensor: Tensor) -> Tensor:
        return tensor[self.num_matches :]

    def _slice_dict(self, d: Dict[str, Tensor], ego: bool = True) -> Dict[str, Tensor]:
        out = {}
        for key, value in d.items():
            if isinstance(value, Tensor) and value.shape[:1] == (
                self._inner.num_envs,
            ):
                out[key] = self._ego(value) if ego else self._opp(value)
            else:
                out[key] = value
        return out

    def opponent_obs(self) -> Dict[str, Tensor]:
        if self._last_full_obs is None:
            self._last_full_obs = self._inner.get_obs()
        return self._slice_dict(self._last_full_obs, ego=False)

    # ------------------------------------------------------------------
    # Env API (ego-half view)
    # ------------------------------------------------------------------
    @property
    def num_envs(self) -> int:
        return self.num_matches

    def get_obs(self) -> Dict[str, Tensor]:
        if self._last_full_obs is None:
            self._last_full_obs = self._inner.get_obs()
        return self._slice_dict(self._last_full_obs, ego=True)

    def reset(self, env_ids=None, **kwargs) -> Tuple[Dict[str, Tensor], dict]:
        # Ego env ids are valid inner ids; BattleEnv expands to match pairs.
        obs, info = self._inner.reset(env_ids, **kwargs)
        self._last_full_obs = obs
        return self._slice_dict(obs, ego=True), info

    def step(self, action: Tensor):
        if self._opponent_policy is None:
            raise RuntimeError(
                "SelfPlayEnvAdapter.step called before set_opponent_policy()"
            )
        with torch.no_grad():
            opp_action = self._opponent_policy(self.opponent_obs())
        full_action = torch.cat([action, opp_action], dim=0)

        obs, rewards, dones, terminated, extras = self._inner.step(full_action)
        self._last_full_obs = obs

        ego_dones = self._ego(dones)
        ego_terminated = self._ego(terminated)
        ego_rewards = self._ego(rewards)
        ego_extras = self._slice_dict(extras, ego=True)

        if self._on_matches_ended is not None and "battle/match_ended" in extras:
            ended = self._ego(extras["battle/match_ended"])
            ego_ids = ended.nonzero(as_tuple=False).flatten()
            if len(ego_ids) > 0:
                self._on_matches_ended(
                    ego_ids,
                    self._ego(extras["battle/win"])[ego_ids],
                    self._ego(extras["battle/lose"])[ego_ids],
                    self._ego(extras["battle/draw"])[ego_ids],
                )

        return (
            self._slice_dict(obs, ego=True),
            ego_rewards,
            ego_dones,
            ego_terminated,
            ego_extras,
        )

    def close(self) -> None:
        self._inner.close()

    # Everything else (motion_lib, motion_manager, dt, robot_config,
    # max_episode_length, save/restore, on_epoch_end, ...) passes through.
    def __getattr__(self, name):
        return getattr(self._inner, name)

    # max_episode_length is assigned by curriculum managers — forward the set.
    def __setattr__(self, name, value):
        if name in ("max_episode_length",) and "_inner" in self.__dict__:
            setattr(self._inner, name, value)
        else:
            object.__setattr__(self, name, value)


__all__ = ["SelfPlayEnvAdapter"]
