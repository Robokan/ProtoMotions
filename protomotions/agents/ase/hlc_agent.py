# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Single-robot ASE high-level controller: frozen LLC + trainable latent policy.

The non-league sibling of protomotions/agents/league/ase_hlc_agent.py. The
policy's action IS a skill latent; a FROZEN low-level controller (an ASE
pretrain's actor, loaded via pretrained_modules["llc"]) decodes it into joint
actions every sim step. Use it for single-robot tasks -- steering, path
following -- where the league's self-play pairing has nothing to pair.

``decision_interval`` (IsaacLabASE's ``llc_steps``) holds one latent across k
sim steps while the LLC re-runs each step with fresh proprioception; rewards
are averaged over the window and dones OR'd.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict

import torch
from tensordict import TensorDict
from torch import Tensor

from protomotions.agents.fine_tuning.agent import FineTuningAgent
from protomotions.agents.fine_tuning.config import FineTuningAgentConfig
from protomotions.agents.ppo.agent import PPO

log = logging.getLogger(__name__)


class HLCEnvAdapter:
    """Presents a latent-action view of a single-robot env.

    The agent emits latents; this adapter runs the frozen LLC over the
    decision window so the agent never sees joint actions.
    """

    def __init__(self, env):
        self._inner = env
        self._latent_translator = None
        self._decision_interval = 1
        self._style_reward_fn = None
        self._task_reward_w = 1.0
        self._style_reward_w = 0.0
        self._last_obs = None

    def set_latent_translator(self, translator) -> None:
        """translator(obs: dict[str, Tensor], latents: Tensor) -> joint actions."""
        self._latent_translator = translator

    def set_decision_interval(self, k: int) -> None:
        self._decision_interval = max(1, int(k))

    def set_style_reward(self, fn, task_w: float, style_w: float) -> None:
        self._style_reward_fn = fn
        self._task_reward_w = float(task_w)
        self._style_reward_w = float(style_w)

    @property
    def inner_env(self):
        return self._inner

    def get_obs(self) -> Dict[str, Tensor]:
        if self._last_obs is None:
            self._last_obs = self._inner.get_obs()
        return self._last_obs

    def reset(self, env_ids=None, **kwargs):
        obs, info = self._inner.reset(env_ids, **kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, latent_action: Tensor):
        if self._latent_translator is None:
            raise RuntimeError(
                "HLCEnvAdapter.step called before set_latent_translator()"
            )
        k = self._decision_interval
        reward_sum = None
        style_obs = []
        dones_any = None
        term_any = None
        for _ in range(k):
            if self._last_obs is None:
                self._last_obs = self._inner.get_obs()
            with torch.no_grad():
                action = self._latent_translator(self._last_obs, latent_action)
            obs, rewards, dones, terminated, extras = self._inner.step(action)
            self._last_obs = obs
            reward_sum = rewards if reward_sum is None else reward_sum + rewards
            if self._style_reward_fn is not None:
                style_obs.append(obs["historical_max_coords_obs"])
            if dones_any is None:
                dones_any, term_any = dones, terminated
            else:
                dones_any = torch.logical_or(dones_any, dones).to(dones.dtype)
                term_any = torch.logical_or(term_any, terminated).to(terminated.dtype)

        total = reward_sum / k
        if style_obs:
            with torch.no_grad():
                stacked = {"historical_max_coords_obs": torch.cat(style_obs, dim=0)}
                z_rep = latent_action.repeat(k, 1)
                style_avg = (
                    self._style_reward_fn(stacked, z_rep).view(k, -1).mean(dim=0)
                )
            total = self._task_reward_w * total + self._style_reward_w * style_avg
            extras["hlc_style_reward"] = style_avg
        return obs, total, dones_any, term_any, extras

    def close(self) -> None:
        self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name in ("max_episode_length",) and "_inner" in self.__dict__:
            setattr(self._inner, name, value)
        else:
            object.__setattr__(self, name, value)


@dataclass
class HLCParams:
    """High-level controller parameters."""

    latent_dim: int = 64
    llc_deterministic: bool = True  # feed the LLC's mean action to the sim
    # One HLC decision per k sim steps. NOTE: training_max_steps counts HLC
    # decisions x envs, so sim steps = k x that budget.
    decision_interval: int = 1
    # Reward mix; style requires pretrained_modules["llc_disc"].
    task_reward_w: float = 1.0
    disc_reward_w: float = 0.0


@dataclass
class ASEHLCAgentConfig(FineTuningAgentConfig):
    """PPO over ASE skill latents with a frozen LLC (single robot)."""

    _target_: str = "protomotions.agents.ase.hlc_agent.ASEHLCAgent"

    hlc: HLCParams = field(default_factory=HLCParams)


class ASEHLCAgent(FineTuningAgent):
    """High-level latent policy trained by PPO over a frozen LLC."""

    config: ASEHLCAgentConfig

    def __init__(self, fabric, env, config, root_dir=None):
        adapter = HLCEnvAdapter(env)
        super().__init__(fabric, adapter, config, root_dir=root_dir)
        self._llc = None  # frozen PPOActor, captured in _post_create_model_hook
        self._llc_disc = None  # frozen discriminator (style anchor), optional

        adapter.set_latent_translator(self._latents_to_joint_actions)
        adapter.set_decision_interval(config.hlc.decision_interval)
        if config.hlc.disc_reward_w > 0:
            adapter.set_style_reward(
                self._style_reward,
                task_w=config.hlc.task_reward_w,
                style_w=config.hlc.disc_reward_w,
            )

    # The LLC stays OUTSIDE the model so checkpoints carry HLC weights only.
    def create_model(self):
        return PPO.create_model(self)

    def _post_create_model_hook(self) -> None:
        llc = self.pretrained.get("llc")
        if llc is None:
            raise ValueError(
                f"{type(self).__name__} requires config.pretrained_modules['llc'] "
                "(the frozen low-level controller from an ASE pretrain, "
                "module_path='actor')."
            )
        self._llc = llc  # frozen + eval via PretrainedModelConfig(freeze=True)
        self._llc_disc = self.pretrained.get("llc_disc")
        if self.config.hlc.disc_reward_w > 0 and self._llc_disc is None:
            raise ValueError(
                "hlc.disc_reward_w > 0 requires pretrained_modules['llc_disc'] "
                "(the pretrain checkpoint's discriminator, "
                "module_path='discriminator')."
            )

    def _latents_to_joint_actions(
        self, obs: Dict[str, Tensor], latents: Tensor
    ) -> Tensor:
        # Project onto the unit hypersphere -- the latent-space convention the
        # LLC was trained under (ASE.sample_latents).
        z = torch.nn.functional.normalize(latents, dim=-1)
        key = "mean_action" if self.config.hlc.llc_deterministic else "action"
        td = TensorDict(
            {"max_coords_obs": obs["max_coords_obs"], "latents": z},
            batch_size=z.shape[0],
        )
        td = self._llc(td)
        return td[key]

    def _style_reward(self, obs: Dict[str, Tensor], latents: Tensor) -> Tensor:
        """Naturalness anchor: the frozen pretrain discriminator's reward over
        the motion-history window (weights never updated in stage 2)."""
        hist = obs["historical_max_coords_obs"]
        z = torch.nn.functional.normalize(latents, dim=-1)
        td = TensorDict(
            {"historical_max_coords_obs": hist, "latents": z},
            batch_size=hist.shape[0],
        )
        td = self._llc_disc(td)
        return self._llc_disc.compute_disc_reward(td["disc_logits"]).view(-1)


__all__ = [
    "HLCEnvAdapter",
    "HLCParams",
    "ASEHLCAgentConfig",
    "ASEHLCAgent",
]
