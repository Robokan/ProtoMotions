# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Paper-faithful ASE battle league: frozen LLC + trainable high-level policy.

Two-level control exactly as in the ASE paper (Peng et al. 2022), mirroring
the GPC league's frozen-prior structure:

- **LLC (stage 1, frozen)**: the latent-conditioned low-level controller from
  an ``examples/experiments/ase/mlp.py`` pretrain. Loaded via
  ``pretrained_modules["llc"]`` (module_path="actor"), eval mode,
  requires_grad False, NO optimizer params. It lives on the agent — NOT on
  the trainable model — so league snapshots, opponent lanes, and training
  checkpoints contain only HLC weights, and a deeper/longer-trained LLC can
  be swapped under an existing HLC via ``--llc-checkpoint``.
- **HLC (stage 2, trained here)**: a small PPO actor-critic over battle task
  obs + self obs whose 64-dim continuous action IS the skill latent z. Before
  the LLC consumes z it is projected onto the unit hypersphere
  (``F.normalize``, the ``ASE.sample_latents`` convention).
- **Action flow**: each control step, HLC(obs) -> z; LLC(z, max_coords_obs)
  -> joint actions. Both halves of every match run through the same frozen
  LLC (:class:`HLCSelfPlayEnvAdapter` translates ego and opponent latents).
  PPO trains purely in latent-action space; the env adapter owns the
  translation, so the training loop and tournament code stay unchanged.
- **League**: :class:`FullModelLeagueMixin` machinery; snapshots are the
  full (small) HLC model state ("architecture": "ase_hlc"). No AMP
  discriminator or MI objective in stage 2 — style is guaranteed by the
  frozen LLC, matching both the paper and the GPC league.

Tournament note: BattleTournament's ``action_hold`` holds the HLC *latent*
between decode frames while the LLC still runs every step with fresh
proprioception — which is precisely the paper's slow-HLC/fast-LLC split.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import torch
from tensordict import TensorDict
from torch import Tensor

from protomotions.agents.fine_tuning.agent import FineTuningAgent
from protomotions.agents.fine_tuning.config import FineTuningAgentConfig
from protomotions.agents.league.agent import LeagueParams
from protomotions.agents.league.full_model_league import FullModelLeagueMixin
from protomotions.agents.league.self_play_env import SelfPlayEnvAdapter
from protomotions.agents.ppo.agent import PPO

log = logging.getLogger(__name__)


class HLCSelfPlayEnvAdapter(SelfPlayEnvAdapter):
    """Self-play adapter whose policies emit skill latents, not joint actions.

    Both the ego action passed to :meth:`step` and the opponent policy's
    output are latent vectors; a translator callback (the agent's frozen LLC)
    maps (obs, latents) -> joint actions for each half before the inner
    BattleEnv steps.

    ``decision_interval`` (the Template's ``llc_steps``) makes one agent-side
    step span k inner sim steps: both fighters' latents are held for the
    window while the LLC re-runs every inner step with fresh proprioception.
    Task (and optional style) rewards are averaged over the window and dones
    are OR'd — the exact semantics of IsaacLabASE's ``hrl_sp_agent.env_step``.
    """

    def __init__(self, env):
        super().__init__(env)
        self._latent_translator = None
        self._decision_interval = 1
        # Optional style anchor: fn(ego_obs, ego_latents) -> [N] raw rewards.
        self._style_reward_fn = None
        self._task_reward_w = 1.0
        self._style_reward_w = 0.0

    def set_latent_translator(self, translator) -> None:
        """translator(obs: dict[str, Tensor], latents: Tensor) -> joint actions."""
        self._latent_translator = translator

    def set_decision_interval(self, k: int) -> None:
        self._decision_interval = max(1, int(k))

    def set_style_reward(self, fn, task_w: float, style_w: float) -> None:
        """Blend a per-step style reward into the task stream (Template's
        task_reward_w/disc_reward_w mix, applied before reward normalization)."""
        self._style_reward_fn = fn
        self._task_reward_w = float(task_w)
        self._style_reward_w = float(style_w)

    def step(self, latent_action: Tensor):
        if self._opponent_policy is None:
            raise RuntimeError(
                "HLCSelfPlayEnvAdapter.step called before set_opponent_policy()"
            )
        if self._latent_translator is None:
            raise RuntimeError(
                "HLCSelfPlayEnvAdapter.step called before set_latent_translator()"
            )
        k = self._decision_interval
        with torch.no_grad():
            # Opponent latents chosen once per decision window, like the ego's.
            opp_latents = self._opponent_policy(self.opponent_obs())
            # Both halves share the frozen LLC: run them as ONE batched
            # forward per sim step (the full unsliced obs lives here anyway).
            full_latents = torch.cat([latent_action, opp_latents], dim=0)

        reward_sum = None
        style_obs = []  # per-inner-step ego history windows (style anchor)
        dones_any = None
        term_any = None
        for _ in range(k):
            if self._last_full_obs is None:
                self._last_full_obs = self._inner.get_obs()
            with torch.no_grad():
                full_action = self._latent_translator(
                    self._last_full_obs, full_latents
                )
            obs, rewards, dones, terminated, extras = self._step_full(full_action)
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
            # One stacked discriminator forward per decision window instead
            # of k separate ones; averaging per env is identical math.
            with torch.no_grad():
                stacked = {"historical_max_coords_obs": torch.cat(style_obs, dim=0)}
                z_rep = latent_action.repeat(k, 1)
                style_avg = (
                    self._style_reward_fn(stacked, z_rep)
                    .view(k, -1)
                    .mean(dim=0)
                )
            total = self._task_reward_w * total + self._style_reward_w * style_avg
            extras["hlc_style_reward"] = style_avg
        return obs, total, dones_any, term_any, extras


@dataclass
class HLCParams:
    """High-level controller parameters."""

    latent_dim: int = 64
    llc_deterministic: bool = True  # feed the LLC's mean action to the sim
    # One HLC decision per k sim steps (Template llc_steps=5); the LLC still
    # runs every sim step with fresh proprio. NOTE: training_max_steps counts
    # HLC decisions x envs, so sim steps = k x that budget.
    decision_interval: int = 1
    # Reward mix (Template: 0.9 task + 0.1 style from the FROZEN LLC
    # discriminator). style requires pretrained_modules["llc_disc"].
    task_reward_w: float = 1.0
    disc_reward_w: float = 0.0
    # Hot-reload the frozen LLC (+ its discriminator) between epochs whenever
    # --llc-checkpoint's file changes on disk — lets a concurrent LLC pretrain
    # on another GPU keep improving under this league (the Template's
    # two-trainer workflow). Snapshots/lanes/ckpts are HLC-only, so a reload
    # touches nothing else.
    llc_hot_reload: bool = True
    # Minimum seconds between reloads (Eric, 2026-07-30: every 10 minutes).
    # The v1 run reloaded every ~58s — the ground moved under ego AND pool
    # every ~2 epochs and the PFSP gate stats were wiped as fast as they
    # accumulated; the 80% gate never fired in 595 epochs.
    llc_reload_min_seconds: float = 600.0


@dataclass
class LeagueASEHLCAgentConfig(FineTuningAgentConfig):
    """PPO-over-latents agent config with a self-play league around it."""

    _target_: str = "protomotions.agents.league.ase_hlc_agent.LeagueASEHLCAgent"
    # ``model`` stays the parent's BaseModelConfig field; experiments must
    # pass a PPOModelConfig (PPOActorConfig has no default mu_key, so it
    # cannot serve as a dataclass default factory).
    league: LeagueParams = field(default_factory=LeagueParams)
    hlc: HLCParams = field(default_factory=HLCParams)
    # Cross-morphology (MULTI_ROBOT_LEAGUE_PLAN Phase 3): when the opponent
    # block hosts a different robot, these carry its identity. The opponent's
    # frozen LLC arrives via pretrained_modules["opp_llc"]; opponent HLC
    # snapshots come from the shared pool (published by that robot's league).
    opponent_robot_name: str = None
    opponent_robot_config: object = None


class LeagueASEHLCAgent(FullModelLeagueMixin, FineTuningAgent):
    """High-level latent policy trained by PPO self-play over a frozen LLC."""

    config: LeagueASEHLCAgentConfig

    snapshot_architecture = "ase_hlc"

    def __init__(self, fabric, env, config, root_dir=None):
        adapter = HLCSelfPlayEnvAdapter(env)
        super().__init__(fabric, adapter, config, root_dir=root_dir)
        self._init_league(adapter, config.league, root_dir)
        self._llc = None  # frozen PPOActor, captured in _post_create_model_hook
        self._llc_disc = None  # frozen discriminator (style anchor), optional
        self._llc_ckpt_mtime = None
        self._llc_reloads = 0
        self._last_llc_reload_time = time.time()
        # Cross-morphology opponent side (Phase 3)
        self._opp_rc = getattr(config, "opponent_robot_config", None)
        self._opp_llc = None  # the opponent robot's frozen LLC
        self._opp_llc_ckpt_mtime = None
        self._opp_model_proto = None  # materialized opponent-HLC template
        self._opp_fingerprint = None

        adapter.set_latent_translator(self._latents_to_joint_actions)
        adapter.set_decision_interval(config.hlc.decision_interval)
        if config.hlc.disc_reward_w > 0:
            adapter.set_style_reward(
                self._style_reward,
                task_w=config.hlc.task_reward_w,
                style_w=config.hlc.disc_reward_w,
            )
        adapter.set_opponent_policy(self._opponent_policy)
        adapter.set_match_end_callback(self._on_matches_ended)

    # ------------------------------------------------------------------
    # Model: plain PPO actor-critic (the HLC). The LLC stays OUTSIDE the
    # model so state_dict/snapshots/lanes carry HLC weights only.
    # ------------------------------------------------------------------
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
        self._opp_llc = self.pretrained.get("opp_llc")
        if self._opp_rc is not None and self._opp_llc is None:
            raise ValueError(
                "opponent_robot_config is set (cross-morphology league) but "
                "pretrained_modules['opp_llc'] is missing — the opponent "
                "block needs ITS robot's frozen LLC "
                "(--opponent-llc-checkpoint)."
            )
        self._opp_llc_ckpt_mtime = self._module_ckpt_mtime("opp_llc")
        if self.config.hlc.disc_reward_w > 0 and self._llc_disc is None:
            raise ValueError(
                "hlc.disc_reward_w > 0 requires pretrained_modules['llc_disc'] "
                "(the pretrain checkpoint's discriminator, "
                "module_path='discriminator')."
            )
        self._llc_ckpt_mtime = self._llc_checkpoint_mtime()

    def _snapshot_extra_meta(self) -> dict:
        """Pin the frozen LLC this HLC snapshot executes under (Phase 2b:
        an ase_hlc bundle is HLC weights + an LLC reference; with hot-reload
        the LLC under a snapshot can drift, so record path AND mtime)."""
        cfg = self.config.pretrained_modules.get("llc")
        return {
            "llc_checkpoint": getattr(cfg, "checkpoint_path", None),
            "llc_checkpoint_mtime": self._llc_ckpt_mtime,
            "llc_reloads": self._llc_reloads,
            "latent_dim": getattr(self.config.hlc, "latent_dim", 64),
            "decision_interval": getattr(self.config.hlc, "decision_interval", 1),
        }

    # ------------------------------------------------------------------
    # LLC hot-reload: pick up new pretrain checkpoints between epochs
    # ------------------------------------------------------------------
    def _module_ckpt_mtime(self, name: str):
        cfg = self.config.pretrained_modules.get(name)
        if cfg is None or not cfg.checkpoint_path:
            return None
        try:
            return Path(cfg.checkpoint_path).stat().st_mtime
        except OSError:
            return None

    def _llc_checkpoint_mtime(self):
        return self._module_ckpt_mtime("llc")

    def _maybe_reload_llc(self) -> None:
        # getattr: frozen configs from before these fields existed get the
        # current defaults.
        if not getattr(self.config.hlc, "llc_hot_reload", True):
            return
        min_gap = getattr(self.config.hlc, "llc_reload_min_seconds", 600.0)
        if time.time() - self._last_llc_reload_time < min_gap:
            return
        mtime = self._llc_checkpoint_mtime()
        opp_mtime = self._module_ckpt_mtime("opp_llc")
        llc_new = mtime is not None and (
            self._llc_ckpt_mtime is None or mtime > self._llc_ckpt_mtime
        )
        opp_new = opp_mtime is not None and (
            self._opp_llc_ckpt_mtime is None
            or opp_mtime > self._opp_llc_ckpt_mtime
        )
        if not (llc_new or opp_new):
            return
        # The pretrain may still be mid-save (torch.save is not atomic);
        # wait until the file has been quiet for a few seconds.
        newest = max(m for m in (mtime, opp_mtime) if m is not None)
        if time.time() - newest < 5.0:
            return  # re-checked next epoch
        try:
            modules = self._load_pretrained_modules()
        except Exception as exc:  # noqa: BLE001 - keep training on a bad read
            log.warning("LLC hot-reload failed (%s); keeping current LLC", exc)
            return
        self._llc = modules["llc"]
        if "llc_disc" in modules:
            self._llc_disc = modules["llc_disc"]
        if "opp_llc" in modules:
            self._opp_llc = modules["opp_llc"]
            self._opp_llc_ckpt_mtime = opp_mtime
        self._llc_ckpt_mtime = mtime
        self._llc_reloads += 1
        self._last_llc_reload_time = time.time()
        # Pool members' HLCs now run over a different LLC than they were
        # trained (and measured) against: their PFSP win-rate stats are
        # stale, so re-measure. Ratings are kept — Elo re-converges on its
        # own, and both sides of every match shifted together.
        if self.pool.members:
            self.pool.reset_stats()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        msg = (
            f"[HLC] LLC checkpoint changed on disk — hot-reloaded frozen LLC "
            f"(#{self._llc_reloads}) from "
            f"{self.config.pretrained_modules['llc'].checkpoint_path} "
            f"(epoch {self.current_epoch})"
        )
        # print() as well: Kit hijacks logging into /tmp/isaaclab/logs, and
        # this must be visible on the training console.
        print(msg, flush=True)
        log.info(msg)

    def post_epoch_logging(self, training_log_dict: dict):
        self._maybe_reload_llc()
        training_log_dict["hlc/llc_reloads"] = float(self._llc_reloads)
        super().post_epoch_logging(training_log_dict)

    # ------------------------------------------------------------------
    # Latent -> joint action translation through the frozen LLC
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _latents_to_joint_actions(
        self, obs: Dict[str, Tensor], latents: Tensor
    ) -> Tensor:
        # Project onto the unit hypersphere — the latent-space convention the
        # LLC was trained under (ASE.sample_latents).
        z = torch.nn.functional.normalize(latents, dim=-1)
        key = "mean_action" if self.config.hlc.llc_deterministic else "action"
        if self._opp_rc is None:
            td = TensorDict(
                {"max_coords_obs": obs["max_coords_obs"], "latents": z},
                batch_size=z.shape[0],
            )
            td = self._llc(td)
            return td[key]
        # Cross-morphology: each block's latents decode through ITS robot's
        # frozen LLC over ITS robot's self-observation. The env-computed
        # max_coords_obs is ego-robot-shaped, so the opponent block's is
        # recomputed from raw sim state with the opponent's body count.
        n = self.env.num_matches
        td_a = TensorDict(
            {"max_coords_obs": obs["max_coords_obs"][:n], "latents": z[:n]},
            batch_size=n,
        )
        act_a = self._llc(td_a)[key]
        td_b = TensorDict(
            {"max_coords_obs": self._opp_max_coords(), "latents": z[n:]},
            batch_size=n,
        )
        act_b = self._opp_llc(td_b)[key]
        # Assemble the padded [2N, max_dofs] action tensor the multi-robot
        # simulator splits per block (extra columns are masked there).
        width = max(act_a.shape[1], act_b.shape[1])
        full = torch.zeros(2 * n, width, device=act_a.device, dtype=act_a.dtype)
        full[:n, : act_a.shape[1]] = act_a
        full[n:, : act_b.shape[1]] = act_b
        return full

    # ------------------------------------------------------------------
    # Cross-morphology opponent side (Phase 3)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _opp_max_coords(self) -> Tensor:
        """The opponent block's self-observation, computed with ITS body
        count from raw (padded) sim state. Settings MUST match the opponent
        LLC's pretrain (ase/mlp.py: local_obs, root_height, no contacts) —
        the same contract the ego max_coords_obs component documents."""
        from protomotions.envs.obs import (
            compute_humanoid_max_coords_observations,
        )

        n = self.env.num_matches
        nb = self._opp_rc.kinematic_info.num_bodies
        state = self.env.simulator.get_robot_state()
        rows = slice(n, 2 * n)
        body_pos = state.rigid_body_pos[rows, :nb]
        ground = self.env.terrain.get_ground_heights(body_pos[:, 0]).view(-1)
        return compute_humanoid_max_coords_observations(
            body_pos=body_pos,
            body_rot=state.rigid_body_rot[rows, :nb],
            body_vel=state.rigid_body_vel[rows, :nb],
            body_ang_vel=state.rigid_body_ang_vel[rows, :nb],
            ground_height=ground,
            body_contacts=torch.zeros_like(body_pos[..., 0]),
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
            w_last=True,
        )

    def _opponent_obs_td(self, opp_obs: Dict[str, Tensor]):
        if self._opp_rc is not None:
            # Opponent HLC snapshots were trained on THEIR robot's obs:
            # swap in the opponent-shaped self-observation (task_obs is
            # shape-stable league-wide and already per-side via the battle
            # tables).
            opp_obs = dict(opp_obs)
            opp_obs["max_coords_obs"] = self._opp_max_coords()
        return super()._opponent_obs_td(opp_obs)

    def _opp_proto_model(self):
        """Materialized opponent-HLC template (built once, deep-copied per
        lane). The HLC architecture is shared league-wide; only the input
        widths differ, so the ego model CONFIG materializes the opponent
        model when forwarded with opponent-shaped observations."""
        if self._opp_model_proto is None:
            from protomotions.utils.hydra_replacement import get_class

            model_cls = get_class(self.config.model._target_)
            proto = model_cls(config=self.config.model).to(self.device)
            mc = self._opp_max_coords()
            task = self.env.opponent_obs()["task_obs"]
            td = TensorDict(
                {"max_coords_obs": mc, "task_obs": task},
                batch_size=mc.shape[0],
            ).to(self.device)
            with torch.no_grad():
                proto(td)  # materialize lazy modules at opponent widths
            proto.eval()
            for p_ in proto.parameters():
                p_.requires_grad_(False)
            self._opp_model_proto = proto
            from protomotions.agents.league import pool_io

            self._opp_fingerprint = pool_io.state_fingerprint(
                proto.state_dict()
            )
        return self._opp_model_proto

    def _opponent_policy(self, opp_obs: Dict[str, Tensor]) -> Tensor:
        if self._opp_rc is None:
            return super()._opponent_policy(opp_obs)
        self._ensure_league_initialized()
        if self.pool.members and not self.league_cfg.force_symmetric_inference:
            return super()._opponent_policy(opp_obs)
        # Empty pool (the opponent robot's league hasn't published yet) or
        # forced-symmetric debug: the EGO HLC picks the opponent's latents
        # from the EGO-shaped obs slice — mechanical sparring through the
        # opponent body's LLC, the rung-2 fallback. The ego model cannot
        # consume opponent-shaped obs, so skip the cross-morph obs swap.
        self._pre_opponent_policy()
        obs_td = FullModelLeagueMixin._opponent_obs_td(self, opp_obs)
        with torch.no_grad():
            out = self.model(obs_td)
        return out["action"]

    # ---- League mixin overrides (cross-morphology pool identity) -------
    def _pool_identity(self) -> dict:
        if self._opp_rc is None:
            return super()._pool_identity()
        self._opp_proto_model()
        return {
            "robot": self.config.opponent_robot_name,
            "architecture": self.snapshot_architecture,
            "fingerprint": self._opp_fingerprint,
        }

    def _host_own_snapshots(self) -> bool:
        return self._opp_rc is None

    def _build_lanes(self) -> None:
        if self._opp_rc is None:
            return super()._build_lanes()
        import copy as _copy

        from protomotions.agents.league.lanes import OpponentLanes

        proto = self._opp_proto_model()

        def factory():
            return _copy.deepcopy(proto)

        def assign_full_state(model, state) -> None:
            model.load_state_dict(state, strict=True)

        self._lanes = OpponentLanes(
            model_factory=factory,
            num_lanes=self.league_cfg.num_lanes,
            share_frozen_base_with=None,
            assign_fn=assign_full_state,
        )

    @torch.no_grad()
    def _style_reward(self, obs: Dict[str, Tensor], latents: Tensor) -> Tensor:
        """Naturalness anchor: -log(1 - D) from the FROZEN pretrain
        discriminator over the ego half's motion-history window (the
        Template's disc_reward path, weights never updated in stage 2)."""
        hist = obs["historical_max_coords_obs"]
        z = torch.nn.functional.normalize(latents, dim=-1)
        td = TensorDict(
            {"historical_max_coords_obs": hist, "latents": z},
            batch_size=hist.shape[0],
        )
        td = self._llc_disc(td)
        return self._llc_disc.compute_disc_reward(td["disc_logits"]).view(-1)

    # ------------------------------------------------------------------
    # League mixin hooks
    # ------------------------------------------------------------------
    def _opponent_action_dim(self) -> int:
        # Lanes serve HLC replicas: their "action" output is the latent.
        return self.config.hlc.latent_dim


__all__ = [
    "HLCSelfPlayEnvAdapter",
    "HLCParams",
    "LeagueASEHLCAgentConfig",
    "LeagueASEHLCAgent",
]
