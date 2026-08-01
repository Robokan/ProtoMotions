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


class FamilyLanesRouter:
    """Routes opponent-lane assignment/inference across robot families.

    Own-family members run ego-model replicas on ego-shaped observations;
    foreign-family members run foreign-width replicas fed the foreign
    self-obs key. Presents the OpponentLanes interface the league mixin
    already talks to."""

    def __init__(self, own, foreign, is_foreign, foreign_obs_key):
        self.own = own
        self.foreign = foreign
        self._is_foreign_fn = is_foreign
        self._cache: Dict[int, bool] = {}
        self.foreign_obs_key = foreign_obs_key

    def _is_foreign(self, member_id: int) -> bool:
        member_id = int(member_id)
        if member_id not in self._cache:
            self._cache[member_id] = bool(self._is_foreign_fn(member_id))
        return self._cache[member_id]

    @property
    def num_lanes(self) -> int:
        return self.own.num_lanes + self.foreign.num_lanes

    def assign(self, member_id, payload, in_use=None):
        foreign = self._is_foreign(member_id)
        lanes = self.foreign if foreign else self.own
        if in_use is not None:
            in_use = {m for m in in_use if self._is_foreign(m) == foreign}
        lanes.assign(member_id, payload, in_use=in_use)

    @torch.no_grad()
    def act(self, obs_td, env_member, action_dim: int):
        members = env_member.tolist()
        foreign_mask = torch.tensor(
            [m >= 0 and self._is_foreign(m) for m in members],
            dtype=torch.bool, device=env_member.device,
        )
        out = None
        own_rows = (~foreign_mask).nonzero(as_tuple=False).flatten()
        if own_rows.numel() > 0:
            acts = self.own.act(obs_td[own_rows], env_member[own_rows], action_dim)
            out = torch.zeros(
                env_member.shape[0], action_dim,
                device=acts.device, dtype=acts.dtype,
            )
            out[own_rows] = acts
        f_rows = foreign_mask.nonzero(as_tuple=False).flatten()
        if f_rows.numel() > 0:
            td = obs_td[f_rows]
            td["max_coords_obs"] = td[self.foreign_obs_key]
            acts = self.foreign.act(td, env_member[f_rows], action_dim)
            if out is None:
                out = torch.zeros(
                    env_member.shape[0], action_dim,
                    device=acts.device, dtype=acts.dtype,
                )
            out[f_rows] = acts
        return out


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
            # No --opponent-llc-checkpoint: the LLC resolves from the first
            # ingested pool snapshot's provenance pin (every ase_hlc snapshot
            # records the LLC checkpoint it executes under). Until one
            # arrives the opponent block holds its default pose.
            print(
                "[HLC] mixed-morphology arenas: the foreign body's LLC "
                "resolves from the shared pool's snapshot provenance when "
                "its first snapshot arrives; until then every match is "
                "ego-family self-play.",
                flush=True,
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
        # Mixed-morphology arenas (rung 4): rows hosting the EGO family
        # (the whole ego half + own-family opponents) decode through the ego
        # LLC on the env-computed obs; rows hosting the foreign body decode
        # through ITS LLC on foreign-shaped obs recomputed from sim state.
        active_b = self.env.simulator.get_active_opponent_mask()
        rows_a = (~active_b).nonzero(as_tuple=False).flatten()
        rows_b = active_b.nonzero(as_tuple=False).flatten()
        width = max(
            self.env.robot_config.number_of_actions,
            self._opp_rc.number_of_actions,
        )
        full = torch.zeros(active_b.shape[0], width, device=z.device, dtype=z.dtype)
        td_a = TensorDict(
            {"max_coords_obs": obs["max_coords_obs"][rows_a], "latents": z[rows_a]},
            batch_size=rows_a.shape[0],
        )
        act_a = self._llc(td_a)[key]
        full[rows_a.unsqueeze(-1), torch.arange(act_a.shape[1], device=z.device)] = act_a
        if rows_b.numel() > 0:
            if self._opp_llc is None:
                pass  # default-pose hold until the foreign LLC resolves
            else:
                td_b = TensorDict(
                    {
                        "max_coords_obs": self._opp_max_coords(rows_b),
                        "latents": z[rows_b],
                    },
                    batch_size=rows_b.shape[0],
                )
                act_b = self._opp_llc(td_b)[key]
                full[
                    rows_b.unsqueeze(-1),
                    torch.arange(act_b.shape[1], device=z.device),
                ] = act_b
        return full

    # ------------------------------------------------------------------
    # Cross-morphology opponent side (Phase 3)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _opp_max_coords(self, rows: Tensor = None) -> Tensor:
        """The opponent block's self-observation, computed with ITS body
        count from raw (padded) sim state. Settings MUST match the opponent
        LLC's pretrain (ase/mlp.py: local_obs, root_height, no contacts) —
        the same contract the ego max_coords_obs component documents."""
        from protomotions.envs.obs import (
            compute_humanoid_max_coords_observations,
        )

        nb = self._opp_rc.kinematic_info.num_bodies
        state = self.env.simulator.get_robot_state()
        if rows is None:
            n = self.env.num_matches
            rows = torch.arange(n, 2 * n, device=self.device)
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
        td = super()._opponent_obs_td(opp_obs)
        if self._opp_rc is not None:
            # Own-family opponents read the ego-shaped max_coords already in
            # the td; foreign-family lanes swap in this foreign-shaped
            # self-obs (aligned by match index; rows hosting the ego family
            # carry junk there and are never routed to foreign lanes).
            td["max_coords_obs_foreign"] = self._opp_max_coords()
        return td

    def _on_snapshot_ingested(self, meta: dict) -> None:
        if self._opp_rc is None or self._opp_llc is not None:
            return
        if meta.get("robot") != self.config.opponent_robot_name:
            return  # own-family snapshots pin the EGO LLC, not the foreign one
        llc_path = meta.get("llc_checkpoint")
        if not llc_path or not Path(llc_path).exists():
            log.warning(
                "Ingested snapshot pins no loadable LLC (%s); opponent "
                "block stays idle", llc_path,
            )
            return
        from protomotions.agents.common.config import PretrainedModelConfig

        self.config.pretrained_modules["opp_llc"] = PretrainedModelConfig(
            checkpoint_path=llc_path, module_path="actor",
        )
        modules = self._load_pretrained_modules()
        self._opp_llc = modules["opp_llc"]
        self._opp_llc_ckpt_mtime = self._module_ckpt_mtime("opp_llc")
        msg = (
            f"[HLC] opponent LLC resolved from pool snapshot provenance: "
            f"{llc_path}"
        )
        print(msg, flush=True)
        log.info(msg)

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

    # ---- League mixin overrides (mixed-morphology pool, rung 4) --------
    # A league always trains ITS robot and self-plays into the pool; when
    # the arena declares a foreign body (--arena-bodies), snapshots of that
    # robot are ALSO hosted, mixed per match. Own snapshots seed the pool
    # like the classic league, so training never needs a foreign publisher.
    def _pool_identities(self) -> list:
        identities = [super()._pool_identity()]
        if self._opp_rc is not None:
            self._opp_proto_model()
            identities.append({
                "robot": self.config.opponent_robot_name,
                "architecture": self.snapshot_architecture,
                "fingerprint": self._opp_fingerprint,
            })
        return identities

    def _member_family_is_foreign(self, member_id: int) -> bool:
        member = self.pool.members.get(int(member_id))
        if member is None or not member.family:
            return False
        robot = member.family.split("/")[0]
        return robot == self.config.opponent_robot_name and robot != (
            getattr(self.league_cfg, "robot_name", None) or ""
        )

    def _build_lanes(self) -> None:
        super()._build_lanes()  # own-family lanes (ego model replicas)
        if self._opp_rc is None:
            return
        import copy as _copy

        from protomotions.agents.league.lanes import OpponentLanes

        proto = self._opp_proto_model()

        def factory():
            return _copy.deepcopy(proto)

        def assign_full_state(model, state) -> None:
            model.load_state_dict(state, strict=True)

        foreign = OpponentLanes(
            model_factory=factory,
            num_lanes=self.league_cfg.num_lanes,
            share_frozen_base_with=None,
            assign_fn=assign_full_state,
        )
        self._lanes = FamilyLanesRouter(
            own=self._lanes,
            foreign=foreign,
            is_foreign=self._member_family_is_foreign,
            foreign_obs_key="max_coords_obs_foreign",
        )

    def _on_opponents_resampled(self, ego_ids: Tensor) -> None:
        super()._on_opponents_resampled(ego_ids)
        if self._opp_rc is None or self._lanes is None:
            return
        # Sync each resampled match's arena morphology with its sampled
        # opponent's family; the match resets right after, landing the new
        # states on the newly-active twin.
        members = self.env_member[ego_ids]
        use_b = torch.tensor(
            [self._member_family_is_foreign(int(m)) for m in members],
            dtype=torch.bool, device=self.device,
        )
        sim = self.env.simulator
        n = self.env.num_matches
        sim.set_active_morphology(ego_ids.to(self.device) + n, use_b)
        self.env.inner_env.battle_control.set_opponent_family(
            ego_ids.to(self.device), use_b.long()
        )
        self.env.inner_env.refresh_default_reset_state()

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
