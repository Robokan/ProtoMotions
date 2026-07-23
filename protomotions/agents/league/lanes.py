# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Opponent inference lanes: batched league inference over adapter snapshots.

League snapshots are PEFT adapters only (<1% of the model), but the base
DiscretePriorPEFT stack has no multi-adapter serving, so opponents run
through a small set of "lanes": full model replicas that share the frozen
prior's parameter storage (aliased — the base is never trained) and differ
only in the adapter tensors loaded into them. Reassignments happen when
match opponents are resampled, not every step, so adapter loads are rare;
per-step cost is one forward per *distinct live member*, each on its
env-group sub-batch.

This realizes the plan's "batched multi-LoRA" intent with a simpler
mechanism; a true per-env batched adapter gather can replace ``act()``
later without touching the callers.
"""

import logging
from typing import Callable, Dict, Optional

import torch
from torch import Tensor

from protomotions.agents.peft.utils.adapter_state import is_adapter_state_key

log = logging.getLogger(__name__)


class OpponentLanes:
    """A pool of frozen model replicas, each pinned to one league member."""

    def __init__(
        self,
        model_factory: Callable[[], torch.nn.Module],
        num_lanes: int,
        share_frozen_base_with: Optional[torch.nn.Module] = None,
        assign_fn: Optional[Callable[[torch.nn.Module, Dict[str, Tensor]], None]] = None,
    ):
        """
        Args:
            model_factory: Builds one ready-to-infer model (PEFT injected,
                eval mode). Called ``num_lanes`` times.
            num_lanes: Maximum number of distinct league members that can be
                live simultaneously. Resampling only draws members that fit.
            share_frozen_base_with: If given, every non-adapter parameter of
                each lane is re-pointed at this model's storage, so N lanes
                cost ~1x base + N x adapter memory.
            assign_fn: How to load a member's snapshot payload into a lane
                model. Default (None) is the PEFT path: load an adapter state
                dict into the lane's actor. Full-model leagues (e.g. ASE)
                pass ``lambda model, state: model.load_state_dict(state)``.
        """
        self._assign_fn = assign_fn
        self.num_lanes = num_lanes
        self.lanes = []
        self.lane_member: list = [None] * num_lanes
        self._lru: list = list(range(num_lanes))

        base_params = (
            dict(share_frozen_base_with.named_parameters())
            if share_frozen_base_with is not None
            else None
        )
        base_buffers = (
            dict(share_frozen_base_with.named_buffers())
            if share_frozen_base_with is not None
            else None
        )

        for lane_idx in range(num_lanes):
            model = model_factory()
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            if base_params is not None:
                shared = 0
                for name, param in model.named_parameters():
                    if not is_adapter_state_key(name) and name in base_params:
                        param.data = base_params[name].data
                        shared += 1
                for name, buf in model.named_buffers():
                    if not is_adapter_state_key(name) and name in base_buffers:
                        buf.data = base_buffers[name].data
                if lane_idx == 0:
                    log.info(f"Opponent lanes share {shared} frozen base tensors")
            self.lanes.append(model)

    # ------------------------------------------------------------------
    def live_members(self) -> set:
        return {m for m in self.lane_member if m is not None}

    def _lane_of(self, member_id: int) -> Optional[int]:
        try:
            return self.lane_member.index(member_id)
        except ValueError:
            return None

    def _touch(self, lane_idx: int) -> None:
        self._lru.remove(lane_idx)
        self._lru.append(lane_idx)

    def assign(
        self,
        member_id: int,
        adapter_state: Dict[str, Tensor],
        in_use: Optional[set] = None,
    ) -> int:
        """Ensure ``member_id``'s adapter is loaded in some lane; return it.

        Args:
            member_id: League member to make live.
            adapter_state: The member's adapter tensors.
            in_use: Member ids that still have envs assigned — their lanes
                must not be evicted. The caller caps distinct live members at
                ``num_lanes`` so a free lane always exists.
        """
        lane_idx = self._lane_of(member_id)
        if lane_idx is None:
            in_use = in_use or set()
            evictable = [
                idx for idx in self._lru if self.lane_member[idx] not in in_use
            ]
            if not evictable:
                raise RuntimeError(
                    f"All {self.num_lanes} opponent lanes are in use; cap the "
                    "number of distinct live league members at num_lanes"
                )
            lane_idx = evictable[0]  # least recently used among evictable
            model = self.lanes[lane_idx]
            if self._assign_fn is not None:
                self._assign_fn(model, adapter_state)
            else:
                actor = getattr(model, "_actor", model)
                actor.load_adapter_state_dict(adapter_state)
            self.lane_member[lane_idx] = member_id
        self._touch(lane_idx)
        return lane_idx

    @torch.no_grad()
    def act(
        self,
        obs_td,
        env_members: Tensor,
        action_dim: int,
    ) -> Tensor:
        """Compute actions for a batch routed per-env to member lanes.

        Args:
            obs_td: TensorDict of opponent observations [N, ...].
            env_members: Member id per env row [N] (long). Every id must
                already be assigned to a lane via :meth:`assign`.
            action_dim: Action dimensionality.

        Returns:
            Actions [N, action_dim].
        """
        num = env_members.shape[0]
        device = env_members.device
        actions = torch.zeros(num, action_dim, device=device)
        for member_id in env_members.unique().tolist():
            lane_idx = self._lane_of(int(member_id))
            if lane_idx is None:
                raise RuntimeError(
                    f"League member {member_id} has envs assigned but no lane"
                )
            rows = (env_members == member_id).nonzero(as_tuple=False).flatten()
            sub_td = obs_td[rows]
            out = self.lanes[lane_idx](sub_td)
            actions[rows] = out["action"]
        return actions


__all__ = ["OpponentLanes"]
