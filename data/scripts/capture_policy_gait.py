# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Headless policy-rollout state capture, for gait forensics.

Rolls a trained checkpoint with mean actions and dumps per-step simulator
state (root pose/vel, dof pos, body positions, and MuJoCo-checked
self-contacts are computed OFFLINE from the dump). Reuses inference_agent's
entire construction path by monkeypatching the evaluator's
simple_test_policy with a fixed-length recording loop -- 400 lines of config
plumbing stay in one place.

    python data/scripts/capture_policy_gait.py \
        --checkpoint results/utahraptor_amp_walk_v3/last.ckpt \
        --out /tmp/raptor_gait.pt --steps 600 --num-envs 16

Everything after --out/--steps is forwarded to inference_agent (which also
consumes --checkpoint). --headless is forced.
"""
from __future__ import annotations

import sys

import torch


def main() -> None:
    # pull our own args out; forward the rest
    argv = sys.argv[1:]
    def take(flag, default=None, cast=str):
        if flag in argv:
            i = argv.index(flag)
            v = cast(argv[i + 1])
            del argv[i : i + 2]
            return v
        return default

    out_path = take("--out", "/tmp/policy_gait.pt")
    n_steps = take("--steps", 600, int)

    sys.path.insert(0, ".")
    from protomotions.agents.evaluators import base_evaluator

    def recording_test_policy(self, collect_metrics: bool = False):
        self.agent.eval()
        rec = {k: [] for k in
               ("root_pos", "root_rot", "dof_pos", "rigid_body_pos", "dones")}
        # done_indices=None resets EVERYTHING -- correct only for the very
        # first call. Passing None every step (the original bug here) records
        # 600 fresh RSI spawns instead of a rollout: root positions teleport
        # ~100 m/frame and every metric is garbage. Mirror the real evaluator:
        # full reset once, then reset only the envs that finished.
        done_indices = None
        for step in range(n_steps):
            obs, _ = self.env.reset(done_indices)
            self.agent.pre_collect_step(step)
            obs = self.agent.add_agent_info_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)
            with torch.no_grad():
                model_outs = self.agent.model(obs_td)
            action = (
                model_outs["mean_action"]
                if "mean_action" in model_outs
                else model_outs["action"]
            )
            _, _, dones, _, _ = self.env.step(action)
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            state = self.env.simulator.get_robot_state()
            rec["root_pos"].append(state.rigid_body_pos[:, 0].cpu().clone())
            rec["root_rot"].append(state.rigid_body_rot[:, 0].cpu().clone())
            rec["dof_pos"].append(state.dof_pos.cpu().clone())
            rec["rigid_body_pos"].append(state.rigid_body_pos.cpu().clone())
            rec["dones"].append(dones.cpu().clone())
        packed = {k: torch.stack(v) for k, v in rec.items()}
        torch.save(packed, out_path)
        print(f"CAPTURED {n_steps} steps x {packed['dof_pos'].shape[1]} envs "
              f"-> {out_path}", flush=True)

    base_evaluator.BaseEvaluator.simple_test_policy = recording_test_policy

    sys.argv = [sys.argv[0]] + argv + ["--headless"]
    from protomotions import inference_agent
    inference_agent.main()


if __name__ == "__main__":
    main()
