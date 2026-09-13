# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Record a trained policy's own motion as a reusable reference corpus.

Same monkeypatch trick as capture_policy_gait.py -- reuse inference_agent's
whole construction path rather than re-plumbing 400 lines of config -- but
records the FULL state a packaged motion library needs. capture_policy_gait
keeps only root pose + dof_pos + body positions, which is enough for gait
forensics and NOT enough to build a corpus: a motion lib also needs body
ROTATIONS (grs), body linear/angular velocities (gvs/gavs) and dof
velocities (dvs).

Velocities come straight from the simulator rather than finite-differencing
the positions afterwards. Finite differences across a reset boundary produce
a ~100 m/s teleport spike, and across the RSI hand-off they blend mocap
velocity into policy velocity.

`dones` is recorded so the segmenter downstream can cut the per-env stream at
episode boundaries instead of welding unrelated episodes into one clip.

    python data/scripts/capture_policy_motion.py \
        --checkpoint results/go2_amp_reversed/last.ckpt \
        --experiment-path examples/experiments/amp/mlp.py \
        --simulator isaaclab \
        --out /tmp/go2_reversed_raw.pt --steps 900 --num-envs 128

Everything after --out/--steps is forwarded to inference_agent (which also
consumes --checkpoint). --headless is forced.
"""
from __future__ import annotations

import sys

import torch

# recorded per step; names match the packaged-motion-lib keys they become
_FIELDS = (
    "rigid_body_pos",      # -> gts
    "rigid_body_rot",      # -> grs
    "rigid_body_vel",      # -> gvs
    "rigid_body_ang_vel",  # -> gavs
    "dof_pos",             # -> dps
    "dof_vel",             # -> dvs
)


def main() -> None:
    argv = sys.argv[1:]

    def take(flag, default=None, cast=str):
        if flag in argv:
            i = argv.index(flag)
            v = cast(argv[i + 1])
            del argv[i : i + 2]
            return v
        return default

    out_path = take("--out", "/tmp/policy_motion.pt")
    n_steps = take("--steps", 900, int)

    sys.path.insert(0, ".")
    from protomotions.agents.evaluators import base_evaluator

    def recording_test_policy(self, collect_metrics: bool = False):
        self.agent.eval()
        rec = {k: [] for k in _FIELDS + ("dones",)}
        # done_indices=None resets EVERYTHING -- correct only for the very
        # first call. Passing None every step records fresh RSI spawns rather
        # than a rollout (root teleports ~100 m/frame). Mirror the real
        # evaluator: full reset once, then reset only the envs that finished.
        done_indices = None
        missing = None
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
            if missing is None:
                # RobotState fields are all Optional; a simulator that leaves
                # velocities unpopulated would otherwise fail 900 steps later
                # with a confusing stack of Nones.
                missing = [f for f in _FIELDS if getattr(state, f, None) is None]
                if missing:
                    raise RuntimeError(
                        f"simulator did not populate {missing}; a motion lib "
                        f"cannot be built without them"
                    )
            for f in _FIELDS:
                rec[f].append(getattr(state, f).cpu().clone())
            rec["dones"].append(dones.cpu().clone())

            if step % 100 == 0:
                print(f"  step {step}/{n_steps}", flush=True)

        packed = {k: torch.stack(v) for k, v in rec.items()}
        packed["control_dt"] = torch.tensor(
            float(self.env.dt), dtype=torch.float32
        )
        torch.save(packed, out_path)
        n_env = packed["dof_pos"].shape[1]
        print(
            f"CAPTURED {n_steps} steps x {n_env} envs "
            f"(dt={float(self.env.dt):.5f}s, "
            f"{n_steps * n_env * float(self.env.dt):.0f}s of material) "
            f"-> {out_path}",
            flush=True,
        )

    base_evaluator.BaseEvaluator.simple_test_policy = recording_test_policy

    sys.argv = [sys.argv[0]] + argv + ["--headless"]
    from protomotions import inference_agent

    inference_agent.main()


if __name__ == "__main__":
    main()
