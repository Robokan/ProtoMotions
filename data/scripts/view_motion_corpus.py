# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Play a packaged motion library in the viewer, one clip per env.

This is KINEMATIC replay: every frame the robots' root pose and joint state
are written straight from the motion library and nothing is simulated. So what
you see is the corpus itself, not a policy's attempt at it -- which is the
point when you want to judge reference data before training on it.

inference_agent has no replay mode (it always drives a policy), so this reuses
its construction path the same way capture_policy_motion.py does -- monkeypatch
the evaluator -- and then ignores the policy entirely. A --checkpoint is still
required because that is how inference_agent finds the robot and sim config;
any checkpoint for the right robot will do.

Envs are laid out on a grid and each plays its own clip on loop, so N clips are
visible side by side. Positions written to the sim are world coordinates
(IsaacLab's write_root_state_to_sim is world-frame) and the library's clips are
segment-local, so the grid offset is added here.

    python data/scripts/view_motion_corpus.py \
        --checkpoint results/go2_amp_reversed/last.ckpt \
        --motion-file data/motions/go2/go2_reversed_mocap.pt \
        --simulator isaaclab --num-envs 9 --spacing 2.5

    # page through a 63-clip corpus 9 at a time
    ... --num-envs 9 --clip-offset 9

Windowed Isaac only runs on GPU 0 -- prefix CUDA_VISIBLE_DEVICES=0.
"""
from __future__ import annotations

import math
import sys

import torch


def main() -> None:
    argv = sys.argv[1:]

    def take(flag, default=None, cast=str):
        if flag in argv:
            i = argv.index(flag)
            v = cast(argv[i + 1])
            del argv[i : i + 2]
            return v
        return default

    spacing = take("--spacing", 2.5, float)
    clip_offset = take("--clip-offset", 0, int)
    speed = take("--speed", 1.0, float)
    n_loops = take("--loops", 0, int)  # 0 = forever

    sys.path.insert(0, ".")
    from protomotions.agents.evaluators import base_evaluator
    from protomotions.simulator.base_simulator.simulator_state import ResetState

    def replay_policy(self, collect_metrics: bool = False):
        env = self.env
        ml = env.motion_lib
        n_motions = int(ml.num_motions())
        n_env = env.num_envs
        device = env.device
        dt = float(env.dt) * speed

        env_ids = torch.arange(n_env, device=device)
        motion_ids = (
            torch.arange(n_env, device=device) + clip_offset
        ) % n_motions
        lengths = ml.get_motion_length(motion_ids)

        # grid layout, world frame; z=0 because the clips carry their own height
        side = int(math.ceil(math.sqrt(n_env)))
        gx = (env_ids % side).float() - (side - 1) / 2.0
        gy = torch.div(env_ids, side, rounding_mode="floor").float() - (
            side - 1
        ) / 2.0
        offset = torch.zeros((n_env, 3), device=device)
        offset[:, 0] = gx * spacing
        offset[:, 1] = gy * spacing

        print(f"\nreplaying {n_motions} clips, {n_env} at a time "
              f"(offset {clip_offset}), {side}x{side} grid @ {spacing}m")
        for e in range(min(n_env, 12)):
            mid = int(motion_ids[e])
            name = str(ml.motion_files[mid]).split("/")[-1] \
                if hasattr(ml, "motion_files") else f"motion {mid}"
            print(f"  env{e:02d} <- clip {mid:3d}  {float(lengths[e]):5.2f}s  {name}")
        print(flush=True)

        times = torch.zeros(n_env, device=device)
        loops = torch.zeros(n_env, device=device)
        while True:
            ref = ml.get_motion_state(motion_ids, times)
            new_states = ResetState.from_robot_state(ref)
            new_states.root_pos = new_states.root_pos + offset
            env.simulator.reset_envs(new_states, None, env_ids)
            env.simulator.render()

            times = times + dt
            wrapped = times >= lengths
            if wrapped.any():
                loops = loops + wrapped.float()
                times = torch.where(wrapped, torch.zeros_like(times), times)
                if n_loops and int(loops.min()) >= n_loops:
                    print(f"completed {n_loops} loop(s)", flush=True)
                    return

    base_evaluator.BaseEvaluator.simple_test_policy = replay_policy

    sys.argv = [sys.argv[0]] + argv
    from protomotions import inference_agent

    inference_agent.main()


if __name__ == "__main__":
    main()
