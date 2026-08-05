# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Train AMP on one clip at a time and report which motions are learnable.

The creature corpora are hand-animated, so some clips are physically
impossible and no policy can ever match them. Rather than guess, this
runs a fixed budget of AMP per clip and records how far it got. The
output is a difficulty ranking: the clips that train well are the ones
worth keeping, and -- since a physics sim produced the motion -- their
rollouts can be recorded back as feasible replacements for the animation.

It also tests a cheap prediction: scan_corpus_feasibility.py flags foot
slip, floating and teleports in seconds. If its ranking matches this
sweep's, the scan can triage the rest of the corpus without training.

SCORING USES THE PEAK, NOT THE FINAL VALUE. Observed on the walk runs:
episode length reached 299 of a 300-step cap and then fell back to ~27
as training continued. A final-value score would call that a failure.

    python data/scripts/amp_clip_sweep.py --robot raptor \
        --motion-dir data/motions/raptor_v5 --minutes 45 --gpus 0 1 \
        --clips WalkFwdLoop RunFwdFast Attack_bite
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def peak_metrics(run_dir: str):
    """Best episode length and style reward the run ever reached."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return None
    dirs = sorted(glob.glob(f"{run_dir}/lightning_logs/version_*"))
    if not dirs:
        return None
    ea = EventAccumulator(dirs[-1], size_guidance={"scalars": 200000})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    out = {}
    for key, tag in (("ep_len", "info/episode_length"),
                     ("amp_reward", "rewards/amp_rewards"),
                     ("agent_acc", "discriminator/agent_acc")):
        if tag not in tags:
            continue
        vals = [s.value for s in ea.Scalars(tag)]
        out[f"peak_{key}"] = max(vals)
        out[f"final_{key}"] = vals[-1]
        out["epochs"] = ea.Scalars(tag)[-1].step
    if "agent_acc" in tags or "discriminator/agent_acc" in tags:
        out["min_agent_acc"] = min(
            s.value for s in ea.Scalars("discriminator/agent_acc"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="raptor")
    ap.add_argument("--motion-dir", default="data/motions/raptor_v5")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip stems to run; default = every clip in the dir")
    ap.add_argument("--minutes", type=float, default=45.0,
                    help="wall-clock budget per clip")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0])
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--warm-start", default=None,
                    help="checkpoint to resume each clip from. A policy that "
                         "already walks learns a turn far faster, but the "
                         "score then measures difficulty GIVEN walking, not "
                         "difficulty from scratch.")
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--out", default="data/amp_clip_sweep.json")
    args = ap.parse_args()

    stems = args.clips or sorted(
        os.path.basename(p)[:-7]
        for p in glob.glob(f"{args.motion_dir}/*.motion")
        if not p.endswith("_M.motion")      # mirrors add no new information
    )
    print(f"sweep: {len(stems)} clips, {args.minutes:.0f} min each, "
          f"GPUs {args.gpus}", flush=True)

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))

    queue = [s for s in stems if s not in results]
    running = []          # (proc, stem, run_name, gpu, deadline, corpus)
    free = list(args.gpus)

    def launch(stem, gpu):
        corpus = f"/tmp/amp_sweep_{stem}.pt"
        single = f"/tmp/amp_sweep_{stem}_dir"
        os.makedirs(single, exist_ok=True)
        for old in glob.glob(f"{single}/*.motion"):
            os.remove(old)
        for suffix in ("", "_M"):
            src = f"{args.motion_dir}/{stem}{suffix}.motion"
            if os.path.exists(src):
                import shutil
                shutil.copy(src, single)
        subprocess.run(
            [sys.executable, "-m", "protomotions.components.motion_lib",
             "--motion-path", single, "--output-file", corpus,
             "--device", "cpu"],
            cwd=REPO, capture_output=True)
        run_name = f"{args.tag}_{stem}"
        cmd = [sys.executable, "protomotions/train_agent.py",
               "--robot-name", args.robot, "--simulator", "isaaclab",
               "--num-envs", str(args.num_envs),
               "--batch-size", str(args.batch_size),
               "--motion-file", corpus,
               "--experiment-path", "examples/experiments/amp/mlp.py",
               "--experiment-name", run_name, "--headless"]
        if args.warm_start:
            cmd += ["--checkpoint", args.warm_start]
        env = dict(os.environ,
                   CUDA_VISIBLE_DEVICES=str(gpu),
                   OMNI_KIT_ACCEPT_EULA="YES")
        log = open(f"/tmp/amp_sweep_{stem}.log", "w")
        proc = subprocess.Popen(cmd, cwd=REPO, env=env,
                                stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        print(f"  [gpu{gpu}] start {stem}", flush=True)
        return (proc, stem, run_name, gpu,
                time.time() + args.minutes * 60, corpus)

    while queue or running:
        while queue and free:
            running.append(launch(queue.pop(0), free.pop(0)))
        time.sleep(20)
        for entry in list(running):
            proc, stem, run_name, gpu, deadline, corpus = entry
            done = proc.poll() is not None
            if done or time.time() > deadline:
                if not done:
                    proc.terminate()
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                m = peak_metrics(f"{REPO}/results/{run_name}") or {}
                m["crashed"] = done and proc.returncode not in (0, -15)
                results[stem] = m
                json.dump(results, open(args.out, "w"), indent=2)
                pk = m.get("peak_ep_len", 0.0)
                print(f"  [gpu{gpu}] done  {stem:<32} peak_ep_len {pk:6.1f}  "
                      f"epochs {m.get('epochs', 0)}", flush=True)
                running.remove(entry)
                free.append(gpu)

    ranked = sorted(results.items(),
                    key=lambda kv: -(kv[1].get("peak_ep_len") or 0))
    print(f"\n{'clip':<36}{'peak ep_len':>12}{'peak amp_rew':>14}{'min acc':>9}")
    for stem, m in ranked:
        print(f"{stem:<36}{m.get('peak_ep_len', 0):12.1f}"
              f"{m.get('peak_amp_reward', 0):14.4f}"
              f"{m.get('min_agent_acc', 1):9.3f}")
    print(f"\nwritten to {args.out}")


main()
