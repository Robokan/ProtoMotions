#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Turn a recorded chase into a LeRobot dataset, using LeRobot's own writer.

The recorder runs inside Isaac, where lerobot cannot be installed (it would
drag its own torch in beside Isaac Sim's). So it writes a plain staging
layout -- one parquet and one mp4 per episode -- and this script, run in the
lerobot environment, hands those frames to ``LeRobotDataset`` and lets it
produce the real thing.

Doing it that way rather than emitting the format by hand is deliberate.
LeRobot v3.0 concatenates many episodes into shared parquet and mp4 files and
addresses each one by row range and timestamp range; getting that subtly
wrong yields a dataset that loads happily and serves the wrong frames. Their
writer already does it, keeps the statistics consistent, and will keep doing
so when the format moves again -- v3.0 is what lerobot 0.6.2 reads, and it
refuses v2.1 outright.

Run it with the lerobot venv, not the Isaac one:

    ~/sparkpack/lerobot/.venv/bin/python scripts/chase_to_lerobot.py \\
        output/datasets/go2_chase --repo-id evaughan/go2_chase

Then train the ResNet on it without writing a training script -- ACT is a
resnet18 backbone feeding a small transformer that emits an action chunk:

    lerobot-train --dataset.repo_id=evaughan/go2_chase \\
        --policy.type=act --policy.vision_backbone=resnet18 \\
        --policy.chunk_size=20 --policy.n_action_steps=10

(ACT's default chunk_size of 100 is 10 s at 10 Hz, far past where a chase
target is still meaningful; 20 is two seconds.)

On this box torchcodec fails to load its shared library on aarch64 and
lerobot falls back to PyAV, which works and is slower. Harmless.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def read_video(path: Path) -> np.ndarray:
    """Decode an mp4 to [N, H, W, 3] uint8 with PyAV, which lerobot ships.

    imageio is not in the lerobot environment; PyAV is, because lerobot
    decodes its own videos with it.
    """
    import av  # noqa: PLC0415

    with av.open(str(path)) as container:
        return np.stack(
            [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "staging", type=Path, nargs="+",
        help="One or more directories the recorder wrote. Recording happens "
             "in sessions -- a box gets rebooted, a run gets restarted -- and "
             "each one starts its episode numbering at zero, so they are "
             "separate directories and get merged here."
    )
    parser.add_argument(
        "--repo-id", required=True, help="e.g. yourname/go2_chase."
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="Where to put the dataset. Default: LeRobot's own home.")
    parser.add_argument(
        "--max-tilt-deg", type=float, default=30.0,
        help="Drop frames where the robot is tilted more than this from "
             "vertical -- it has fallen over, and the target it is being "
             "given is one no policy could act on. Splits the episode at the "
             "gap rather than splicing across it. 0 keeps everything.")
    parser.add_argument(
        "--min-frames", type=int, default=10,
        help="Discard runs shorter than this (1 s at 10 Hz): too short to "
             "slice an action chunk out of.")
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Convert only the first N episodes. Useful for a quick look at a "
             "recording that is still running -- completed episodes are "
             "already final, only the metadata tail grows.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete an existing dataset at that location first.")
    return parser.parse_args()


def staging_features(info: dict) -> dict:
    """The recorder's feature dict, minus the columns LeRobot adds itself.

    timestamp / frame_index / episode_index / index / task_index are the
    writer's own bookkeeping; declaring them again collides with it.
    """
    reserved = {
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    features = {}
    for key, spec in info["features"].items():
        if key in reserved:
            continue
        features[key] = {
            "dtype": spec["dtype"],
            "shape": tuple(spec["shape"]),
            "names": spec.get("names"),
        }
    return features


def upright_runs(state: np.ndarray, max_tilt_deg: float) -> list[tuple[int, int]]:
    """Split an episode into runs of frames where the robot is on its feet.

    The first three elements of observation.state are the gravity direction
    in the body frame, so -g_z is the cosine of the tilt from vertical: 1 is
    level, 0 is lying on its side.

    Frames are dropped rather than repaired, and the episode is SPLIT at the
    gap instead of being stitched back together -- a join would put a jump in
    the timeline and imply a transition that never happened.
    """
    if max_tilt_deg <= 0:
        return [(0, len(state))]
    import math

    upright = np.clip(-state[:, 2], -1.0, 1.0) >= math.cos(math.radians(max_tilt_deg))
    runs, start = [], None
    for i, ok in enumerate(upright):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(upright)))
    return runs


def _add_run(dataset, columns, frames, features, vector_keys, task, start, stop):
    """Feed one contiguous run of frames to the dataset writer."""
    for i in range(start, stop):
        frame = {"task": task}
        for key in vector_keys:
            value = columns[key][i]
            spec = features[key]
            if spec["dtype"] == "bool":
                frame[key] = np.array([bool(value)])
            elif isinstance(value, list):
                frame[key] = np.asarray(value, dtype=np.float32)
            else:
                frame[key] = np.asarray([value], dtype=np.float32)
        for key, video in frames.items():
            frame[key] = np.asarray(video[i])
        dataset.add_frame(frame)


def main() -> None:
    args = parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # (staging dir, episode metadata) pairs, in the order given.
    work = []
    info = None
    for staging in args.staging:
        this_info = json.loads((staging / "meta" / "info.json").read_text())
        if info is None:
            info = this_info
        elif this_info["features"] != info["features"] or (
            this_info["fps"] != info["fps"]
        ):
            raise SystemExit(
                f"{staging} was recorded with different features or fps than "
                f"{args.staging[0]}; they cannot go in one dataset."
            )
        episodes = [
            json.loads(line)
            for line in (staging / "meta" / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if args.max_episodes is not None:
            episodes = episodes[: args.max_episodes]
        work.extend((staging, meta) for meta in episodes)
    tasks = [
        json.loads(line)
        for line in (args.staging[0] / "meta" / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    task_by_index = {t["task_index"]: t["task"] for t in tasks}
    features = staging_features(info)
    video_keys = [k for k, v in features.items() if v["dtype"] in ("video", "image")]
    vector_keys = [k for k in features if k not in video_keys]

    root = args.root
    if root is not None and root.exists() and args.overwrite:
        shutil.rmtree(root)

    dropped_tilt = dropped_short = written = 0
    print(f"{len(work)} staged episodes from {len(args.staging)} recording(s) "
          f"at {info['fps']} Hz")
    print("features: " + ", ".join(f"{k}{tuple(v['shape'])}" for k, v in features.items()))

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(round(info["fps"])),
        features=features,
        root=root,
        robot_type=info.get("robot_type"),
        use_videos=bool(video_keys),
    )

    for staging, meta in work:
        index = meta["episode_index"]
        table = pq.read_table(
            staging / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
        )
        columns = {k: table[k].to_pylist() for k in vector_keys}
        # The parquet is the authority on how long the episode is; the
        # metadata line is a summary of it. Trusting the summary once hid a
        # second recorder writing into the same directory, so cross-check.
        length = table.num_rows
        if length != meta["length"]:
            raise SystemExit(
                f"episode {index}: parquet has {length} rows but "
                f"meta/episodes.jsonl says {meta['length']}. Something else "
                "wrote into this directory -- do not trust it."
            )
        task = meta["tasks"][0] if meta.get("tasks") else task_by_index[0]
        frames = {
            key: read_video(
                staging / "videos" / "chunk-000" / key / f"episode_{index:06d}.mp4"
            )
            for key in video_keys
        }
        for key, video in frames.items():
            if len(video) != length:
                raise SystemExit(
                    f"episode {index}: {key} has {len(video)} frames but the "
                    f"table has {length}. The staging pair is inconsistent."
                )
        state = np.asarray(columns["observation.state"], dtype=np.float32)
        runs = upright_runs(state, args.max_tilt_deg)
        kept = sum(b - a for a, b in runs if b - a >= args.min_frames)
        dropped_tilt += length - sum(b - a for a, b in runs)
        for start, stop in runs:
            if stop - start < args.min_frames:
                dropped_short += stop - start
                continue
            _add_run(dataset, columns, frames, features, vector_keys,
                     task, start, stop)
            dataset.save_episode()
            written += 1
        print(f"  {staging.name} episode {index}: {kept}/{length} frames in "
              f"{sum(1 for a, b in runs if b - a >= args.min_frames)} run(s)")

    dataset.finalize()
    print(f"\nwrote {dataset.meta.total_episodes} episodes / "
          f"{dataset.meta.total_frames} frames as "
          f"{dataset.meta.info.codebase_version} to {dataset.root}")
    print(f"dropped {dropped_tilt} frames with the robot tilted past "
          f"{args.max_tilt_deg:.0f} deg, {dropped_short} in runs too short "
          f"to use")


if __name__ == "__main__":
    main()
