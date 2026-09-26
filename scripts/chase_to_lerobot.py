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
        "staging", type=Path, help="Directory the recorder wrote."
    )
    parser.add_argument(
        "--repo-id", required=True, help="e.g. yourname/go2_chase."
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="Where to put the dataset. Default: LeRobot's own home.")
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


def main() -> None:
    args = parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    staging = args.staging
    info = json.loads((staging / "meta" / "info.json").read_text())
    tasks = [
        json.loads(line)
        for line in (staging / "meta" / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    task_by_index = {t["task_index"]: t["task"] for t in tasks}
    episodes = [
        json.loads(line)
        for line in (staging / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    features = staging_features(info)
    video_keys = [k for k, v in features.items() if v["dtype"] in ("video", "image")]
    vector_keys = [k for k in features if k not in video_keys]

    root = args.root
    if root is not None and root.exists() and args.overwrite:
        shutil.rmtree(root)

    print(f"{len(episodes)} episodes, {info['total_frames']} frames at "
          f"{info['fps']} Hz")
    print("features: " + ", ".join(f"{k}{tuple(v['shape'])}" for k, v in features.items()))

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(round(info["fps"])),
        features=features,
        root=root,
        robot_type=info.get("robot_type"),
        use_videos=bool(video_keys),
    )

    for meta in episodes:
        index = meta["episode_index"]
        table = pq.read_table(
            staging / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
        )
        columns = {k: table[k].to_pylist() for k in vector_keys}
        task = meta["tasks"][0] if meta.get("tasks") else task_by_index[0]
        frames = {
            key: read_video(
                staging / "videos" / "chunk-000" / key / f"episode_{index:06d}.mp4"
            )
            for key in video_keys
        }
        length = meta["length"]
        for key, video in frames.items():
            if len(video) != length:
                raise SystemExit(
                    f"episode {index}: {key} has {len(video)} frames but the "
                    f"table has {length}. The staging pair is inconsistent."
                )
        for i in range(length):
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
        dataset.save_episode()
        print(f"  episode {index}: {length} frames")

    dataset.finalize()
    print(f"\nwrote {dataset.meta.total_episodes} episodes / "
          f"{dataset.meta.total_frames} frames as "
          f"{dataset.meta.info.codebase_version} to {dataset.root}")


if __name__ == "__main__":
    main()
