# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Scan converted motion clips for elevated foot supports.

For each clip, detects stance phases (low foot speed) and flags sustained
stances above ground level — evidence the original capture had stairs,
platforms, or other structures the simulation terrain must reproduce.

Output: a YAML manifest classifying each clip as `flat` or `needs_support`,
with per-clip support boxes (motion-local coordinates) that a terrain
builder can stamp into the heightfield.

Usage:
    python data/scripts/scan_clip_support_geometry.py \
        --clips-dir data/motions/anymal_d/clips \
        --output data/motions/anymal_d/support_manifest.yaml
"""

from pathlib import Path

import numpy as np
import torch
import typer
import yaml

app = typer.Typer(pretty_exceptions_enable=False)

# Stance detection
STANCE_SPEED_THRESHOLD = 0.25  # [m/s] foot slower than this counts as planted
MIN_STANCE_FRAMES = 6  # minimum consecutive frames (~0.1s @ 60fps)

# Support detection
ELEVATION_THRESHOLD = 0.10  # [m] stance height above ground to need support
BOX_PADDING = 0.15  # [m] horizontal padding around foot cluster
MIN_BOX_EXTENT = 0.30  # [m] minimum box side length


def find_stance_segments(speed: np.ndarray) -> list:
    """Return [(start, end)] frame ranges where speed < threshold."""
    planted = speed < STANCE_SPEED_THRESHOLD
    segments = []
    start = None
    for i, p in enumerate(planted):
        if p and start is None:
            start = i
        elif not p and start is not None:
            if i - start >= MIN_STANCE_FRAMES:
                segments.append((start, i))
            start = None
    if start is not None and len(planted) - start >= MIN_STANCE_FRAMES:
        segments.append((start, len(planted)))
    return segments


def scan_clip(motion_path: Path, foot_indices: list) -> dict:
    """Scan one clip; return classification and support boxes."""
    m = torch.load(motion_path, weights_only=False, map_location="cpu")
    body_pos = m["rigid_body_pos"].numpy()  # (N, B, 3)
    body_vel = m["rigid_body_vel"].numpy()  # (N, B, 3)
    fps = float(m["fps"])

    foot_pos = body_pos[:, foot_indices, :]  # (N, F, 3)
    foot_speed = np.linalg.norm(body_vel[:, foot_indices, :], axis=-1)  # (N, F)

    # Collect all stance foot positions across feet
    stance_points = []  # (x, y, z)
    for f in range(foot_pos.shape[1]):
        for s, e in find_stance_segments(foot_speed[:, f]):
            seg = foot_pos[s:e, f, :]
            stance_points.append(seg.mean(axis=0))
    if not stance_points:
        return {"classification": "no_stance", "support_boxes": []}

    stance_points = np.array(stance_points)
    ground_z = float(np.percentile(stance_points[:, 2], 10))

    elevated = stance_points[stance_points[:, 2] > ground_z + ELEVATION_THRESHOLD]
    if len(elevated) == 0:
        return {"classification": "flat", "support_boxes": []}

    # Cluster elevated stances by height (simple 5cm binning), one box per
    # height level covering the horizontal extent of its stance cluster.
    boxes = []
    heights = np.round(elevated[:, 2] / 0.05) * 0.05
    for h in sorted(set(heights.tolist())):
        pts = elevated[np.isclose(heights, h)]
        x_min, y_min = pts[:, :2].min(axis=0) - BOX_PADDING
        x_max, y_max = pts[:, :2].max(axis=0) + BOX_PADDING
        boxes.append(
            {
                "center_x": round(float((x_min + x_max) / 2), 3),
                "center_y": round(float((y_min + y_max) / 2), 3),
                "extent_x": round(max(float(x_max - x_min), MIN_BOX_EXTENT), 3),
                "extent_y": round(max(float(y_max - y_min), MIN_BOX_EXTENT), 3),
                "top_z": round(float(h - ground_z), 3),  # height above ground
            }
        )

    return {
        "classification": "needs_support",
        "ground_z": round(ground_z, 3),
        "duration_s": round(body_pos.shape[0] / fps, 2),
        "support_boxes": boxes,
    }


@app.command()
def main(
    clips_dir: Path = typer.Option(..., help="Directory of converted .motion files"),
    output: Path = typer.Option(..., help="Output YAML manifest path"),
    foot_body_names: str = typer.Option(
        "FOOT,foot",
        help="Comma-separated substrings identifying foot bodies (matched against "
        "robot body order; requires --robot-name OR uses default quadruped index "
        "layout base + 4x(HIP,THIGH,SHANK,FOOT))",
    ),
):
    # Default quadruped layout: feet at indices 4, 8, 12, 16
    foot_indices = [4, 8, 12, 16]

    clips = sorted(clips_dir.glob("*.motion"))
    if not clips:
        raise typer.Exit(f"No .motion files in {clips_dir}")

    manifest = {}
    counts = {"flat": 0, "needs_support": 0, "no_stance": 0}
    for clip in clips:
        result = scan_clip(clip, foot_indices)
        manifest[clip.name] = result
        counts[result["classification"]] += 1
        if result["classification"] == "needs_support":
            top = max(b["top_z"] for b in result["support_boxes"])
            print(f"NEEDS SUPPORT: {clip.name} ({len(result['support_boxes'])} boxes, max height {top:.2f}m)")

    with open(output, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=True)

    print(f"\nScanned {len(clips)} clips: {counts['flat']} flat, "
          f"{counts['needs_support']} need support, {counts['no_stance']} no stance detected")
    print(f"Manifest written to {output}")


if __name__ == "__main__":
    app()
