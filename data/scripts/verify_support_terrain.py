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
"""Numeric verification of the motion-support terrain feature.

Constructs a Terrain directly (no simulator / GUI) with a support manifest and
asserts that:
  1. For several flagged motions, the terrain height at each support box
     center (anchor + box center) equals the box's top_z.
  2. A point inside the cell but outside all boxes is at ground level (~0).
  3. Random spawn sampling never lands inside any motion-support cell.

Usage:
    python data/scripts/verify_support_terrain.py \
        [--manifest data/motions/anymal_d/support_manifest.yaml] \
        [--motion-lib data/motions/anymal_d/anymal_d_full.pt]
"""

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from protomotions.components.terrains.config import TerrainConfig  # noqa: E402
from protomotions.components.terrains.terrain import Terrain  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=os.path.join(REPO_ROOT, "data/motions/anymal_d/support_manifest.yaml"),
    )
    parser.add_argument(
        "--motion-lib",
        default=os.path.join(REPO_ROOT, "data/motions/anymal_d/anymal_d_full.pt"),
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    config = TerrainConfig(
        num_levels=2,
        num_terrains=2,
        border_size=10.0,
        motion_support_manifest=args.manifest,
        motion_support_motion_lib=args.motion_lib,
    )
    num_envs = 4
    terrain = Terrain(config=config, num_envs=num_envs, device=device)

    cells = terrain._motion_support_cells
    assert len(cells) > 0, "No motion-support cells were created"
    assert len(terrain.motion_support_origins) == len(cells)
    print(f"Created {len(cells)} motion-support cells")
    print(
        f"Terrain: {terrain.tot_rows} x {terrain.tot_cols} px, "
        f"strip rows: {terrain._motion_support_rows}"
    )

    # is_flat() must be False so height correction is not skipped
    assert terrain.is_flat() is False, "is_flat() must be False with support cells"
    print("is_flat() correctly returns False")

    tol = config.vertical_scale + 1e-6
    margin = config.motion_support_margin
    h_scale = config.horizontal_scale

    # 1 + 2. Check box-top heights and a flat point for the first few cells
    num_checked = 0
    for cell in cells[:3]:
        motion_id = cell["motion_id"]
        anchor_x, anchor_y = terrain.motion_support_origins[motion_id]
        for box in cell["boxes"]:
            query = torch.tensor(
                [[anchor_x + box["center_x"], anchor_y + box["center_y"], 0.0]],
                device=device,
            )
            height = terrain.get_ground_heights(query).item()
            # Overlapping boxes keep the maximum height, so the expected
            # height at a box center is the max top_z over covering boxes.
            expected = max(
                other["top_z"]
                for other in cell["boxes"]
                if abs(other["center_x"] - box["center_x"]) <= other["extent_x"] / 2
                and abs(other["center_y"] - box["center_y"]) <= other["extent_y"] / 2
            )
            assert expected >= box["top_z"]
            assert abs(height - expected) <= tol, (
                f"Motion {motion_id}: box center height {height:.4f} != "
                f"expected {expected:.4f} (own top_z {box['top_z']:.4f})"
            )
        # Point inside the cell's flat margin band (outside all boxes)
        probe = torch.tensor(
            [
                [
                    (cell["start_row"] + 1) * h_scale + margin / 2,
                    (cell["start_col"] + 1) * h_scale + margin / 2,
                    0.0,
                ]
            ],
            device=device,
        )
        flat_height = terrain.get_ground_heights(probe).item()
        assert abs(flat_height) <= tol, (
            f"Motion {motion_id}: flat probe height {flat_height:.4f} != 0"
        )
        num_checked += 1
        print(
            f"Motion {motion_id}: {len(cell['boxes'])} box tops match top_z "
            f"(anchor=({anchor_x:.2f}, {anchor_y:.2f})), flat probe ~0"
        )
    assert num_checked >= 2, "Expected at least 2 flagged motions to verify"

    # 3. Random spawn sampling must never land inside any support cell
    for sample_flat in (False, True):
        locs = terrain.sample_valid_locations(1000, sample_flat=sample_flat)
        strip_start_x = (terrain.tot_rows - terrain._motion_support_rows) * h_scale
        assert (locs[:, 0] < strip_start_x).all(), (
            "Random spawn locations landed in the motion-support strip "
            f"(sample_flat={sample_flat})"
        )
        for cell in cells:
            x0 = cell["start_row"] * h_scale
            x1 = (cell["start_row"] + cell["num_rows"]) * h_scale
            y0 = cell["start_col"] * h_scale
            y1 = (cell["start_col"] + cell["num_cols"]) * h_scale
            inside = (
                (locs[:, 0] >= x0)
                & (locs[:, 0] < x1)
                & (locs[:, 1] >= y0)
                & (locs[:, 1] < y1)
            )
            assert not inside.any(), (
                f"Spawn location inside support cell of motion {cell['motion_id']}"
            )
        print(
            f"1000 sampled locations (sample_flat={sample_flat}) all outside "
            "the support strip"
        )

    print("\nAll motion-support terrain checks passed.")


if __name__ == "__main__":
    main()
