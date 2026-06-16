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

# Stance detection. Both criteria must hold so that jump apexes (briefly slow
# but never height-stable) are not mistaken for planted stances:
# - sustained low speed: ballistic feet stay below the speed threshold only
#   ~0.05s around the apex, half the minimum window
# - stable height: a planted foot holds constant z; an airborne foot does not
STANCE_SPEED_THRESHOLD = 0.25  # [m/s] foot slower than this counts as planted
MIN_STANCE_FRAMES = 6  # minimum consecutive frames (~0.1s @ 60fps)
STANCE_HEIGHT_STD_MAX = 0.02  # [m] max foot z std-dev within a stance segment

# Detection params for "foot rests on an elevated surface" (classification +
# box placement). Looser than the legacy strict pass so real climbs whose feet
# only briefly settle (e.g. 20_clip_1) aren't missed; the all-4-feet + elevation
# gate keeps rearing/sitting out regardless.
STANCE_DETECT_SPEED = 0.45  # [m/s]
STANCE_DETECT_MIN_FRAMES = 3  # frames (~0.05s)
STANCE_DETECT_STD_MAX = 0.05  # [m]

# Free-fall gate: standing on a surface => root vertical accel ~ 0; a jump is in
# free-fall (~ -9.8 m/s^2) for its whole flight, apex included. A clip needs
# support only if there is a sustained window where all 4 feet are elevated AND
# the body is NOT in free-fall (resting on an elevated surface).
BALLISTIC_ACCEL_THRESHOLD = -4.0  # [m/s^2] below this = airborne (jump)
# A jump that lands back on the floor still shows a brief (~0.1-0.2s) burst of
# "all 4 elevated + not falling" around its apex/landing. Standing on a real
# platform lasts far longer (observed >1.3s). Require a sustained run so jumps
# (which never settle onto an elevated surface) are not mistaken for climbs.
MIN_SUPPORT_SECONDS = 1.0  # continuous supported-elevated stand to count
# Minimum height of the surface the feet rest on for it to count as a support
# structure (vs a low artifact / a curb the robot just steps over). Real climbs
# observed at 0.5-1.0m; lie-down/hop artifacts at 0.14-0.19m.
MIN_PLATFORM_ELEVATION = 0.30  # [m] above the clip's ground baseline

# Load-bearing test: a foot only needs support if the body stands above it
# (leg extended downward, weight passing through). A sitting robot holding a
# paw up is slow and height-stable too, but the paw is at/above root height
# and bears no load — no structure needed.
#
# Absolute thresholds, visually validated on both ANYmal-D and Go2. (A
# height-scaled variant was tried and over-flagged the smaller robot; the
# retargets do not scale structure heights linearly with robot size.) For
# same-source datasets, use --force-flag-manifest for cross-robot consistency.
ROOT_CLEARANCE_MIN = 0.25  # [m] strict pass: root must stand above the foot
RELAXED_CLEARANCE_MIN = 0.12  # [m] relaxed box-generation pass

# Support detection
ELEVATION_THRESHOLD = 0.10  # [m] stance height above ground to need support
BOX_PADDING = 0.15  # [m] horizontal padding around foot cluster
MIN_BOX_EXTENT = 0.30  # [m] minimum box side length


def find_stance_segments(speed: np.ndarray, speed_thr: float, min_frames: int) -> list:
    """Return [(start, end)] frame ranges where speed < speed_thr."""
    planted = speed < speed_thr
    segments = []
    start = None
    for i, p in enumerate(planted):
        if p and start is None:
            start = i
        elif not p and start is not None:
            if i - start >= min_frames:
                segments.append((start, i))
            start = None
    if start is not None and len(planted) - start >= min_frames:
        segments.append((start, len(planted)))
    return segments


def foot_elevated_stances(
    foot_pos, foot_speed, f, ground_z, speed_thr, min_frames, std_max, elev_thr
):
    """Elevated planted stance points (x,y,z) for ONE foot index `f`.

    A stance is: slow (planted), height-stable, and its mean height is more
    than `elev_thr` above the clip's ground. No load-bearing/clearance test —
    "is this foot resting on an elevated surface" is purely about the foot.
    """
    points = []
    for s, e in find_stance_segments(foot_speed[:, f], speed_thr, min_frames):
        seg = foot_pos[s:e, f, :]
        if seg[:, 2].std() > std_max:
            continue
        if (seg[:, 2].mean() - ground_z) <= elev_thr:
            continue
        points.append(seg.mean(axis=0))
    return points


def scan_clip(
    motion_path: Path,
    foot_indices: list,
    standing_height: float,
    force_flag: bool = False,
) -> dict:
    """Scan one clip; return classification and support boxes."""
    m = torch.load(motion_path, weights_only=False, map_location="cpu")
    body_pos = m["rigid_body_pos"].numpy()  # (N, B, 3)
    body_vel = m["rigid_body_vel"].numpy()  # (N, B, 3)
    fps = float(m["fps"])

    foot_pos = body_pos[:, foot_indices, :]  # (N, F, 3)
    foot_speed = np.linalg.norm(body_vel[:, foot_indices, :], axis=-1)  # (N, F)
    nfeet = foot_pos.shape[1]
    elevation_threshold = ELEVATION_THRESHOLD

    # Ground baseline: robust low percentile of all foot heights (the original
    # floor the clip was captured on).
    ground_z = float(np.percentile(foot_pos[:, :, 2], 5))

    # CLASSIFICATION (owner's criterion): a clip needs support ONLY if ALL FOUR
    # feet each leave the ground onto an elevated surface at some point — i.e.
    # the robot fully transfers its weight off the original floor (climbs onto
    # a block / up stairs). If even one foot stays grounded throughout, the
    # robot is still floor-supported and the elevated feet are just lifted in
    # the air (rearing, sitting with paws up, side-stepping) — NO support.
    # "Leaves the ground onto a surface" = a sustained, height-stable, planted
    # stance whose height is elevated above the ground. (No load-bearing test —
    # that wrongly rejected tall robots whose body sits level with high feet.)
    # One detection for both classification and box placement: a foot "rests on
    # an elevated surface" if it has a planted (slow), height-stable, sustained
    # stance elevated above the ground. Moderately loose so real climbs aren't
    # missed (e.g. 20_clip_1's feet at 0.3-0.4m); the all-4 requirement + the
    # elevation threshold keep rearing/sitting (grounded feet) out regardless.
    per_foot = [
        foot_elevated_stances(
            foot_pos, foot_speed, k, ground_z,
            STANCE_DETECT_SPEED, STANCE_DETECT_MIN_FRAMES,
            STANCE_DETECT_STD_MAX, elevation_threshold,
        )
        for k in range(nfeet)
    ]
    all_feet_elevated = all(len(p) > 0 for p in per_foot)

    # FREE-FALL gate: all 4 feet off the ground also happens during a JUMP, which
    # needs no support. Distinguish by the body's vertical acceleration: standing
    # on a surface => accel ~ 0; airborne (jump) => accel ~ -9.8 m/s^2 throughout
    # flight (including the apex). A clip needs support only if there is a
    # SUSTAINED window where all 4 feet are elevated AND the body is supported
    # (not in free-fall) — i.e. genuinely standing on an elevated surface.
    root_z = body_pos[:, 0, 2]
    win = 5
    kernel = np.ones(win) / win
    root_s = np.convolve(root_z, kernel, mode="same")  # smooth before 2nd deriv
    root_acc = np.gradient(np.gradient(root_s)) * fps * fps
    foot_elev = foot_pos[:, :, 2] - ground_z  # (N, F) per-foot height above ground
    all4_elev = (foot_elev > elevation_threshold).all(axis=1)
    supported = root_acc > BALLISTIC_ACCEL_THRESHOLD
    # PLATFORM-HEIGHT gate: the surface the feet rest on must be a real block,
    # not a low artifact. When a robot lies down later in a clip a foot can dip
    # very low, dragging ground_z down so the normal standing feet look "elevated"
    # by ~0.15m (anymal 1_clip_2); and a small hop tops out ~0.2m. Genuine climbs
    # put all feet (and the body) far higher. Require the supporting surface at a
    # frame (mean foot height above ground) to clear MIN_PLATFORM_ELEVATION.
    high_enough = foot_elev.mean(axis=1) > MIN_PLATFORM_ELEVATION
    support_frame = all4_elev & supported & high_enough

    # Contiguous supported-elevated segments lasting >= MIN_SUPPORT_SECONDS. Each
    # is a window where the robot genuinely stands on an elevated surface; a clip
    # may have several (climb up, stand, climb down) interleaved with flat travel.
    min_support_frames = max(1, int(round(MIN_SUPPORT_SECONDS * fps)))
    support_segments = []
    s = None
    for i, v in enumerate(np.append(support_frame, False)):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= min_support_frames:
                support_segments.append([s, i])
            s = None
    standing_on_elevated = len(support_segments) > 0

    if not (all_feet_elevated and standing_on_elevated) and not force_flag:
        return {"classification": "flat", "support_boxes": []}

    # Box placement: all elevated stance points across the four feet.
    relaxed_pts = [p for foot in per_foot for p in foot]
    elevated = np.array(relaxed_pts) if relaxed_pts else np.empty((0, 3))

    if len(elevated) == 0:
        # Flagged (e.g. force-flagged) but no elevated stances derivable: exclude
        # from flat training (terrain builder skips no-box entries).
        return {"classification": "needs_support", "support_boxes": []}

    # Cluster elevated stances by height (simple 5cm binning), one box per
    # height level covering the horizontal extent of its stance cluster.
    # Foot samples that must NOT end up inside a box volume: any airborne
    # trajectory point below a box top would clip through the geometry (e.g.
    # a jump up alongside the block face). Used to carve back padded faces.
    all_foot_pts = foot_pos.reshape(-1, 3)

    boxes = []
    heights = np.round(elevated[:, 2] / 0.05) * 0.05
    for h in sorted(set(heights.tolist())):
        pts = elevated[np.isclose(heights, h)]
        sx_min, sy_min = pts[:, :2].min(axis=0)  # stance bounds (must keep)
        sx_max, sy_max = pts[:, :2].max(axis=0)
        x_min, y_min = sx_min - BOX_PADDING, sy_min - BOX_PADDING
        x_max, y_max = sx_max + BOX_PADDING, sy_max + BOX_PADDING

        # Carve each padded face inward (never past the stance bounds) to
        # exclude airborne foot points that would be inside the box volume.
        viol = all_foot_pts[
            (all_foot_pts[:, 2] > ground_z + 0.05) & (all_foot_pts[:, 2] < h - 0.04)
        ]
        if len(viol):
            for _ in range(8):  # iterate until no violator remains inside
                inside = viol[
                    (viol[:, 0] > x_min) & (viol[:, 0] < x_max)
                    & (viol[:, 1] > y_min) & (viol[:, 1] < y_max)
                ]
                if len(inside) == 0:
                    break
                p = inside[0]
                # Push out via the cheapest face that doesn't cut stance support
                cands = []
                if p[0] <= sx_min: cands.append(("x_min", p[0] + 0.02))
                if p[0] >= sx_max: cands.append(("x_max", p[0] - 0.02))
                if p[1] <= sy_min: cands.append(("y_min", p[1] + 0.02))
                if p[1] >= sy_max: cands.append(("y_max", p[1] - 0.02))
                if not cands:
                    break  # violator inside stance footprint — unavoidable
                face, val = cands[0]
                if face == "x_min": x_min = max(x_min, val)
                elif face == "x_max": x_max = min(x_max, val)
                elif face == "y_min": y_min = max(y_min, val)
                elif face == "y_max": y_max = min(y_max, val)
        boxes.append(
            {
                "center_x": round(float((x_min + x_max) / 2), 3),
                "center_y": round(float((y_min + y_max) / 2), 3),
                "extent_x": round(max(float(x_max - x_min), MIN_BOX_EXTENT), 3),
                "extent_y": round(max(float(y_max - y_min), MIN_BOX_EXTENT), 3),
                "top_z": round(float(h - ground_z), 3),  # height above ground
            }
        )

    # Root xy travel bounds (motion-local coords) — used by the terrain
    # builder to size this clip's support cell.
    root_xy = body_pos[:, 0, :2]
    # Supported-elevated time windows (seconds), for clip splitting: the robot
    # only needs terrain during these; the rest of the clip is flat-trainable.
    support_segments_s = [
        [round(a / fps, 2), round(b / fps, 2)] for a, b in support_segments
    ]
    spans_whole = (
        len(support_segments) == 1
        and support_segments[0][0] <= int(0.05 * len(support_frame))
        and support_segments[0][1] >= int(0.95 * len(support_frame))
    )
    return {
        "classification": "needs_support",
        "ground_z": round(ground_z, 3),
        "duration_s": round(body_pos.shape[0] / fps, 2),
        "root_xy_min": [round(float(v), 3) for v in root_xy.min(axis=0)],
        "root_xy_max": [round(float(v), 3) for v in root_xy.max(axis=0)],
        "support_segments": support_segments_s,
        "splittable": not spans_whole,
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
    motion_lib: Path = typer.Option(
        None,
        help="Packed MotionLib .pt file. When given together with --exclude-file, "
        "writes the motion IDs of needs_support clips (for flat-ground training "
        "via motion_manager.exclude_motions_file).",
    ),
    exclude_file: Path = typer.Option(
        None, help="Output exclusion file (one motion ID per line)."
    ),
    standing_height: float = typer.Option(
        0.6,
        help="Robot nominal standing height [m] (default_root_height). Scales "
        "the load-bearing and elevation thresholds, e.g. 0.6 for ANYmal-D, "
        "0.34 for Go2.",
    ),
    force_flag_manifest: Path = typer.Option(
        None,
        help="Another robot's support manifest (same source captures): clips "
        "flagged needs_support there are force-flagged here too, with boxes "
        "from this robot's relaxed stance pass.",
    ),
):
    # Default quadruped layout: feet at indices 4, 8, 12, 16
    foot_indices = [4, 8, 12, 16]

    clips = sorted(clips_dir.glob("*.motion"))
    if not clips:
        raise typer.Exit(f"No .motion files in {clips_dir}")

    force_names = set()
    if force_flag_manifest is not None:
        other = yaml.safe_load(open(force_flag_manifest))
        force_names = {
            k for k, v in other.items() if v["classification"] == "needs_support"
        }

    manifest = {}
    counts = {"flat": 0, "needs_support": 0, "no_stance": 0}
    for clip in clips:
        result = scan_clip(
            clip, foot_indices, standing_height, force_flag=clip.name in force_names
        )
        manifest[clip.name] = result
        counts[result["classification"]] += 1
        if result["classification"] == "needs_support":
            if result["support_boxes"]:
                top = max(b["top_z"] for b in result["support_boxes"])
                print(f"NEEDS SUPPORT: {clip.name} ({len(result['support_boxes'])} boxes, max height {top:.2f}m)")
            else:
                print(f"NEEDS SUPPORT (no boxes derivable — excluded only): {clip.name}")

    with open(output, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=True)

    print(f"\nScanned {len(clips)} clips: {counts['flat']} flat, "
          f"{counts['needs_support']} need support, {counts['no_stance']} no stance detected")
    print(f"Manifest written to {output}")

    if motion_lib is not None and exclude_file is not None:
        lib = torch.load(motion_lib, weights_only=False, map_location="cpu")
        motion_files = lib["motion_files"] if isinstance(lib, dict) else lib.motion_files
        flagged = {
            name for name, r in manifest.items()
            if r["classification"] == "needs_support"
        }
        excluded_ids = [
            i for i, f in enumerate(motion_files) if Path(f).name in flagged
        ]
        with open(exclude_file, "w") as f:
            f.write("\n".join(str(i) for i in excluded_ids) + "\n")
        print(f"Exclusion file written to {exclude_file}: "
              f"{len(excluded_ids)} of {len(motion_files)} motions excluded")


if __name__ == "__main__":
    app()
