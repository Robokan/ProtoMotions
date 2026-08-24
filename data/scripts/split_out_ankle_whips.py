# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Split dog corpus clips around ankle whip artifacts, dropping the bad frames.

The source retarget carries fast ankle whips (predominantly the RIGHT foot:
140-205 deg sweeps inside 0.5 s, 400-660 deg/s peaks; worst clip 28) that
render as visible paw spins. Rather than smoothing (synthetic data the
discriminator can fingerprint), each clip is SPLIT at the defective segments
and the frames are removed (Eric, 2026-08-24).

Detection runs on the ORIGINAL clips' both feet; each clip's mirror is cut
with the SAME spans (mirrors are exact frame-aligned reflections, so the
original's right-ankle whip IS the mirror's left-ankle whip at the same
frames).

Sub-clips inherit the parent's name with a #<n> suffix and the parent's
sampling weight scaled by their share of its frames, preserving total
sampling mass. Sub-clips shorter than MIN_KEEP_SECONDS are discarded.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

CORPORA = ["data/motions/dog_v2/dog_full.pt", "data/motions/dog_v2/dog_flat.pt"]
BODY_NAMES = ['trunk','Spine','Spine1','Neck','Head','LeftShoulder','LeftArm',
              'LeftForeArm','LeftHand','RightShoulder','RightArm','RightForeArm',
              'RightHand','LeftUpLeg','LeftLeg','LeftFoot','RightUpLeg',
              'RightLeg','RightFoot','Tail','Tail1']
FEET = [("LeftFoot", "LeftLeg"), ("RightFoot", "RightLeg")]

SPEED_THRESHOLD_DEG_S = 300.0   # instantaneous local angular speed marking a whip
PAD_FRAMES = 5                  # widen each detected segment by this much
MERGE_GAP_FRAMES = 15           # merge segments closer than this
MIN_KEEP_SECONDS = 2.0          # discard clean spans shorter than this

FRAME_KEYS = ("gts", "grs", "gvs", "gavs", "dvs", "dps")


def bad_frames(grs: np.ndarray, dt: float) -> np.ndarray:
    """Boolean mask of frames inside a whip segment (either foot)."""
    F = grs.shape[0]
    mask = np.zeros(F, dtype=bool)
    for foot, parent in FEET:
        li, pi = BODY_NAMES.index(foot), BODY_NAMES.index(parent)
        loc = R.from_quat(grs[:, pi]).inv() * R.from_quat(grs[:, li])
        speed = np.degrees((loc[1:] * loc[:-1].inv()).magnitude()) / dt
        hot = np.where(speed > SPEED_THRESHOLD_DEG_S)[0]
        for f in hot:
            mask[max(0, f - PAD_FRAMES): min(F, f + 1 + PAD_FRAMES)] = True
    # merge segments separated by short gaps
    idx = np.where(mask)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 0 < b - a <= MERGE_GAP_FRAMES:
            mask[a:b] = True
    return mask


def keep_spans(mask: np.ndarray, min_frames: int):
    """Contiguous clean spans [(start, end)) at least min_frames long."""
    spans, start = [], None
    for f, bad in enumerate(mask):
        if not bad and start is None:
            start = f
        elif bad and start is not None:
            if f - start >= min_frames:
                spans.append((start, f))
            start = None
    if start is not None and len(mask) - start >= min_frames:
        spans.append((start, len(mask)))
    return spans


def main():
    for path in CORPORA:
        d = torch.load(path, map_location="cpu", weights_only=False)
        files = [f.split("/")[-1] for f in d["motion_files"]]
        # detect on originals only; mirrors reuse the original's spans
        spans_by_clip = {}
        for i, f in enumerate(files):
            if "mirror" in f:
                continue
            s, F = int(d["length_starts"][i]), int(d["motion_num_frames"][i])
            dt = float(d["motion_dt"][i])
            grs = d["grs"][s:s + F].numpy().astype(np.float64)
            mask = bad_frames(grs, dt)
            spans_by_clip[f] = keep_spans(mask, int(MIN_KEEP_SECONDS / dt))

        out = {k: [] for k in FRAME_KEYS}
        meta = {"motion_files": [], "motion_dt": [], "motion_num_frames": [],
                "motion_lengths": [], "motion_weights": [], "length_starts": []}
        cursor, dropped_s, split_clips = 0, 0.0, 0
        for i, f in enumerate(files):
            src = f.replace("_mirror", "") if "mirror" in f else f
            s, F = int(d["length_starts"][i]), int(d["motion_num_frames"][i])
            dt = float(d["motion_dt"][i])
            spans = spans_by_clip[src]
            kept = sum(b - a for a, b in spans)
            if kept < F:
                split_clips += 1
                dropped_s += (F - kept) * dt
            for n, (a, b) in enumerate(spans):
                for k in FRAME_KEYS:
                    out[k].append(d[k][s + a: s + b])
                sub = b - a
                suffix = f"#{n}" if len(spans) > 1 else ""
                meta["motion_files"].append(d["motion_files"][i] + suffix)
                meta["motion_dt"].append(dt)
                meta["motion_num_frames"].append(sub)
                meta["motion_lengths"].append(sub * dt)
                meta["motion_weights"].append(
                    float(d["motion_weights"][i]) * sub / F)
                meta["length_starts"].append(cursor)
                cursor += sub

        new = {k: torch.cat(v, 0) for k, v in out.items()}
        new["motion_files"] = tuple(meta["motion_files"])
        new["motion_dt"] = torch.tensor(meta["motion_dt"],
                                        dtype=d["motion_dt"].dtype)
        new["motion_num_frames"] = torch.tensor(meta["motion_num_frames"],
                                                dtype=d["motion_num_frames"].dtype)
        new["motion_lengths"] = torch.tensor(meta["motion_lengths"],
                                             dtype=d["motion_lengths"].dtype)
        new["motion_weights"] = torch.tensor(meta["motion_weights"],
                                             dtype=d["motion_weights"].dtype)
        new["length_starts"] = torch.tensor(meta["length_starts"],
                                            dtype=d["length_starts"].dtype)

        backup = path + ".bak_ankle_split"
        torch.save(d, backup)
        torch.save(new, path)
        print(f"{path}: {len(files)} clips -> {len(meta['motion_files'])} "
              f"({split_clips} clips cut, {dropped_s:.1f}s dropped, "
              f"{cursor} frames kept of {int(sum(d['motion_num_frames']))}) "
              f"backup: {backup}")


if __name__ == "__main__":
    main()
