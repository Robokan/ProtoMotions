# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Remove genuine self-penetration frames from the Go2 corpus.

Ground truth comes from MuJoCo's own collision engine (not a distance proxy):
for every frame of every clip, the corpus pose is set into go2.xml's real
collision geometry and mj_forward's contacts are inspected for any non-floor,
inter-body penetration deeper than 2mm. 168/385 clips (1.62% of all frames)
are affected -- almost certainly a retarget artifact (opposite hips sit
9.3cm apart at rest but come within 0.2cm in these frames) that fires a
violent self-collision separation impulse on any reset landing near one
(Eric, 2026-08-24).

Two dispositions:
  - Clips where a majority of frames are bad (the `synthetic_side_step_*`
    clips are 100% bad -- procedurally generated, never checked against the
    real robot's geometry) are dropped entirely.
  - Otherwise the clip is split around the bad segments exactly like
    split_out_ankle_whips.py: pad, merge close regions, keep clean spans
    >= MIN_KEEP_SECONDS, rescale each sub-clip's sampling weight by its
    share of the parent.

Run scan_go2_selfcollision.py (or reuse its pickled mask) first.
"""

import pickle
import numpy as np
import torch

CORPUS = "data/motions/go2/go2_full.pt"
MASK_PICKLE = (
    "/tmp/claude-1000/-home-bizon-sparkpack/"
    "b231cc2e-5a14-4c97-a569-0f861bf24d01/scratchpad/go2_selfcollision_mask.pkl"
)
PAD_FRAMES = 3
MERGE_GAP_FRAMES = 10
MIN_KEEP_SECONDS = 1.0
DROP_CLIP_IF_BAD_FRAC_ABOVE = 0.5

FRAME_KEYS = ("gts", "grs", "gvs", "gavs", "dvs", "dps")


def keep_spans(mask, min_frames):
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
    with open(MASK_PICKLE, "rb") as fh:
        bad_by_clip = pickle.load(fh)

    d = torch.load(CORPUS, map_location="cpu", weights_only=False)
    files = [f.split("/")[-1] for f in d["motion_files"]]

    out = {k: [] for k in FRAME_KEYS}
    meta = {k: [] for k in ("motion_files", "motion_dt", "motion_num_frames",
                             "motion_lengths", "motion_weights", "length_starts")}
    cursor = 0
    dropped_clips, split_clips, dropped_frames = 0, 0, 0

    for i, f in enumerate(files):
        s, F = int(d["length_starts"][i]), int(d["motion_num_frames"][i])
        dt = float(d["motion_dt"][i])
        mask = bad_by_clip.get(f)

        if mask is None:
            spans = [(0, F)]
        elif mask.mean() > DROP_CLIP_IF_BAD_FRAC_ABOVE:
            dropped_clips += 1
            dropped_frames += F
            print(f"{f}: DROPPED (bad {mask.mean()*100:.0f}% of {F} frames)")
            continue
        else:
            padded = mask.copy()
            idx = np.where(mask)[0]
            for fr in idx:
                padded[max(0, fr - PAD_FRAMES): fr + 1 + PAD_FRAMES] = True
            bad_idx = np.where(padded)[0]
            for a, b in zip(bad_idx[:-1], bad_idx[1:]):
                if 0 < b - a <= MERGE_GAP_FRAMES:
                    padded[a:b] = True
            spans = keep_spans(padded, int(MIN_KEEP_SECONDS / dt))
            kept = sum(b - a for a, b in spans)
            dropped_frames += F - kept
            if kept < F:
                split_clips += 1
                print(f"{f}: split into {len(spans)} span(s), "
                      f"kept {kept}/{F} frames")

        for n, (a, b) in enumerate(spans):
            for k in FRAME_KEYS:
                out[k].append(d[k][s + a: s + b])
            sub = b - a
            suffix = f"#{n}" if len(spans) > 1 else ""
            meta["motion_files"].append(d["motion_files"][i] + suffix)
            meta["motion_dt"].append(dt)
            meta["motion_num_frames"].append(sub)
            meta["motion_lengths"].append(sub * dt)
            meta["motion_weights"].append(float(d["motion_weights"][i]) * sub / F)
            meta["length_starts"].append(cursor)
            cursor += sub

    new = {k: torch.cat(v, 0) for k, v in out.items()}
    new["motion_files"] = tuple(meta["motion_files"])
    for k in ("motion_dt", "motion_num_frames", "motion_lengths",
              "motion_weights", "length_starts"):
        new[k] = torch.tensor(meta[k], dtype=d[k].dtype)

    backup = CORPUS + ".bak_selfcollision"
    torch.save(d, backup)
    torch.save(new, CORPUS)
    print(f"\n{CORPUS}: {len(files)} clips -> {len(meta['motion_files'])} "
          f"({dropped_clips} dropped, {split_clips} split, "
          f"{dropped_frames} frames removed of "
          f"{int(sum(d['motion_num_frames']))}, "
          f"{dropped_frames/60:.1f}s)")
    print(f"backup: {backup}")


if __name__ == "__main__":
    main()
