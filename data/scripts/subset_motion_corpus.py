#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a smaller packed corpus from a subset of an existing one.

Frame-indexed tensors are sliced per clip and length_starts is rebuilt, so the
output is a valid standalone motion library -- useful for pulling a handful of
suspect clips out for visual review before deciding to drop them.

    python data/scripts/subset_motion_corpus.py IN.pt OUT.pt --clips 3 17 42
    python data/scripts/subset_motion_corpus.py IN.pt OUT.pt --names 45_clip_2
"""

import argparse

import torch


def subset(d: dict, keep: list) -> dict:
    n_frames = int(d["gts"].shape[0])
    n_clips = len(d["motion_files"])
    starts, counts = d["length_starts"], d["motion_num_frames"]

    idx = torch.cat(
        [torch.arange(int(starts[i]), int(starts[i]) + int(counts[i])) for i in keep]
    )
    keep_t = torch.tensor(keep, dtype=torch.long)

    out = {}
    for k, v in d.items():
        if torch.is_tensor(v) and v.shape and v.shape[0] == n_frames:
            out[k] = v[idx]
        elif torch.is_tensor(v) and v.shape and v.shape[0] == n_clips:
            out[k] = v[keep_t]
        elif not torch.is_tensor(v) and hasattr(v, "__len__") and len(v) == n_clips:
            out[k] = tuple(v[i] for i in keep)
        else:
            out[k] = v

    # length_starts must be recomputed for the new packing, not sliced.
    shifted = out["motion_num_frames"].roll(1)
    shifted[0] = 0
    out["length_starts"] = shifted.cumsum(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--clips", type=int, nargs="*", default=[])
    ap.add_argument("--names", nargs="*", default=[])
    args = ap.parse_args()

    d = torch.load(args.infile, map_location="cpu", weights_only=False)
    names = [str(x) for x in d["motion_files"]]

    keep = list(args.clips)
    for want in args.names:
        hits = [i for i, nm in enumerate(names) if want in nm]
        if not hits:
            raise SystemExit(f"no clip matching {want!r}")
        keep.extend(hits)
    keep = sorted(set(keep))
    if not keep:
        raise SystemExit("nothing selected; pass --clips and/or --names")

    out = subset(d, keep)
    torch.save(out, args.outfile)

    chk = torch.load(args.outfile, map_location="cpu", weights_only=False)
    assert int(chk["gts"].shape[0]) == int(chk["motion_num_frames"].sum())
    assert len(chk["motion_files"]) == len(keep)
    print(f"wrote {args.outfile}: {len(keep)} clips, "
          f"{int(chk['gts'].shape[0])} frames, "
          f"{float(chk['motion_lengths'].sum()):.1f}s (verified on reload)")
    for i, nm in enumerate(chk["motion_files"]):
        print(f"  [{i+1}] {str(nm).split('/')[-1]}")


if __name__ == "__main__":
    main()
