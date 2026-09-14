# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Mirror the unmirrored Go2 clips and rebalance sampling weight by speed.

Two independent edits to a packaged Go2 motion lib, both pure data:

1. MIRROR the clips that have no mirror. In go2_flat_policy_backwards.pt that
   is the 39 `policy_reversed_*` backward captures plus
   `synthetic_trot_to_gallop` -- 31.7% of the sampling weight, and the only
   laterally asymmetric part of the corpus (the backward captures drift right:
   12 left vs 27 right, net -0.131 m/s body-lateral). Because motion_weights
   feeds BOTH reset sampling and `get_expert_disc_obs`, that asymmetry sat in
   the discriminator's notion of expert backward motion.

   The transform is a reflection through the world xz-plane, verified against
   the corpus's own 170 existing mirror pairs before use:
     positions      (x, y, z)     -> ( x, -y,  z)
     linear vel     (x, y, z)     -> ( x, -y,  z)
     angular vel    (x, y, z)     -> (-x,  y, -z)   (pseudovector)
     rotation  (x, y, z, w) xyzw  -> (-x,  y, -z, w)
     bodies  L<->R swap: FL<->FR, RL<->RR  (base stays)
     dofs    L<->R swap, abduction (hip) sign flipped, thigh/calf kept
   The body and dof maps are DERIVED from the existing pairs at run time
   rather than hard-coded, so a corpus with a different packing order still
   mirrors correctly (and the derivation fails loudly if it is not a
   permutation).

2. REBALANCE weight across speed buckets. The LLC standing-around problem is
   not frozen frames (7-15% of weight) -- it is that the corpus MIX is slow:
   clips averaging under 0.3 m/s held 59% of the weight and everything above
   1 m/s was 28 clips / 128 s out of 2608 s. Weight is reassigned so each
   bucket gets a target share, proportional to duration within the bucket:

     mean |forward| speed     target share
       < 0.3 m/s                  15%
       0.3 - 0.6 m/s              20%
       0.6 - 1.0 m/s              25%
       > 1.0 m/s                  40%

   That moves the median expert forward speed from +0.06 to about +0.65 m/s
   without inventing data. It buys this by oversampling the ~128 s of fast
   material about 2.5x, which is the deliberate trade (Eric: "We do not have
   more source material"); weighting straight by clip mean speed would have
   been 3.5x and narrows the discriminator onto one gait.

Usage:
    python data/scripts/mirror_and_balance_go2_corpus.py \
        --in-lib  data/motions/go2/go2_flat_policy_backwards.pt \
        --out-lib data/motions/go2/go2_flat_mirrored_balanced.pt
"""
from __future__ import annotations

import argparse
import collections
import os
from pathlib import Path

import numpy as np
import torch

PER = ["gts", "grs", "gvs", "gavs", "dvs", "dps"]
BUCKETS = [(0.0, 0.3, 0.15), (0.3, 0.6, 0.20), (0.6, 1.0, 0.25), (1.0, 1e9, 0.40)]


def base_name(n: str) -> str:
    """Identity key shared by a clip and its mirror."""
    if "_mirror" in n:
        return n.replace("_mirror", "")
    if n.endswith("_L") or n.endswith("_R"):
        return n[:-2]
    return n


def mirror_name(n: str) -> str:
    if n.endswith("_R"):
        return n[:-2] + "_mirror_L"
    return n + "_mirror"


def pair_up(names):
    g = collections.defaultdict(dict)
    for n in names:
        side = "M" if ("_mirror" in n or n.endswith("_L")) else "B"
        g[base_name(n)][side] = n
    return g


def derive_maps(m, names, clips):
    """Recover the body permutation and the dof permutation+signs from the
    corpus's OWN existing mirror pairs, so nothing is assumed about packing."""
    g = pair_up(names)
    pairs = [
        (v["B"], v["M"])
        for v in g.values()
        if len(v) == 2 and clips[v["B"]][1] == clips[v["M"]][1]
    ]
    if not pairs:
        raise SystemExit("no equal-length mirror pairs to derive the maps from")

    def cat(key, which):
        return torch.cat([m[key][clips[p[which]][0]:clips[p[which]][0] + clips[p[which]][1]] for p in pairs])

    # body permutation: match y-reflected base bodies onto mirror bodies
    B, M = cat("gts", 0), cat("gts", 1)
    Bm = B.clone()
    Bm[..., 1] *= -1
    nb = B.shape[1]
    body_perm = []
    for i in range(nb):
        err = (M - Bm[:, i:i + 1, :]).norm(dim=2).mean(0)
        body_perm.append(int(err.argmin()))
    if sorted(body_perm) != list(range(nb)):
        raise SystemExit(f"derived body map is not a permutation: {body_perm}")

    # dof permutation + sign
    Bd, Md = cat("dps", 0), cat("dps", 1)
    nd = Bd.shape[1]
    dof_perm, dof_sign = [], []
    for j in range(nd):
        ep = (Md - Bd[:, j:j + 1]).abs().mean(0)
        en = (Md + Bd[:, j:j + 1]).abs().mean(0)
        jp, jn = int(ep.argmin()), int(en.argmin())
        if ep[jp] <= en[jn]:
            dof_perm.append(jp); dof_sign.append(1.0)
        else:
            dof_perm.append(jn); dof_sign.append(-1.0)
    if sorted(dof_perm) != list(range(nd)):
        raise SystemExit(f"derived dof map is not a permutation: {dof_perm}")
    return (
        torch.tensor(body_perm),
        torch.tensor(dof_perm),
        torch.tensor(dof_sign, dtype=torch.float32),
        len(pairs),
    )


def mirror_clip(seg, body_perm, dof_perm, dof_sign):
    gts, grs, gvs, gavs, dvs, dps = (seg[k] for k in PER)
    out = {}
    out["gts"] = torch.stack([gts[..., 0], -gts[..., 1], gts[..., 2]], -1)[:, body_perm, :]
    out["gvs"] = torch.stack([gvs[..., 0], -gvs[..., 1], gvs[..., 2]], -1)[:, body_perm, :]
    out["gavs"] = torch.stack([-gavs[..., 0], gavs[..., 1], -gavs[..., 2]], -1)[:, body_perm, :]
    out["grs"] = torch.stack(
        [-grs[..., 0], grs[..., 1], -grs[..., 2], grs[..., 3]], -1
    )[:, body_perm, :]
    for key, src in (("dps", dps), ("dvs", dvs)):
        t = torch.zeros_like(src)
        t[:, dof_perm] = src * dof_sign
        out[key] = t
    return out


def heading_forward(grs_root, gvs_root):
    x, y, z, w = grs_root[:, 0], grs_root[:, 1], grs_root[:, 2], grs_root[:, 3]
    h = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.cos(h) * gvs_root[:, 0] + torch.sin(h) * gvs_root[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-lib", required=True, type=Path)
    ap.add_argument("--out-lib", required=True, type=Path)
    ap.add_argument("--no-mirror", action="store_true", help="skip step 1")
    ap.add_argument("--no-rebalance", action="store_true", help="skip step 2")
    ap.add_argument("--keep-pad-sec", type=float, default=None,
                    help="Trim the LEADING and TRAILING still run of every clip "
                         "down to at most this many seconds (None = no trim). "
                         "This corpus has no interior standing left to split -- "
                         "interior still runs have a median length of 1 frame and "
                         "none reach 1s -- but clips still carry a trailing pad "
                         "averaging 0.44s. Trimming it also nudges each clip's "
                         "mean speed up, so clips land in faster buckets.")
    ap.add_argument("--still-joint-vel", type=float, default=0.5,
                    help="max|dof_vel| below which a frame counts as still.")
    args = ap.parse_args()

    m = torch.load(args.in_lib, map_location="cpu", weights_only=False)
    names = [os.path.basename(str(p)).replace(".motion", "") for p in m["motion_files"]]
    paths = [str(p) for p in m["motion_files"]]
    starts = [int(a) for a in m["length_starts"]]
    lens = [int(a) for a in m["motion_num_frames"]]
    clips = {n: (starts[i], lens[i]) for i, n in enumerate(names)}
    dt = m["motion_dt"]
    w0 = m["motion_weights"].double()

    body_perm, dof_perm, dof_sign, npairs = derive_maps(m, names, clips)
    print(f"derived from {npairs} existing mirror pairs:")
    print(f"  body perm {body_perm.tolist()}")
    print(f"  dof  perm {dof_perm.tolist()}  signs {[int(s) for s in dof_sign]}")

    out_per = {k: [] for k in PER}
    out_files, out_nf, out_dt, out_w = [], [], [], []

    def emit(name, seg, nframes, dtv, wv):
        for k in PER:
            out_per[k].append(seg[k])
        out_files.append(name if name.endswith(".motion") else name + ".motion")
        out_nf.append(nframes)
        out_dt.append(float(dtv))
        out_w.append(float(wv))

    g = pair_up(names)
    unmirrored = [v["B"] for v in g.values() if len(v) == 1 and "B" in v]
    unmirrored += [v["M"] for v in g.values() if len(v) == 1 and "M" in v]

    fps = 1.0 / float(dt[0])
    keep_frames = (
        None if args.keep_pad_sec is None else int(round(args.keep_pad_sec * fps))
    )

    def pad_cuts(n):
        """(front, back) frames to drop from this clip, before pairing."""
        if keep_frames is None:
            return 0, 0
        a, L = clips[n]
        still = (
            m["dvs"][a:a + L].abs().max(dim=1).values < args.still_joint_vel
        ).numpy()
        lo = 0
        while lo < L and still[lo]:
            lo += 1
        hi = L
        while hi > lo and still[hi - 1]:
            hi -= 1
        return max(0, lo - keep_frames), max(0, L - (hi + keep_frames))

    # A clip and its mirror MUST be cut identically, or the pair stops being a
    # pair. The two carry the same motion through different numerical paths, so
    # their still masks differ by a frame here and there; cutting each on its
    # own produced 53 length-mismatched pairs. Take the least aggressive cut
    # across the pair and apply it to both.
    cuts = {n: pad_cuts(n) for n in names}
    for v in pair_up(names).values():
        if len(v) == 2:
            a_, b_ = v["B"], v["M"]
            f = min(cuts[a_][0], cuts[b_][0])
            bk = min(cuts[a_][1], cuts[b_][1])
            cuts[a_] = cuts[b_] = (f, bk)

    cut_frames = 0
    for i, n in enumerate(names):
        a, L = clips[n]
        f, bk = cuts[n]
        if L - f - bk < 2:
            f = bk = 0
        seg = {k: m[k][a + f:a + L - bk] for k in PER}
        cut_frames += f + bk
        emit(paths[i], seg, L - f - bk, dt[i], w0[i])
    if args.keep_pad_sec is not None:
        print(f"trimmed {cut_frames / fps:.0f}s of leading/trailing "
              f"stillness (kept {args.keep_pad_sec}s pads)")

    made = 0
    if not args.no_mirror:
        for n in unmirrored:
            i = names.index(n)
            a, L = clips[n]
            f, bk = cuts[n]
            if L - f - bk < 2:
                f = bk = 0
            seg = {k: m[k][a + f:a + L - bk] for k in PER}
            L = L - f - bk
            emit(
                os.path.join(os.path.dirname(paths[i]), mirror_name(n)),
                mirror_clip(seg, body_perm, dof_perm, dof_sign),
                L, dt[i], w0[i],
            )
            made += 1
    print(f"mirrored {made} previously unmirrored clips")

    nf = torch.tensor(out_nf, dtype=m["motion_num_frames"].dtype)
    cursor, new_starts = 0, []
    for v in out_nf:
        new_starts.append(cursor); cursor += v
    out = {k: torch.cat(out_per[k], dim=0) for k in PER}
    out["length_starts"] = torch.tensor(new_starts, dtype=m["length_starts"].dtype)
    out["motion_num_frames"] = nf
    out["motion_dt"] = torch.tensor(out_dt, dtype=m["motion_dt"].dtype)
    out["motion_lengths"] = torch.tensor(
        [(v - 1) * d for v, d in zip(out_nf, out_dt)], dtype=m["motion_lengths"].dtype
    )
    out["motion_files"] = tuple(out_files)

    weights = torch.tensor(out_w, dtype=m["motion_weights"].dtype)
    if not args.no_rebalance:
        speed = np.array([
            heading_forward(
                out["grs"][s:s + L, 0, :], out["gvs"][s:s + L, 0, :]
            ).abs().mean().item()
            for s, L in zip(new_starts, out_nf)
        ])
        secs = np.array([(v - 1) * d for v, d in zip(out_nf, out_dt)])
        W = np.zeros(len(speed))
        print("\nrebalancing weight by clip mean |forward| speed:")
        for lo, hi, tgt in BUCKETS:
            sel = (speed >= lo) & (speed < hi)
            if not sel.any():
                print(f"  {lo:.1f}-{hi:.1f}: EMPTY"); continue
            W[sel] = tgt * (secs[sel] / secs[sel].sum())
            label = f"{lo:.1f}-{hi:.1f} m/s" if hi < 1e8 else f">{lo:.1f} m/s"
            print(f"  {label:>14s}: {sel.sum():4d} clips, {secs[sel].sum():7.0f}s -> {100*tgt:4.0f}% of weight")
        weights = torch.tensor(W / W.sum(), dtype=m["motion_weights"].dtype)
    out["motion_weights"] = weights

    torch.save(out, args.out_lib)
    print(f"\nclips {len(names)} -> {len(out_files)},  "
          f"frames {sum(lens)} -> {sum(out_nf)},  "
          f"duration {sum(lens)*float(dt[0]):.0f}s -> {sum(out_nf)*float(dt[0]):.0f}s")
    print(f"wrote {args.out_lib}")


if __name__ == "__main__":
    main()
