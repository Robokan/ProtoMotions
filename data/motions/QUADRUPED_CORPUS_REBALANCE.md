# Quadruped corpus rebalance (go2 + anymal_d), 2026-09-14

Built by `data/scripts/mirror_and_balance_go2_corpus.py`. Both corpora are pure
data edits of an existing packaged motion lib: no recapture, no retarget.

## Why

The go2 ASE low-level controller (`results/go2_ase_llc_v3`, epoch 123710) mostly
stood still under random latents even though running, turning and backward
walking were all present in its latent space. Two candidate causes were tested
and one survived.

**Not undertraining.** The LLC converged at roughly epoch 45000 and then ran a
further 78000 epochs without moving:

| epoch | style reward | encoder reward | episode length |
|---|---|---|---|
| 15000 | 0.377 | 0.581 | 289 |
| 45000 | 0.394 | 0.588 | 299 |
| 120000 | 0.401 | 0.589 | 299 |

**Not frozen frames either.** Genuinely still frames were only 7-15% of the
sampling weight depending on threshold, and there was no interior standing left
to split out: interior still runs had a median length of one frame and none
reached 1 s. Whatever earlier pass split clips on long stillness had already
done its work.

**It was the corpus mix.** Clips averaging under 0.3 m/s held 59% of the go2
sampling weight, and everything above 1 m/s was 28 clips totalling 128 s out of
2608 s. The median frame the discriminator was shown moved at 0.06 m/s. Since
`motion_weights` feeds BOTH reset sampling and `get_expert_disc_obs`, the robot
both started slow and was told slow is what real motion looks like. Standing was
a legitimately high-reward behaviour.

## What the pipeline does

1. **Mirror** any clip that has no mirror. The reflection is through the world
   xz-plane; the body permutation and the dof permutation+signs are DERIVED at
   run time from the corpus's own existing mirror pairs rather than hard-coded,
   and the run aborts if either derivation is not a permutation. This is what
   let the same script handle both robots: go2 packs bodies base+FL+FR+RL+RR
   while anymal_d packs base+left8+right8, and the derivation found each.
2. **Trim** only the LEADING and TRAILING still run of every clip, keeping a
   pad (`--keep-pad-sec`). Interior content is untouched. A clip and its mirror
   always receive the SAME cut: trimming them independently desynchronised 53
   go2 pairs, because the two carry the same motion through different numerical
   paths and disagree by a frame or two about where stillness starts.
3. **Rebalance** sampling weight to hit target shares across mean-|forward|
   speed buckets, proportional to duration within each bucket:

   | bucket | target share |
   |---|---|
   | < 0.3 m/s | 15% |
   | 0.3 - 0.6 m/s | 20% |
   | 0.6 - 1.0 m/s | 25% |
   | > 1.0 m/s | 40% |

## Results

### go2

`go2_flat_policy_backwards.pt` -> `go2_flat_mirrored_balanced.pt` (426 clips, 2528 s)

| | before | after |
|---|---|---|
| unmirrored clips | 40 | 0 |
| body-lateral bias | -0.044 m/s | +0.0007 m/s |
| median expert forward speed | +0.06 m/s | +0.61 m/s |
| frames above 1 m/s | 13.0% | 30.7% |
| standing retained | 377 s | 243 s |
| weight on named transition clips | 3.1% | 12.8% |

The 40 newly mirrored clips are the 39 `policy_reversed_*` backward captures
plus `synthetic_trot_to_gallop`. They were 31.7% of the weight and the only
laterally asymmetric part of the corpus: the backward captures drifted right,
12 left against 27 right, net -0.131 m/s body-lateral, and that asymmetry sat
inside the discriminator's notion of expert backward motion. The new mirrors are
algebraically exact, unlike the pipeline-produced originals: forward-path error
and lateral cancellation both measured exactly zero.

### anymal_d

`anymal_d_flat_simframes.pt` -> `anymal_d_flat_simframes_balanced.pt` (372 clips, 2616 s)

Base is the sim-repacked corpus, NOT `anymal_d_flat.pt` — see the ANYmal
convention notes: the raw corpus stores retarget-frame rotations and
link-origin velocities that AMP cannot train against.

| | before | after |
|---|---|---|
| unmirrored clips | 0 | 0 |
| median expert forward speed | +0.32 m/s | +0.63 m/s |
| frames above 1 m/s | 33.7% | 37.2% |
| standing retained | - | 249 s |

ANYmal needed much less work than go2. It was already fully mirrored, already
laterally unbiased, and its fast bucket already held 39.1% of the weight against
a 40% target. The rebalance mostly moved weight out of the slow bucket (33.6%
-> 15%) into the two middle buckets, which were starved at 16.8% and 10.6%.

## Reproduce

```bash
python data/scripts/mirror_and_balance_go2_corpus.py \
    --in-lib  data/motions/go2/go2_flat_policy_backwards.pt \
    --out-lib data/motions/go2/go2_flat_mirrored_balanced.pt \
    --keep-pad-sec 0.15 --still-joint-vel 0.5

python data/scripts/mirror_and_balance_go2_corpus.py \
    --in-lib  data/motions/anymal_d/anymal_d_flat_simframes.pt \
    --out-lib data/motions/anymal_d/anymal_d_flat_simframes_balanced.pt \
    --keep-pad-sec 0.15 --still-joint-vel 0.5
```

Inspect either with the weighted visualizer, which draws clips by the library's
own sampling weights the way training does, so the rebalance is actually visible:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/motion_libs_visualizer.py \
    --motion_files data/motions/go2/go2_flat_mirrored_balanced.pt \
    --robot go2 --simulator isaaclab --weighted-random
```

## Open items

- **Backward share on go2 was not decided, it fell out.** Mirroring doubled the
  backward material but the speed buckets key on mean ABSOLUTE forward speed,
  so those slow captures landed in the 15% bucket and their weight went from
  27.4% to 8.0%. Backward frames are now 7.6% against 66.1% forward. If backward
  deserves a deliberate share it needs its own bucket rather than being sorted
  by |speed|.
- **go2 has no more fast material.** 40% of the weight now rides on 136 s,
  oversampled about 2.5x. Weighting straight by clip mean speed would have been
  3.5x and narrows the discriminator onto a single gait. More source material is
  the real ceiling.
- **Neither LLC has been retrained yet.** These corpora are built and validated
  but unused. Both LLCs, and then any HLC on top of them, need a fresh run.
- The 3 length-mismatched go2 pairs and 1 anymal_d pair predate this work and
  were carried through unchanged.
