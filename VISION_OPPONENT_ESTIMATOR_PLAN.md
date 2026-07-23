# Plan: Vision-Based Opponent Estimator (fighter's-eye view)

Goal: mount a camera on the SOMA fighter's face, record bouts (video +
ground-truth observations), and train a network that reconstructs the
policy's **opponent observation block** from vision alone — so a champion
policy can eventually fight from first-person video instead of privileged
omniscient state.

This is the standard teacher-student privileged-distillation pattern: the
simulator already computes perfect opponent state (even behind the agent);
the student learns to estimate it from a front-facing camera, with memory
covering the out-of-view case.

Status: PLANNED — not started. Written 2026-07-22 for a later session.

## Why it works here

- The battle recorder already does real offscreen RGB rendering
  (`battle_tournament.py --record`, RGB annotator → ffmpeg). Head-mounting
  is a camera-prim change, not new infrastructure.
- The battle env builds the opponent obs as a distinct block (opponent
  body pos/vel in the agent's frame) — easy to log per step.
- The league snapshot pool provides diverse sparring styles for free, so
  the estimator sees varied opponent behavior without extra work.

## Key design decisions (settled in discussion, 2026-07-22)

1. **Estimate, don't end-to-end RL.** Supervised video→obs regression is
   10-100× cheaper than RL from pixels and reuses the trained champion.
2. **Recurrent core + ego-motion input.** Vision only covers the front;
   the estimator carries a hidden state (GRU) to track an out-of-view
   opponent. CRITICAL: feed the agent's own proprioception / head-pose
   delta each step — remembered opponent position lives in the agent's
   body frame and must counter-rotate when the agent turns. Without
   ego-motion input the memory is useless.
3. **Uncertainty head.** Predict per-step variance alongside the estimate
   (NLL loss). Fresh sighting → tight; unseen for 2 s → wide. Enables a
   later policy to learn "turn to reacquire when uncertain."
4. **Sequence training.** Truncated BPTT over clips (e.g. 64-128 steps),
   never single frames, or the memory never learns to bridge occlusions.

## Phase 1 — Head camera + data logger

- Attach a `TiledCamera` to the `Head` body of each SOMA fighter
  (IsaacLab supports per-env tiled cameras). Forward-facing, FOV ~90-110°,
  resolution 128×128 RGB (raise later only if needed).
- Extend the battle env / a dedicated collection script to log, per step:
  - camera frame,
  - the agent's opponent-obs block (regression target),
  - the agent's proprioception + head pose (estimator input),
  - opponent-visible flag (opponent root within camera frustum) — not a
    training input, but invaluable for analysis (error vs time-unseen).
- Collection run: 64-128 parallel headless bouts, league snapshots vs the
  pool (both fighters log — 2 samples per env-step). 30 Hz. One overnight
  run ≈ millions of aligned pairs.
- Storage: shard to disk as npz/webdataset chunks; 128×128×3 @ 30 Hz
  compresses fine with per-chunk zstd or jpg. Budget ~100-300 GB.
- Sanity check (validate-one rule): record ONE bout with video playback +
  obs overlay and eyeball alignment before the mass run.

## Phase 2 — Estimator network

- Architecture: small CNN (or a pretrained frozen backbone + trainable
  head — try simple first) → concat ego-motion features → GRU (1-2 layers,
  256-512) → MLP → (opponent obs vector, log-variance vector).
- Loss: Gaussian NLL (regression + uncertainty). Optionally weight the
  root-position dims up — they matter most tactically.
- Curriculum detail: reset hidden state at episode starts; sample training
  windows that include occlusion stretches (use the visible-flag to
  oversample hard windows).
- Eval metrics: obs error split by "opponent visible" vs "unseen for N
  steps"; calibration of the variance head. Target: near-truth when
  visible, graceful drift (not divergence) up to ~2 s unseen.
- Cheap to train (supervised): hours-days on the Spark, no sim needed.

## Phase 3 — Vision-driven fighter

- Plumbing: champion policy frozen; replace its ground-truth opponent obs
  block with estimator output at inference. Everything else (own-body
  obs) stays privileged.
- Expect degradation → two escalation steps:
  1. DAgger-style: roll out WITH estimated obs, supervise estimator
     against truth on the states the closed loop actually visits.
  2. Fine-tune the policy briefly on estimated obs (league or SFT stage)
     so it tolerates estimator noise; optionally give it the uncertainty
     as an extra input.
- Showcase: record a bout with the champion fighting from vision vs a
  privileged opponent; overlay the estimator's opponent belief on the
  video (predicted skeleton ghost) — great demo artifact.

## Open questions (decide when starting)

- Camera on both fighters or one? (Both doubles data for free; start both.)
- Depth channel? RGB-only keeps the door open for real cameras; add depth
  only if RGB struggles.
- Which obs exactly count as "opponent block" — pull the precise slice
  from the battle env obs builder at implementation time.
- Frame stack vs pure recurrence for short-term motion (probably 2-frame
  stack + GRU).

## Effort estimate

| Phase | Work |
|---|---|
| 1 camera + logger | 2-4 days incl. the one-bout validation |
| 2 estimator | 1-2 days code, hours-days training |
| 3 integration | ~1 day plumbing + optional fine-tune |

Prereqs: none on the Atlas track — this is SOMA-side and independent of
the Atlas tracker work. GPU-heavy only in Phase 1 (rendering) and any
Phase 3 fine-tune.
