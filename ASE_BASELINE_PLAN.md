# ASE-vs-GPC Baseline — Experiment Plan (T800)

**Status: PLAN ONLY** (2026-07-22). Question: does the GPC stack (FSQ tracker
→ autoregressive prior → DoRA RLFT league) actually outperform the classic
ASE recipe on the same robot, same corpus, same battle rules — measured by a
head-to-head tournament?

## Why this is cheap to ask

- ASE is first-class in this repo: `protomotions/agents/ase/`
  (agent/model/config) and `examples/experiments/ase/mlp.py`, with tests.
- ASE needs **no tracker** — its AMP-style discriminator consumes body-state
  features straight from clips, so the ASE arm starts from the corpus file
  alone (no dependency on the tracker currently training).
- Both arms are the same robot (T800) in the same BattleEnv → cross-
  architecture bouts are physically trivial; only model loading differs.

## Shared, controlled inputs (both arms identical)

- **Corpus**: `data/t800_prior_corpus.pt` — 1,548 clips, combat-first by
  construction: all SEED combat tier + all Reallusion (with its 4× sampling
  multiplier), fight-support capped 25/family, locomotion capped 40/family.
  Deliberately favorable to each arm in the dimension it cares about
  (GPC: kick coverage; ASE: concentrated combat distribution).
- **Battle rules**: the current frozen ruleset — KE damage (gate 0), stun-
  gated KO, 5 s count-out, chest facing, win 500 + early-finish bonus,
  ke_reward_ref 5. Pin the exact config by generating both leagues'
  resolved configs from the same experiment-file commit.
- **Budget parity**: give each league arm the same WALL-CLOCK on the same
  GPU (not the same step count — ASE steps are ~an order of magnitude
  cheaper than 8-forward autoregressive GPC decodes; equal wall-clock is the
  fair deployment-relevant comparison; log steps too and report both).

## Arm A — GPC (mostly already in motion)

1. Finish `t800_tracker_combat_v1` (running, GPU 1, 4096 envs).
2. Train the T800 GPC prior on the corpus (`examples/experiments/gpc/prior.py`
   pattern, `--tracker-checkpoint` = new tracker).
3. Combat SFT adapter, then league RLFT (`battle_league_prior_peft.py`
   with a T800 battle body-table — strike/damage/key/head names for the
   T800 skeleton + a KE-gate calibration probe; see MULTI_ROBOT_LEAGUE_PLAN
   notes on per-robot tables).

## Arm B — ASE

1. **Pretrain** the ASE low-level on the corpus (`examples/experiments/ase/
   mlp.py` as the base; T800 robot config). Watch for the known failure:
   discriminator mode collapse at 1.5k-clip scale — log skill-space coverage
   (which clip families the discriminator actually rewards) from the start.
2. **League seat for ASE** (the real new work, est. 3–5 days):
   - A league agent variant whose snapshots are full high-level state dicts
     (small MLPs) instead of PEFT adapter slices — touches `_take_snapshot`/
     `_load_member_adapter`/lanes `model_factory` (lanes routing is already
     architecture-agnostic per member).
   - Reuse PFSP pool, gating (0.7/200), staleness, Elo unchanged.
3. Same battle env config, same GPU-class, same wall-clock budget.

## The tournament (the actual measurement)

1. **Cross-architecture loader** (small): `battle_tournament.py` currently
   loads two adapters into one ego model. Extend `--exhibition` to accept a
   *policy bundle* per side (architecture + weights, same obs/action space) —
   the same-robot easy case of the multi-robot plan's "opponent bundles".
2. **Protocol**:
   - Round-robin: GPC champion + top-4 pool vs ASE champion + top-4 pool,
     both colors, N≥50 bouts/pairing (regression-gate machinery in
     tournament.py already computes clean stats).
   - Report: head-to-head Elo, KO/count-out/points/draw split, per-hit KE
     distribution (probe), hands/legs usage, and eyeball review of recorded
     bouts (motion quality is a first-class result — record with each arm's
     natural inference decode).
3. **Secondary condition** (only if ASE mode-collapses on the full corpus):
   re-pretrain ASE on a combat-tier-only subset (~200–400 clips, its proven
   regime) and rerun the tournament — distinguishes "architecture can't
   scale" from "overfed". If ASE-at-its-best still loses to GPC-at-scale,
   the question is settled.

## What would falsify the GPC bet

ASE arm shows: comparable corpus coverage (no collapsed families), sharper
strike dynamics (higher per-hit KE tail), and ≥55% head-to-head — at a
fraction of the training compute. If ASE wins on strikes but loses on
vocabulary breadth, the interesting hybrid is ASE-style adversarial reward
as an auxiliary term inside GPC RLFT rather than an architecture switch.

## Sequencing / resources

- Blocked behind current runs: v5 league (GPU 0, ~2 days left) and the T800
  tracker (GPU 1). The ASE pretrain is the natural first tenant of whichever
  GPU frees first — it does not need the tracker.
- Order: ASE pretrain (B1) ∥ GPC prior (A2) → ASE league seat code (B2,
  CPU-side work, can overlap) → both leagues (sequential or split across
  GPUs) → tournament.
- The third 4090, if it ever returns, makes both leagues concurrent.
