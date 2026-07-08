# SOMA Fight Club: GPC + Tournament Self-Play Training Plan

> **Implementation status (July 2026, branch `battle`):**
>
> | Phase | Status | Where |
> |---|---|---|
> | 0 — upstream sync | **Done** | fork main merged with NVlabs upstream; GPC stack + `soma_bones_fsq` verified present |
> | 1 — dataset | **Tooling ready, data pending** | `scripts/prepare_soma_combat_dataset.sh` (BONES-SEED download + combat DCC retarget are manual inputs) |
> | 2 — tracker validation | Pending data | commands below, unchanged |
> | 3 — GPC prior | Pending data | upstream recipe, unchanged |
> | 4 — combat SFT | **Implemented** | `examples/experiments/gpc/sft_combat_prior_peft.py` + `protomotions/envs/battle/virtual_opponent.py`; launcher `scripts/train_soma_sft_combat.sh` |
> | 5 — battle env | **Implemented** | `protomotions/envs/battle/` (paired envs, hit FSM, knockdown grace, win/lose/draw, arena spawning, fall-init) |
> | 6 — league trainer | **Implemented** | `protomotions/agents/league/` (PFSP + fixes, adapter lanes, staleness cap, informed eviction, Elo, exploiter role); experiment `examples/experiments/battle/battle_league_prior_peft.py`; launcher `scripts/train_soma_battle_league.sh` |
> | 7 — eval tournament | **Implemented** | `protomotions/battle_tournament.py` + `protomotions/agents/league/tournament.py` (round-robin ladder, H2H matrix, regression gate, exhibition); launcher `scripts/run_soma_battle_tournament.sh` |
>
> Unit tests: `protomotions/tests/test_battle_components.py`,
> `protomotions/tests/test_league_pfsp.py`. End-to-end sim runs still need the
> Phase 1-3 artifacts (motion libraries + trained prior).

Goal: train two **SOMA humanoid characters** (`soma23` — the GPC paper's native character:
human proportions, 23 actuated bodies) to fight each other with natural, human-like combat
motion. The motor foundation is a **GPC (Generative Pretrained Controller)** trained on a
**combat + BONES-SEED** dataset using this repo (ProtoMotions), and the fighting behavior
is trained with **tournament-style self-play** — taking the league design from the
IsaacLabASE repo's `battle/` + `pfsp_player_pool.py` system, corrected with lessons from
AlphaStar and from the known flaws in that implementation.

**Why SOMA instead of a robot (G1/H1-2):**

- The **released FSQ tracker checkpoint**
  (`data/pretrained_models/motion_tracker/soma_bones_fsq/`) is a SOMA tracker trained on
  BONES-SEED — with SOMA as the target we can *use it as-is* instead of spending 1–2 weeks
  of GPU time training our own (Phase 2 becomes an evaluation, not a training run).
- **BONES-SEED (~142K clips) is natively SOMA-format** — the breadth dataset needs zero
  retargeting, versus the lossy AMASS→robot PyRoki pipeline.
- **Human proportions**: combat mocap retargets to SOMA nearly losslessly (human→human),
  and the full arm/torso articulation supports real boxing/martial-arts motion — no
  "kickboxing robot" compromise from short arms and reduced DOFs.
- Fighting is a sim-only application anyway; nothing here needs to survive a real robot.
  (A later G1 port is discussed in Risks §6 — the battle env and league carry over.)

Repo locations (siblings on disk):

- **This repo:** `ProtoMotions/` — motor foundation training (FSQ tracker, GPC prior, SFT,
  RLFT) plus the new battle env and league trainer built here.
- **Reference repo:** `../IsaacLabASE/` — the existing ASE/AMP battle system whose solved
  problems we port (see the Appendix).

References:

- GPC paper: *GPC: Large-Scale Generative Pretraining for Transferable Motor Control*
  (Shi, Jiang, Tessler, Peng — SIGGRAPH 2026), [arxiv.org/abs/2606.29148](https://arxiv.org/abs/2606.29148)
- AlphaStar league training: Vinyals et al., *Grandmaster level in StarCraft II using
  multi-agent reinforcement learning* (Nature 2019) — PFSP, main agents + exploiters
- ProtoMotions upstream: [github.com/NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions)
  (this checkout is a fork behind upstream — see Phase 0)
- BONES-SEED dataset: [huggingface.co/datasets/bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed)
  (~142K clips, SOMA skeleton; prep guide: `docs/source/getting_started/seed_bvh_preparation.rst`)
- IsaacLabASE's existing self-play system (what we're improving on):
  `../IsaacLabASE/source/IsaacLabASE/IsaacLabASE/tasks/direct/ase/battle/` —
  `battle_task.py`, `hrl_sp_agent.py`, `pfsp_player_pool.py`

---

## Architecture Overview

Three subsystems, trained in order:

```
1. MOTOR FOUNDATION (ProtoMotions, single SOMA, no opponent)
   combat + BONES-SEED mocap ──► FSQ tracker ──► GPC prior ──► combat SFT adapter
                                  (RELEASED —     (GPT over      (biases token
                                   tokenizer +     tokens)        choice to combat)
                                   decoder)

2. BATTLE ENVIRONMENT (greenfield: two SOMAs per env)
   paired spawning, opponent observations, hit/knockdown scoring, win/loss/draw

3. LEAGUE (tournament self-play, PPO on PEFT adapters)
   main agent + exploiters, PFSP opponent sampling, ratings, snapshot gating
```

Why this decomposition works (validated by the GPC paper and IsaacLabASE's ASE experience):

- The fight policy acts in **token space**, not joint space — every action decodes through
  the frozen FSQ decoder to a physically-executable, human-looking movement. No AMP
  discriminator needed in the self-play loop; no GAN instability; no latent-manifold holes.
- League snapshots are **PEFT adapters only** (`inference_last.ckpt` is adapter state:
  `lora`/`gamma`/`beta`/`m` tensors, <1% of the model). A 32-member league shares one frozen
  prior + decoder in memory. This replaces `pfsp_player_pool.py`'s full-model copies.
- Get-up/recovery is **in the same vocabulary** as fighting — no separate getup policy or
  HRL switching, provided the dataset includes fall/recovery clips (BONES-SEED does; we
  add combat knockdowns on top).

### Status of the pieces (verified July 2026)

| Component | Status |
|---|---|
| **SOMA FSQ tracker checkpoint** (`data/pretrained_models/motion_tracker/soma_bones_fsq/`) | **Released — we use it directly** (trained on BONES-SEED) |
| FSQ tracker training (`examples/experiments/mimic/fsq.py`) | Released (fallback only, if the released tracker's combat coverage is inadequate) |
| GPC prior training (`examples/experiments/gpc/prior.py`) | **Released** (code; no pretrained prior ckpt yet — we train our own) |
| SFT + RLFT PEFT configs (`examples/experiments/gpc/*.py`, `protomotions/agents/peft/`) | **Released** (DoRA adapters, FiLM task conditioning, KL anchor, prior-constrained nucleus sampling) |
| `soma23` robot config + assets (`protomotions/robot_configs/soma23.py`, `soma23_humanoid.xml`) | Released |
| BONES-SEED → SOMA pipeline (`data/scripts/convert_soma23_bvh_to_proto.py`) | Released (`docs/source/getting_started/seed_bvh_preparation.rst`) |
| **Two-humanoid env / self-play** | **Not in ProtoMotions — we build it (Phases 5–6)** |

---

## Phase 0 — Sync this checkout with upstream

This checkout is a fork (`Robokan/ProtoMotions`, last commit May 2026) that **predates all
GPC code**. Sync with upstream before anything else:

```bash
git remote add upstream https://github.com/NVlabs/ProtoMotions.git
git fetch upstream
git checkout main
git merge upstream/main        # or rebase local changes; resolve as needed
git lfs pull                   # pull checkpoint/config LFS objects — includes soma_bones_fsq
```

Sanity check that the GPC stack and the SOMA pieces exist afterward:

```bash
ls examples/experiments/gpc/           # prior.py, sft_target_prior_peft.py, task_*_prior_peft*.py
ls protomotions/agents/peft/           # actor.py, adapters.py, prior_agent.py, sft_agent.py, ...
ls data/pretrained_models/motion_tracker/soma_bones_fsq/   # last.ckpt, inference_last.ckpt, ...
ls protomotions/robot_configs/soma23.py
```

Read `docs/source/user_guide/gpc.rst` — it is the canonical reference for the
tracker → prior → SFT → RLFT flow and the checkpoint contract
(`last.ckpt` = full training state, `inference_last.ckpt` = slim adapter-only artifact).
Its examples are written for `soma23`, so the commands below match the doc almost verbatim.

Environment setup: use **uv** (the recommended tool for the IsaacLab backend — see
`docs/source/getting_started/installation.rst`; conda is only suggested for
IsaacGym/Genesis/MuJoCo backends). IsaacLab 2.x requires Python 3.11:

```bash
uv venv --python 3.11 env_isaaclab
source env_isaaclab/bin/activate
uv pip install torch==2.7.0 torchvision==0.22.0
uv pip install "isaaclab[isaacsim,all]==2.3.0" --extra-index-url https://pypi.nvidia.com
uv pip install -e .
uv pip install -r requirements_isaaclab.txt
```

Simulator backend: `isaaclab` — the natural choice given the IsaacLabASE experience.
(No PyRoki environment needed — SOMA requires no robot retargeting.)

---

## Phase 1 — Build the combat + BONES-SEED SOMA dataset

Target artifacts:

- `soma_combat_seed.pt` — combined MotionLib (or sharded chunks) with sampling weights
  biased toward combat. Used for prior training (Phase 3).
- `soma_combat_only.pt` — combat clips only. Used for tracker coverage eval (Phase 2)
  and combat SFT (Phase 4).

### 1a. BONES-SEED base (breadth: locomotion, falls, recovery, athletics, martial arts)

BONES-SEED is natively SOMA — no retargeting, just format conversion. Follow
`docs/source/getting_started/seed_bvh_preparation.rst` (use the **SOMA Uniform** BVH
variant — it matches the single `soma23_humanoid.xml` model):

```bash
# BONES-SEED BVH (77 joints, 120 fps, Y-up) → proto .motion (23 bodies, 30 fps, Z-up)
python data/scripts/convert_soma23_bvh_to_proto.py \
    --input-dir /path/to/bones-seed/soma_uniform/bvh \
    --output-dir /path/to/motions/seed \
    --input-fps 120 --output-fps 30
```

Curation: the full set is ~142K clips — more than the prior needs and enough to demand
sharded MotionLibs (`chunk_slurmrank.pt` pattern; see the prep doc's "Scaling Up"
section). Two workable strategies:

- **Subset (recommended to start):** select a few thousand clips covering locomotion,
  falls/get-up, sports/athletics, and any martial-arts/stunt categories, packaged as a
  single `.pt`. Since the *tracker* was already trained on full BONES-SEED, the prior's
  dataset controls behavior distribution, not skill existence — a curated subset is fine.
- **Full set, sharded:** only if prior quality on the subset disappoints.

### 1b. Our combat mocap → SOMA

Source: the Reallusion/Unreal combat clips in IsaacLabASE
(`../IsaacLabASE/source/IsaacLabASE/ase/poselib/data/animations/amp/combat/`,
`.../reallusion_combat/`). These are human-skeleton clips, so this is a
**human→human retarget** — far easier and less lossy than the old →G1 pipeline.
Two routes, in order of preference:

1. **BVH route:** export/convert the combat clips to BVH (Blender imports the FBX
   sources; IsaacLabASE's poselib `.npy` motions can also be written back out), retarget
   the skeleton to the 77-joint SOMA BVH convention in a DCC tool (Blender +
   auto-rig/Rokoko retargeter — standard humanoid bone mapping), then run
   `convert_soma23_bvh_to_proto.py` exactly as in 1a. The converter's T-pose offsets
   (`data/soma/standard_t_pose_global_offsets_rots.p`) handle the rest.
2. **SMPL route:** fit the clips to SMPL (or export via a SMPL-compatible tool) and use
   the SOMA↔SMPL conversion scripts (`data/scripts/convert_soma23_*`) as reference for
   the mapping.

Include in this pass: strikes, blocks, dodges, footwork, knockdowns, and **get-up clips**
(critical — see Phase 3/4 acceptance). Unarmed only for now: `soma23_humanoid.xml` has no
weapon bodies, so sword/shield clips are out of scope (revisit only if a prop-extended
SOMA model is built later).

### 1c. Package the libraries

```bash
# Combined (seed + combat subdirs under one root):
python protomotions/components/motion_lib.py \
    --motion-path /path/to/motions/ \
    --output-file data/soma_combat_seed.pt --device cpu

# Combat-only:
python protomotions/components/motion_lib.py \
    --motion-path /path/to/motions/combat/ \
    --output-file data/soma_combat_only.pt --device cpu
```

Keep IsaacLabASE's curation discipline (its `animation info` / `bad motions` logs):
maintain a rejection log; garbage clips poison prior training. The SEED converter's
built-in quality filter (velocity/underground/airborne checks) helps but won't catch
retargeting artifacts in the combat clips.

**Milestone check:** visually inspect retargeted combat clips with
`examples/motion_libs_visualizer.py --robot soma23` and kinematic playback
(`examples/env_kinematic_playback.py`) before spending GPU time on training.

---

## Phase 2 — Validate the released FSQ tracker (train only if it fails)

The big win of the SOMA choice: **the tokenizer + decoder already exist.**
`data/pretrained_models/motion_tracker/soma_bones_fsq/` is an FSQ tracker trained on
BONES-SEED. Phase 2 is an *evaluation*, not a training run.

### 2a. Coverage test on the combat clips

Run the tracker evaluator against the combat-only library
(`inference_agent.py --full-eval`, then `scripts/analyze_mimic_most_failed_motions.py`):

```bash
python protomotions/inference_agent.py \
    --robot-name soma23 \
    --simulator isaaclab \
    --motion-file data/soma_combat_only.pt \
    --checkpoint data/pretrained_models/motion_tracker/soma_bones_fsq/last.ckpt \
    --full-eval
```

**Acceptance criteria:**

- Per-clip tracking success (joint pos error < 0.5 m criterion) on **every**
  strike/dodge/knockdown/get-up family. A failed clip means those skills will not exist
  as tokens for anything downstream.
- Expect this to mostly pass: BONES-SEED is large and diverse (dynamic/stunt motion
  included), and FSQ vocabularies generalize across motions of the skeleton they were
  trained on. Isolated failures usually indicate retargeting artifacts — fix the clip
  first, not the tracker.

### 2b. Fallback: fine-tune the tracker (only if coverage gaps are real)

If genuine combat skills fail to track (not fixable by re-retargeting), fine-tune from
the released checkpoint via `examples/experiments/mimic/fsq.py` on
combat + a BONES-SEED replay subset (replay prevents catastrophic forgetting):

```bash
python protomotions/train_agent.py \
    --robot-name soma23 \
    --simulator isaaclab \
    --motion-file data/soma_combat_seed.pt \
    --experiment-path examples/experiments/mimic/fsq.py \
    --checkpoint data/pretrained_models/motion_tracker/soma_bones_fsq/last.ckpt \
    --num-envs 4096 --batch-size 16384 --ngpu 3 \
    --experiment-name soma_fsq_combat_ft
```

**Caveat — do this only if forced:** fine-tuning the tracker changes token semantics.
Everything downstream (prior, SFT, league) must then be trained against *our* tracker,
and if NVIDIA later releases a pretrained SOMA GPC **prior**, it will align with *their*
frozen tracker, not ours. Keeping the tracker frozen preserves the option to swap in or
warm-start from that prior when it ships.

---

## Phase 3 — Train the GPC prior

Straight from the upstream recipe (`examples/experiments/gpc/prior.py`): a 6-layer,
d=1024 causal transformer trained with cross-entropy to predict the frozen tracker's FSQ
tokens (grouped 5 scalars/token → 8 tokens/step) from `max_coords_obs` context. Expert
rollouts come from the frozen tracker, so this is supervised — stable and much cheaper
than tracker training. Train on the combined library with combat upweighted:

```bash
python protomotions/train_agent.py \
    --robot-name soma23 \
    --simulator isaaclab \
    --motion-file data/soma_combat_seed.pt \
    --experiment-path examples/experiments/gpc/prior.py \
    --tracker-checkpoint data/pretrained_models/motion_tracker/soma_bones_fsq/last.ckpt \
    --num-envs 1024 --batch-size 1024 \
    --experiment-name soma_gpc_prior
```

**Acceptance criteria:** run the prior unconditionally (inference on the prior checkpoint)
and confirm (a) stable, natural locomotion and idling, (b) **emergent get-up** — push the
character over (`J` key applies forces in `inference_agent.py`) and verify it recovers. If
recovery is unreliable, the dataset needs more fall/get-up clips → loop back to Phase 1.

Notes:

- The prior checkpoint embeds the tracker decoder (`latent_decoder`), so downstream
  phases only need the prior checkpoint.
- If NVIDIA ships a pretrained SOMA GPC prior later **and** we kept the tracker frozen
  (Phase 2b avoided), that prior is drop-in compatible: either replace ours outright, or
  use it as the warm-start and run only a short combat-upweighting pass.

---

## Phase 4 — Combat SFT (bias the prior toward fighting)

Adapt `examples/experiments/gpc/sft_target_prior_peft.py`. SFT trains a DoRA-style PEFT
adapter with cross-entropy against the frozen tracker-encoder's tokens **on combat clips
only**, with the task observation coming from the same factory RLFT will use — keeping the
SFT data path aligned with the later fight-training path (this is the upstream design
intent; preserve it).

Changes from the stock config:

- `--motion-file data/soma_combat_only.pt` (combat clips, weighted toward strikes).
- Replace the target-reaching task obs with a placeholder **opponent observation** derived
  from the reference clips (e.g., a virtual opponent position where the clip's strikes are
  aimed, plus jitter — analogous to how the stock SFT jitters a future root-XY target).
  Simplest viable version: opponent = point in front of the character at strike-appropriate
  range.

```bash
python protomotions/train_agent.py \
    --robot-name soma23 \
    --simulator isaaclab \
    --motion-file data/soma_combat_only.pt \
    --experiment-path examples/experiments/gpc/sft_combat_prior_peft.py \
    --prior-checkpoint results/soma_gpc_prior/last.ckpt \
    --tracker-checkpoint data/pretrained_models/motion_tracker/soma_bones_fsq/last.ckpt \
    --num-envs 1024 --batch-size 1024 \
    --experiment-name soma_sft_combat
```

**Acceptance criteria:** sampled rollouts show shadow-boxing-like behavior (strikes,
guard, footwork) while retaining balance and recovery. This SFT checkpoint warm-starts
every league member in Phase 6.

---

## Phase 5 — Two-SOMA battle environment (greenfield)

ProtoMotions is strictly single-character: `BaseEnv` + all simulator backends spawn **one
humanoid per parallel env**, with no multi-agent/opponent support anywhere. This phase is
the main engineering lift. Port the game design from IsaacLabASE's
`battle/battle_task.py`; build the env here so it composes with the GPC agents.

### 5a. Paired spawning

Extend the simulator layer (start with `isaaclab` backend) so each logical *match* owns
two SOMA articulations in a shared arena. Two viable layouts:

1. **Two characters per env instance** — cleanest physics (they naturally collide),
   requires the simulator config to instantiate two articulation views and the env to
   expose both. (This is what IsaacLabASE's `battle_task.py` does: a second `Articulation`
   (`robot_op`) in the same env prim.)
2. **Ego/opponent as env pairs** (IsaacLabASE's agent-side batching approach: env `i` and
   env `i + num_actors` form a match; obs tensor is `[ego_batch; opp_batch]`) — reuses more
   of the single-character plumbing; the batching trick in `hrl_sp_agent.env_step` shows how
   actions for both sides go through one `vec_env.step`.

Recommendation: layout 2 with paired-env collision groups (assign collision groups so env
pairs collide, co-locate their spawn origins) — it keeps every ProtoMotions component
(obs kernels, PD action processing, motion manager) operating on flat batches.

### 5b. Observations

New observation component `opponent_obs` (register alongside
`protomotions/envs/obs/*`, wired like `steering_obs_factory` is in
`examples/experiments/gpc/task_steering_headvel_prior_peft.py`):

- Opponent root position/orientation/velocity in ego local frame
- Opponent key-body positions (hands, feet, head) in ego frame — needed to read strikes
- Own + opponent "stamina"/hit-state scalars (see 5c)
- Time remaining in the round

This becomes `task_obs` = `in_keys` of the PEFT actor; the frozen prior keeps consuming
`max_coords_obs` (its context keys are auto-discovered from the checkpoint — see
`gpc.rst`, "PEFT Config Contract").

### 5c. Scoring, rewards, terminations (port from `battle_task.py`)

Port the mechanics that already work in IsaacLabASE's battles:

- **Hit detection:** contact sensors on fists/feet vs opponent body regions; score by
  impact energy with per-body-region multipliers (head > torso > limbs), as in
  `BodyHitState`. Require minimum relative velocity to count (prevents "pushing" exploits).
- **Stamina/health:** accumulated hit energy depletes health; knockdown = torso/head below
  height threshold for N steps, or health exhausted.
- **Win/lose/draw:** knockdown → win/lose; timeout → draw (or points decision on
  health difference — reduces the draw-stalemate problem IsaacLabASE fights with its
  draw-rate filtering).
- **Reward shaping (keep it thin — the prior does the style work):**
  - Sparse: +1 win, −1 loss, 0 draw (zero-sum core)
  - Dense (small weights): hit-landed energy, hit-received penalty, facing/range shaping
    early in training (anneal to near-zero as the league matures — AlphaStar lesson:
    dense shaping helps bootstrap but caps strategy diversity if left on)
  - **No AMP/style reward needed** — naturalness comes from prior-constrained sampling.
    Keep `task_*_prior_peft_amp.py` as a fallback if fights drift ugly with high KL budgets.
- **Terminations:** knockdown (grace period for get-up attempts — don't end the episode on
  first fall; give ~2 s to recover, which makes get-up tokens *tactically valuable*),
  arena out-of-bounds, timeout.

### 5d. Episode/reset logic

- Spawn at randomized ranges/angles (vary engagement distance)
- Reference-state init from combat clips for a fraction of resets (Phase 1 data), fall-state
  init for a fraction (drives get-up robustness — port the idea from IsaacLabASE's
  `AmpGetupEnv`: `recovery_episode_prob`/`fall_init_prob`)

**Milestone check:** two scripted/random-token SOMAs in an arena, hits detected and scored
correctly, resets stable at 1000+ parallel matches. Validate hit plausibility visually.

---

## Phase 6 — League training (tournament self-play, done correctly)

The trainer is a new agent: `DiscretePriorPEFTRLFTAgent` (PPO over token logits, KL to
anchor, prior-constrained nucleus sampling) extended with a league. Port the *structure* of
IsaacLabASE's `hrl_sp_agent.py` + `pfsp_player_pool.py`, with these corrections.

### 6a. What we keep from IsaacLabASE (it was right)

- **PFSP opponent sampling** with the standard weightings
  (`variance: x(1-x)`, `linear: 1-x`, `squared: (1-x)^2`) over per-opponent win rates
- **EMA-decayed per-opponent win/loss/draw counters** (stats track the *current* agent)
- **Per-env opponent assignment**: on episode end, that env samples a new opponent —
  the agent fights the whole league simultaneously across the batch
- **Vectorized opponent inference** — now even better: all league members share the frozen
  prior + decoder; a snapshot is just an adapter (`inference_last.ckpt` state). Batch all
  opponents' token decoding through one prior forward with per-env adapter selection
  (batched LoRA/DoRA: gather adapter weights by env index; this is standard multi-LoRA
  serving practice).
- **Match accounting on the env side** (win/lose/draw flags in `infos`, as `battle_task.py`
  does)

### 6b. What we fix (bugs and design flaws found in IsaacLabASE's implementation)

1. **Dead snapshot gate** — in `hrl_sp_agent.check_update_opponent`, `force_add` is
   computed but unused (`if True:` follows). Consequence: if the agent plateaus below
   `update_win_rate`, the league silently stops growing. Fix: hard staleness cap — if no
   snapshot has been added in K epochs, add one regardless of win rate.
2. **FIFO ring-buffer eviction** — the current pool evicts oldest members regardless of
   usefulness. Fix: evict by *least informative* (lowest PFSP sampling weight over a
   trailing window), and always retain a small protected set: the earliest snapshot
   (anti-cycling canary) and the highest-rated member.
3. **Inconsistent draw weighting** — draws count 0.25 in `win_rate`, 0.5 in
   `conservative_score` and current-policy win rate. Fix: one constant, one place
   (recommend 0.5 = standard game-theoretic value), and make stalemate discouragement
   explicit in the reward (points decision on timeout) instead of hidden in statistics.
4. **Win-rate-only ratings.** Add a proper rating system updated online from training
   matches: Elo is fine to start (`multielo`, as IsaacLabASE uses in eval);
   TrueSkill if draws stay frequent. Ratings serve monitoring and eval seeding — PFSP
   sampling stays driven by win rate vs. the *current* agent (that part AlphaStar also did).

### 6c. What we add from AlphaStar (the "done correctly" part)

- **League roles.** Beyond the main agent's historical snapshots:
  - **Main exploiter** (1 seat): trains *only* against the current main agent, from the SFT
    checkpoint, reset after each successful exploit cycle. Finds the main agent's specific
    holes fast.
  - **League exploiter** (1 seat, optional at first): trains against the whole league via
    PFSP; finds systemic weaknesses. Snapshot into the league when it beats most members.
  - IsaacLabASE's "10% revive a random old checkpoint" hack is superseded by exploiters +
    smarter eviction, but is cheap to keep as well.
- **Snapshot cadence:** gate on pool-average win rate (as now) **plus** the staleness cap
  (6b.1). On snapshot: freeze adapter to disk (`policy_dir/`), add to league, reset
  per-opponent stats, resample all envs (as `restore_opponents_to_resume` does — keep the
  resumability, it pairs well with restart-loop launcher scripts).
- **PFSP weighting schedule:** start `linear` (uniform-ish pressure), move to `variance`
  (focus on ~50% opponents) as the league matures.
- **KL/prior budget per role:** main agent keeps meaningful KL to the SFT anchor (fights
  stay human); exploiters get a looser budget (their job is finding holes, not looking
  pretty; they never ship as the final artifact).

### 6d. Training loop mechanics

- Both sides of every match run token policies; ego side collects PPO experience,
  opponent side is inference-only (frozen adapters). A `force_symmetric_inference`-style
  debug switch is worth porting for validating env symmetry.
- Zero-sum symmetry: optionally collect experience from *both* sides when both are the
  main agent (self-mirror matches) — doubles sample efficiency for the cost of correlation.
- **Throughput budget (the honest cost):** 8 sequential transformer decodes per agent per
  control step, two agents per match. Mitigations, in order of value:
  1. batch token decoding across all envs (the sequential dimension is only within-step),
  2. single shared backbone + batched per-env adapters (multi-LoRA),
  3. bf16 inference for opponents,
  4. if still bound: hold sampled tokens for 2 control steps (validate against
     per-step sampling first — this deviates from pretraining).
  Expect fewer parallel envs than the ASE battles (thousands → high hundreds/low
  thousands); GPC's sample-efficiency from the pretrained prior should compensate.

**Acceptance criteria:** rating of the main agent rises over league generations;
periodic eval vs. frozen early snapshots shows monotone improvement; fights remain
human-plausible (spot-check videos every N generations); get-up used under pressure.

---

## Phase 7 — Inference & evaluation tournament system

Separate from training (the analog of IsaacLabASE's `play_battle.py` / `hrl_sp_player.py`):

1. **Round-robin evaluator:** load N adapter checkpoints, run a full round-robin (or Swiss)
   schedule across parallel arenas, M matches per pairing with randomized spawns; update
   Elo/TrueSkill from outcomes (`multielo` handles multiplayer Elo updates). Output a
   ladder table + head-to-head matrix (exposes non-transitivity — the interesting result).
2. **Exhibition mode:** two chosen checkpoints, one arena, viewer on, deterministic
   (temperature↓ / top-p↓) or stochastic sampling; camera follow; record video
   (`scripts/create_video.sh`). This is the demo artifact.
3. **Regression gate:** before a snapshot enters the league (Phase 6), it must beat the
   previous snapshot ≥55% over a fixed match count in *this* evaluator — decoupling
   "gate" statistics from noisy training-time counters (a correctness fix over IsaacLabASE,
   which gates on training stats only).
4. **Human-in-the-loop (stretch):** gamepad-driven opponent — map sticks/buttons to task
   obs overrides; IsaacLabASE's `utils/xbox_controller.py` experience applies directly.

All evaluation runs on slim `inference_last.ckpt` adapter artifacts + the one shared prior
checkpoint (see `gpc.rst` "Checkpoint Roles": keep `resolved_configs_inference.pt` beside
each adapter; override `agent.pretrained_modules.prior.checkpoint_path` when relocating).

---

## Compute & timeline (3-GPU workstation, rough)

| Phase | Work | Wall-clock estimate |
|---|---|---|
| 0 | Sync fork, env setup | days |
| 1 | Dataset build (SEED conversion + combat retarget + curation) | ~1 week (mostly human time; no PyRoki pipeline) |
| 2 | Tracker **evaluation** (released ckpt) | ~1 day GPU (fallback fine-tune: +3–5 days, only if needed) |
| 3 | GPC prior | 2–4 days GPU |
| 4 | Combat SFT | ~1 day GPU |
| 5 | Battle env | 2–4 weeks engineering |
| 6 | League training | 2+ weeks GPU (open-ended — leagues improve as long as you run them) |
| 7 | Eval tournament | ~1 week engineering |

Using SOMA removes the plan's former longest GPU run (1–2 weeks of from-scratch tracker
training) and the AMASS→robot retargeting pipeline entirely. Phases 2–4 are sequential;
Phase 5 can proceed in parallel with 2–4 (it only needs the tracker/prior at integration
time, and can be developed against random-token policies).

---

## Risks & open questions

1. **Combat token coverage** (top risk, but much reduced vs. a robot target): if the
   released tracker can't track a retargeted strike, that skill doesn't exist downstream.
   Mitigation: Phase 2a per-clip acceptance test; BONES-SEED's dynamic/stunt coverage and
   the human skeleton make broad failures unlikely; isolated failures are usually
   retargeting artifacts — fix the clip before touching the tracker (Phase 2b caveat).
2. **Tracker freeze vs. fine-tune:** fine-tuning the tracker (Phase 2b) forks our token
   vocabulary away from NVIDIA's, forfeiting compatibility with any future released SOMA
   prior. Default to frozen; treat 2b as a last resort.
3. **Self-play stability:** leagues can still cycle or collapse to stalling. The
   staleness cap, exploiters, points-decision on timeout, and the head-to-head eval matrix
   are the countermeasures; watch draw rate as the leading indicator.
4. **Prior constraint vs. peak skill:** prior-constrained nucleus sampling keeps fights
   human but caps off-manifold exploits. If the league plateaus, experiment with
   `--peft-sampling-mode nucleus` (student nucleus + KL) on exploiter seats first.
5. **Upstream churn:** GPC code landed recently; expect API movement. Pin a known-good
   upstream commit; keep our battle env additions in clearly separated modules
   (`protomotions/envs/battle/`, `examples/experiments/battle/`) to ease rebasing.
   Upside of the SOMA choice: if NVIDIA ships a SOMA GPC prior checkpoint, it slots
   directly into our pipeline (see Phase 3 notes) instead of being wasted on us.
6. **No hardware path (by design):** SOMA is a simulation character; there is no robot to
   deploy to. If a real-robot fight (e.g., G1) is wanted later, the battle environment,
   league trainer, and evaluation system all carry over unchanged — only the motor
   foundation is embodiment-specific: retarget the dataset to G1 (PyRoki pipeline), train
   a G1 FSQ tracker with the hardware-robustness pieces from
   `data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py`
   (BeyondMimic-style observations, L2C2 smoothness, action-rate penalties, domain
   randomization), then rerun Phases 3–4. The SOMA league also provides trained opponents
   and a behavior reference for that effort.

---

## Appendix — Problems already solved in IsaacLabASE (reuse these, don't re-derive them)

IsaacLabASE's battle system took real debugging time to get right. Most of that work
transfers to ProtoMotions as either directly portable code, a template to translate, or a
documented trap to avoid. Unless otherwise noted, short file names below live under
`../IsaacLabASE/source/IsaacLabASE/IsaacLabASE/tasks/direct/ase/` (battle files in its
`battle/` subdirectory). Grouped by the phase where it saves time:

### Battle environment (saves time in Phase 5)

| Problem | Where it's solved | How to reuse in ProtoMotions |
|---|---|---|
| **Two colliding characters in Isaac Lab** | `battle_task.py::_setup_scene` — second `Articulation` (`robot_op`) registered in the same env prim, so PhysX collides the pair naturally; per-robot contact + damage sensors registered alongside | Proof that the two-articulations-per-env route works in Isaac Lab. If Phase 5 instead uses paired-env collision groups, this code is still the reference for sensor wiring |
| **Action timing bias** | `battle_task.py` (~line 582): opponent actions are applied *before* ego actions, explicitly "to eliminate timing bias" | Port the convention as-is. This is an easy-to-miss asymmetry that silently favors one side and corrupts self-play statistics |
| **Hit detection & scoring** | `BodyHitState` (`battle_task.py` ~line 33): contact forces from filtered sensors, per-body-region damage multipliers, weapon-orientation multipliers, minimum-force thresholds | Translate to a ProtoMotions reward component. The damage-multiplier tables and force thresholds are tuned values — start from them, don't re-tune from scratch |
| **Death/knockdown handling** | `death_timer` + `death_interval` logic: a downed character isn't insta-reset; a timer runs (get-up window), and a normalized `died_observation` tells both policies about it | This is exactly the "knockdown grace period" Phase 5c calls for — the state machine already exists, including exposing death state to the opponent's observations |
| **Opponent observations that don't explode** | `battle_task.py` obs builders (~lines 1884–1957): opponent body pos/vel in ego heading frame, with explicit clamping (`±2× arena border`, `±50 m/s`) on every opponent-derived term | Port the frame math and the clamps. The clamps exist because ragdolling opponents produce velocity spikes that destabilize value functions — a lesson that cost training runs |
| **Arena placement & spawn randomization** | Reset logic (~lines 1239–1273): spawn placement relative to arena center vs. relative to opponent, randomized range/angle | Template for Phase 5d spawn logic |
| **Anti-stalling** | `idle_time` tracking + fitness penalties in `battle_task.py` | Port alongside the points-decision-on-timeout rule; stalling is the dominant failure mode of fight self-play |
| **Fall-state initialization curriculum** | `amp_getup_env.py`: `recovery_episode_prob`, `fall_init_prob`, `recovery_steps` — mixing fallen/recovering starts into resets | Reimplement as reset options in the battle motion manager; this is what makes get-up tokens reliably used under pressure |

### Self-play training loop (saves time in Phase 6)

| Problem | Where it's solved | How to reuse |
|---|---|---|
| **Ego/opponent batch split over a single vec env** | `hrl_sp_agent.py::env_step` / `env_reset`: obs stacked `[ego; op]`, sliced by `num_actors`; both sides' actions concatenated into one `vec_env.step` | The core data-flow pattern for hosting self-play on a framework that assumes one policy. Translates directly to a ProtoMotions agent subclass |
| **Match-level done semantics** | `hrl_sp_agent.play_steps`: `dones.view(num_actors, num_agents).all(dim=1)` — an env pair is "done" only when the match is, with win/lose/draw flags accumulated across LLC substeps and binarized | Subtle and easy to get wrong; port the accounting pattern (including the timeout/value-bootstrap interaction) |
| **Opponent resampling on episode end** | `hrl_sp_agent.resample_op` + `SinglePlayer.add_envs/remove_envs/reset_envs`: per-env opponent assignment via boolean masks → compact index tensors | Port nearly verbatim — this machinery is framework-agnostic tensor bookkeeping |
| **Batched league inference** | `PFSPPlayerVectorizedPool` + `vectorized_network_builder.py`: all pool members stacked into one batched model, one forward per step, per-env routing | The *concept* ports; the implementation gets simpler here because snapshots are PEFT adapters over one shared frozen prior (batched multi-LoRA gather instead of stacked full models) |
| **PFSP statistics done carefully** | `SinglePlayer` (`pfsp_player_pool.py`): EMA-decayed win/loss/draw counters with a half-life, Beta-prior `conservative_score`, minimum-decisive-ratio filter before an opponent is eligible for weighted sampling | Port the statistics layer as-is — it's plain Python/torch with no framework coupling, and it encodes several stalemate-related lessons |
| **League persistence & resume** | `hrl_sp_agent`: snapshots saved to `policy_dir/`, `restore_opponents_to_resume` rebuilds the pool (most recent `max_length` by ctime) on restart; paired with restart-loop launcher scripts (`../IsaacLabASE/scripts/build/humanoid/train_hrl_battle.sh`) | Long league runs *will* crash; resumability was retrofitted there after pain. Build it into the ProtoMotions league agent from day one |
| **Symmetry debug switch** | `force_symmetric_inference` flag: both sides run the identical policy through the identical inference path | Port it. It's the fastest way to catch env asymmetries (timing, obs frames, spawn bias): with identical policies, win rate must be ~50% |
| **Live LLC/prior hot-reload** | `hrl_agent.reload_llc_network_if_needed()` (in `../IsaacLabASE/source/IsaacLabASE/ase/learning/hrl_agent.py`, checked every 5 epochs) | Optional, but useful if the prior gets refined while league training runs |

### Evaluation & tooling (saves time in Phase 7)

| Problem | Where it's solved | How to reuse |
|---|---|---|
| **Checkpoint-vs-checkpoint evaluation** | `../IsaacLabASE/scripts/rl_games_amp/play_battle.py` (`--checkpoint`, `--op_checkpoint`, `--player_pool_type`) + `battle/hrl_sp_player.py` with `MultiElo` rating updates | Structure of the round-robin evaluator: match scheduling, outcome aggregation, Elo updates already worked out |
| **Human-in-the-loop control** | `../IsaacLabASE/source/IsaacLabASE/utils/xbox_controller.py` (+ its use in game-controller tasks) | Drop-in for the Phase 7 exhibition/human-opponent mode |
| **Spectator tooling** | `../IsaacLabASE/source/IsaacLabASE/utils/keyboard_viewport_camera_tracker.py` (camera follow, debug-vis hotkeys, motion capture key) | Template for viewer QoL during exhibitions and debugging |
| **Motion curation discipline** | `../IsaacLabASE/animation info` / `bad motions` logs; per-clip weighted YAML manifests; view-motion playback tasks | The *workflow* for Phase 1: inspect every retargeted clip, keep a rejection log, weight sampling per clip |

### Known traps (negative knowledge — cheaper than re-discovering)

Documented in Phase 6b, listed here for completeness: the dead snapshot gate
(`force_add` computed but unused → league growth can silently stall), FIFO pool eviction
(loses informative old strategies), inconsistent draw weighting (0.25 vs 0.5 across
metrics), and gating league admission on noisy training-time counters instead of clean
eval matches. Each of these was found by inspection of the IsaacLabASE code — the
ProtoMotions implementation should treat them as review checklist items.

**Net effect on the estimate:** the Phase 5–6 work is mostly *translation* of
already-debugged logic rather than invention. The genuinely new engineering reduces to
(a) hosting the ego/opponent split inside ProtoMotions' Lightning Fabric loop, and
(b) batched adapter inference for the league — everything else above has a working
reference implementation in IsaacLabASE.

---

## Key file map

Paths relative to this repo unless prefixed with `../IsaacLabASE/`.

| Purpose | Path |
|---|---|
| **Released SOMA FSQ tracker (use as-is)** | `data/pretrained_models/motion_tracker/soma_bones_fsq/` |
| SOMA robot config + model | `protomotions/robot_configs/soma23.py`, `protomotions/data/assets/mjcf/soma23_humanoid.xml` |
| BONES-SEED BVH → proto converter | `data/scripts/convert_soma23_bvh_to_proto.py` (guide: `docs/source/getting_started/seed_bvh_preparation.rst`) |
| SOMA T-pose offsets (used by converter) | `data/soma/standard_t_pose_global_offsets_rots.p` |
| FSQ tracker experiment (fallback fine-tune only) | `examples/experiments/mimic/fsq.py` |
| GPC prior training | `examples/experiments/gpc/prior.py` |
| SFT template (fork → combat SFT) | `examples/experiments/gpc/sft_target_prior_peft.py` |
| RLFT template (fork → battle RLFT) | `examples/experiments/gpc/task_steering_headvel_prior_peft.py` |
| PEFT agent internals | `protomotions/agents/peft/` |
| GPC user guide (canonical workflow; examples use soma23) | `docs/source/user_guide/gpc.rst` |
| Battle mechanics to port | `../IsaacLabASE/source/IsaacLabASE/IsaacLabASE/tasks/direct/ase/battle/battle_task.py` |
| League/PFSP to port (and fix) | `../IsaacLabASE/source/IsaacLabASE/IsaacLabASE/tasks/direct/ase/battle/pfsp_player_pool.py`, `hrl_sp_agent.py` |
| Combat mocap source | `../IsaacLabASE/source/IsaacLabASE/ase/poselib/data/animations/amp/combat/`, `.../reallusion_combat/` |
| Getup curriculum reference | `../IsaacLabASE/source/IsaacLabASE/IsaacLabASE/tasks/direct/ase/amp_getup_env.py` |
| Eval tournament reference | `../IsaacLabASE/scripts/rl_games_amp/play_battle.py`, `.../battle/hrl_sp_player.py` |
