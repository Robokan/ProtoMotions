# G1 Fight Club: GPC + Tournament Self-Play Training Plan

Goal: train two Unitree G1 humanoids to fight each other with natural, human-like combat
motion. The motor foundation is a **GPC (Generative Pretrained Controller)** trained on a
**combat + AMASS** dataset using this repo (ProtoMotions), and the fighting behavior is
trained with **tournament-style self-play** — taking the league design from the IsaacLabASE
repo's `battle/` + `pfsp_player_pool.py` system, corrected with lessons from AlphaStar and
from the known flaws in that implementation.

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
- IsaacLabASE's existing self-play system (what we're improving on):
  `../IsaacLabASE/source/IsaacLabASE/IsaacLabASE/tasks/direct/ase/battle/` —
  `battle_task.py`, `hrl_sp_agent.py`, `pfsp_player_pool.py`

---

## Architecture Overview

Three subsystems, trained in order:

```
1. MOTOR FOUNDATION (ProtoMotions, single G1, no opponent)
   combat + AMASS mocap ──► FSQ tracker ──► GPC prior ──► combat SFT adapter
                             (tokenizer +    (GPT over      (biases token
                              decoder)        tokens)        choice to combat)

2. BATTLE ENVIRONMENT (greenfield: two G1s per env)
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
  HRL switching, provided the dataset includes fall/recovery clips (it will).

### Status of the pieces (verified July 2026)

| Component | Status |
|---|---|
| FSQ tracker training (`examples/experiments/mimic/fsq.py`) | **Released** in upstream ProtoMotions |
| GPC prior training (`examples/experiments/gpc/prior.py`) | **Released** (code; no pretrained prior ckpt — we train our own) |
| SFT + RLFT PEFT configs (`examples/experiments/gpc/*.py`, `protomotions/agents/peft/`) | **Released** (DoRA adapters, FiLM task conditioning, KL anchor, prior-constrained nucleus sampling) |
| SOMA FSQ tracker checkpoint (`data/pretrained_models/motion_tracker/soma_bones_fsq/`) | Released (not needed — wrong skeleton; we train a G1 tracker) |
| AMASS → G1 retargeting (PyRoki pipeline) | Released |
| PHUMA G1 combat-adjacent clips (kungfu, sword) | Available (`data/yaml_files/g1_phuma_train.yaml`) |
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
git lfs pull                   # pull checkpoint/config LFS objects
```

Sanity check that the GPC stack exists afterward:

```bash
ls examples/experiments/gpc/           # prior.py, sft_target_prior_peft.py, task_*_prior_peft*.py
ls protomotions/agents/peft/           # actor.py, adapters.py, prior_agent.py, sft_agent.py, ...
ls examples/experiments/mimic/fsq.py
```

Read `docs/source/user_guide/gpc.rst` — it is the canonical reference for the
tracker → prior → SFT → RLFT flow and the checkpoint contract
(`last.ckpt` = full training state, `inference_last.ckpt` = slim adapter-only artifact).

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

PyRoki retargeting needs a **separate** environment (its dependencies conflict); a second
uv venv works — the retarget script just takes both interpreters' paths (see Phase 1a).
Simulator backend: `isaaclab` — the natural choice given the IsaacLabASE experience, and
what the GPC doc examples use.

---

## Phase 1 — Build the combat + AMASS G1 dataset

Target artifact: one packaged MotionLib file, e.g. `g1_combat_amass.pt`, containing
retargeted G1 motions with sampling weights biased toward combat.

### 1a. AMASS base (breadth: locomotion, falls, recovery, athletics)

Follow `docs/source/getting_started/amass_preparation.rst`, then retarget to G1:

```bash
# AMASS (SMPL) → proto .motion files → packaged SMPL motionlib
python data/scripts/convert_amass_to_motionlib.py /path/to/amass_root /path/to/out \
    --motion-config data/yaml_files/amass_smpl_train.yaml

# SMPL motionlib → G1 via PyRoki (two separate envs; see retargeting_pyroki.rst)
# The script takes both interpreters explicitly — point it at your two uv venvs:
./scripts/retarget_amass_to_robot.sh \
    ./env_isaaclab/bin/python \
    ./env_pyroki/bin/python \
    /path/to/out/amass_smpl_train.pt g1 1
# → proto-g1.pt
```

Curation: AMASS is ~40 h. Keep locomotion, falling, getting up, sports, dynamic motion;
consider dropping long sequences of sitting/hand-detail clips that waste tracker capacity.
`data/scripts/motion_filter.py` and `scripts/subset_motion_lib.py` help here.

### 1b. PHUMA combat-adjacent clips (already retargeted to G1)

`data/yaml_files/g1_phuma_train.yaml` includes `g1/kungfu/*` and `g1/haa500/sword_*`
entries — free combat-adjacent data, no retargeting needed:

```bash
python data/scripts/convert_phuma_to_motionlib.py /path/to/PHUMA/data /path/to/out \
    --humanoid-type g1 --motion-config data/yaml_files/g1_phuma_train.yaml
```

### 1c. Our combat mocap → G1

Source: the Reallusion/Unreal combat clips in IsaacLabASE
(`../IsaacLabASE/source/IsaacLabASE/ase/poselib/data/animations/amp/combat/`,
`.../g1/combat/` — already retargeted once to IsaacLabASE's G1 boxer via
`../IsaacLabASE/scripts/retarget_amp/retarget_combat_to_g1.py`).

ProtoMotions has **no FBX ingestion**, so the path of least resistance is through keypoints:

1. Export/convert the combat clips to SMPL-format motions (or extract 3D keypoints from the
   existing poselib `.npy` skeleton motions — write a small converter from poselib
   `SkeletonMotion` to the keypoint `.npz` format consumed by
   `pyroki/batch_retarget_to_g1_from_keypoints.py`; see
   `data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py` for the
   expected schema).
2. Retarget through PyRoki to G1 like the AMASS clips.
3. Convert to proto: `data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py`.

Include in this pass: strikes, blocks, dodges, footwork, knockdowns, and **get-up clips**
(critical — see Phase 4). No sword/shield: the G1 fights unarmed (boxer), which matches
both the hardware and the dataset.

### 1d. Package the combined library

```bash
# Put all .motion files under one root (amass/, phuma/, combat/ subdirs), then:
python protomotions/components/motion_lib.py \
    --motion-path /path/to/combined_motions/ \
    --output-file data/g1_combat_amass.pt --device cpu
```

Also build a **combat-only** library (`g1_combat_only.pt`) — used for SFT (Phase 4) and for
the tracker coverage evaluation (Phase 2). Keep IsaacLabASE's curation discipline
(its `animation info` / `bad motions` logs): maintain a rejection log; garbage clips poison
the token vocabulary.

**Milestone check:** visually inspect retargeted combat clips with
`examples/motion_libs_visualizer.py` and kinematic playback
(`examples/env_kinematic_playback.py`) before spending GPU time on training.

---

## Phase 2 — Train the G1 FSQ tracker (tokenizer + decoder)

Fork `examples/experiments/mimic/fsq.py`. It is robot-agnostic; the paper/default settings
(40 FSQ scalars × 9 levels, encoder/decoder MLPs) are a sound starting point. Consider
merging in the G1-specific robustness pieces from the production G1 tracker config
(`data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py`):
BeyondMimic-style reduced-coord observations, L2C2 smoothness, action-rate penalties —
these matter if any policy is ever to run on real hardware.

```bash
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaaclab \
    --motion-file data/g1_combat_amass.pt \
    --experiment-path examples/experiments/mimic/fsq.py \
    --num-envs 4096 --batch-size 16384 \
    --ngpu 3 \
    --experiment-name g1_fsq_combat_amass
```

**Acceptance criteria (the coverage test):**

- Overall tracking success (joint pos error < 0.5 m criterion) ≥ ~95% on the full set.
- **Per-clip success on the combat subset** — run the evaluator against `g1_combat_only.pt`
  (`inference_agent.py --full-eval`, plus `scripts/analyze_mimic_most_failed_motions.py`).
  Every strike/dodge/get-up family must track; a failed clip means those skills will not
  exist as tokens. Fix by upweighting failed clips (the `MimicEvaluatorConfig`
  motion-weight rules do this automatically) or re-retargeting them.

This is the longest single training run in the plan (order of 1–2 weeks on a 3-GPU
workstation for ~40 h of motion; the paper's AMASS-scale runs used a single A100).

---

## Phase 3 — Train the GPC prior

Straight from the upstream recipe (`examples/experiments/gpc/prior.py`): a 6-layer,
d=1024 causal transformer trained with cross-entropy to predict the frozen tracker's FSQ
tokens (grouped 5 scalars/token → 8 tokens/step) from `max_coords_obs` context. Expert
rollouts come from the frozen tracker, so this is supervised — stable and much cheaper than
Phase 2.

```bash
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaaclab \
    --motion-file data/g1_combat_amass.pt \
    --experiment-path examples/experiments/gpc/prior.py \
    --tracker-checkpoint results/g1_fsq_combat_amass/last.ckpt \
    --num-envs 1024 --batch-size 1024 \
    --experiment-name g1_gpc_prior
```

**Acceptance criteria:** run the prior unconditionally (inference on the prior checkpoint)
and confirm (a) stable, natural locomotion and idling, (b) **emergent get-up** — push the
robot over (`J` key applies forces in `inference_agent.py`) and verify it recovers. If
recovery is unreliable, the dataset needs more fall/get-up clips → loop back to Phase 1.

Note: the prior checkpoint embeds the tracker decoder (`latent_decoder`), so downstream
phases only need the prior checkpoint.

---

## Phase 4 — Combat SFT (bias the prior toward fighting)

Adapt `examples/experiments/gpc/sft_target_prior_peft.py`. SFT trains a DoRA-style PEFT
adapter with cross-entropy against the frozen tracker-encoder's tokens **on combat clips
only**, with the task observation coming from the same factory RLFT will use — keeping the
SFT data path aligned with the later fight-training path (this is the upstream design
intent; preserve it).

Changes from the stock config:

- `--motion-file data/g1_combat_only.pt` (combat clips, weighted toward strikes).
- Replace the target-reaching task obs with a placeholder **opponent observation** derived
  from the reference clips (e.g., a virtual opponent position where the clip's strikes are
  aimed, plus jitter — analogous to how the stock SFT jitters a future root-XY target).
  Simplest viable version: opponent = point in front of the character at strike-appropriate
  range.

```bash
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaaclab \
    --motion-file data/g1_combat_only.pt \
    --experiment-path examples/experiments/gpc/sft_combat_prior_peft.py \
    --prior-checkpoint results/g1_gpc_prior/last.ckpt \
    --tracker-checkpoint results/g1_fsq_combat_amass/last.ckpt \
    --num-envs 1024 --batch-size 1024 \
    --experiment-name g1_sft_combat
```

**Acceptance criteria:** sampled rollouts show shadow-boxing-like behavior (strikes,
guard, footwork) while retaining balance and recovery. This SFT checkpoint warm-starts
every league member in Phase 6.

---

## Phase 5 — Two-G1 battle environment (greenfield)

ProtoMotions is strictly single-character: `BaseEnv` + all simulator backends spawn **one
humanoid per parallel env**, with no multi-agent/opponent support anywhere. This phase is
the main engineering lift. Port the game design from IsaacLabASE's
`battle/battle_task.py`; build the env here so it composes with the GPC agents.

### 5a. Paired spawning

Extend the simulator layer (start with `isaaclab` backend) so each logical *match* owns two
G1 articulations in a shared arena. Two viable layouts:

1. **Two robots per env instance** — cleanest physics (they naturally collide), requires
   the simulator config to instantiate two articulation views and the env to expose both.
   (This is what IsaacLabASE's `battle_task.py` does: a second `Articulation` (`robot_op`)
   in the same env prim.)
2. **Ego/opponent as env pairs** (IsaacLabASE's agent-side batching approach: env `i` and
   env `i + num_actors` form a match; obs tensor is `[ego_batch; opp_batch]`) — reuses more
   of the single-robot plumbing; the batching trick in `hrl_sp_agent.env_step` shows how
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

**Milestone check:** two scripted/random-token G1s in an arena, hits detected and scored
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
| 1 | Dataset build + retarget + curation | 1–2 weeks (mostly human time) |
| 2 | FSQ tracker (combat+AMASS) | 1–2 weeks GPU |
| 3 | GPC prior | 2–4 days GPU |
| 4 | Combat SFT | ~1 day GPU |
| 5 | Battle env | 2–4 weeks engineering |
| 6 | League training | 2+ weeks GPU (open-ended — leagues improve as long as you run them) |
| 7 | Eval tournament | ~1 week engineering |

Phases 2–4 are sequential; Phase 5 can proceed in parallel with 2–4 (it only needs the
tracker/prior at integration time, and can be developed against random-token policies).

---

## Risks & open questions

1. **Combat token coverage** (top risk): if the FSQ tracker can't track retargeted strikes
   on G1's 29-DOF body, those skills don't exist downstream. Mitigation: Phase 2 per-clip
   acceptance test; retarget quality iteration; PHUMA kungfu clips as a second combat source.
2. **G1 morphology limits:** the G1 is small and light with limited arm articulation —
   expect "kickboxing robot" rather than fencing. Unarmed combat only (no sword/shield —
   the dataset, skeleton, and hardware all say no).
3. **Self-play stability:** leagues can still cycle or collapse to stalling. The
   staleness cap, exploiters, points-decision on timeout, and the head-to-head eval matrix
   are the countermeasures; watch draw rate as the leading indicator.
4. **Prior constraint vs. peak skill:** prior-constrained nucleus sampling keeps fights
   human but caps off-manifold exploits. If the league plateaus, experiment with
   `--peft-sampling-mode nucleus` (student nucleus + KL) on exploiter seats first.
5. **Upstream churn:** GPC code landed recently; expect API movement. Pin a known-good
   upstream commit; keep our battle env additions in clearly separated modules
   (`protomotions/envs/battle/`, `examples/experiments/battle/`) to ease rebasing. If
   NVIDIA ships a SOMA GPC prior checkpoint later, it does **not** replace our G1 prior
   (different skeleton) — our pipeline is self-sufficient.
6. **Sim-to-real (future):** everything above is sim-only. The G1 deployment path exists
   here (`deployment/export_bm_tracker_onnx.py`, MuJoCo contract, RoboJuDo), and
   Jetson-Thor-class hardware can run the token loop in real time — but a real-robot fight
   would additionally need domain randomization in Phase 2/6 and a safety layer. Out of
   scope for this plan.

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
| **Live LLC/prior hot-reload** | `hrl_agent.reload_llc_network_if_needed()` (in `../IsaacLabASE/source/IsaacLabASE/ase/learning/hrl_agent.py`, checked every 5 epochs) | Optional, but useful if the FSQ tracker or prior gets refined while league training runs |

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
| FSQ tracker experiment (fork this) | `examples/experiments/mimic/fsq.py` |
| GPC prior training | `examples/experiments/gpc/prior.py` |
| SFT template (fork → combat SFT) | `examples/experiments/gpc/sft_target_prior_peft.py` |
| RLFT template (fork → battle RLFT) | `examples/experiments/gpc/task_steering_headvel_prior_peft.py` |
| PEFT agent internals | `protomotions/agents/peft/` |
| GPC user guide (canonical workflow) | `docs/source/user_guide/gpc.rst` |
| AMASS→G1 retarget | `scripts/retarget_amass_to_robot.sh`, `pyroki/` |
| PHUMA G1 combat clips manifest | `data/yaml_files/g1_phuma_train.yaml` |
| G1 robot config | `protomotions/robot_configs/g1.py` |
| Battle mechanics to port | `../IsaacLabASE/source/IsaacLabASE/IsaacLabASE/tasks/direct/ase/battle/battle_task.py` |
| League/PFSP to port (and fix) | `../IsaacLabASE/source/IsaacLabASE/IsaacLabASE/tasks/direct/ase/battle/pfsp_player_pool.py`, `hrl_sp_agent.py` |
| Combat mocap source | `../IsaacLabASE/source/IsaacLabASE/ase/poselib/data/animations/amp/combat/`, `.../reallusion_combat/` |
| Getup curriculum reference | `../IsaacLabASE/source/IsaacLabASE/IsaacLabASE/tasks/direct/ase/amp_getup_env.py` |
| Eval tournament reference | `../IsaacLabASE/scripts/rl_games_amp/play_battle.py`, `.../battle/hrl_sp_player.py` |
