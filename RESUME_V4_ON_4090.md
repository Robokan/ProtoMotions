# Resuming the SOMA battle league (v4) on the 3× RTX 4090 machine

Instructions for the Claude session on the 4090 box. Context: `soma_battle_league_v4`
is a kickboxing self-play league (PPO over a frozen 198M GPC prior via a DoRA
adapter). It trained to ~epoch 557 on the DGX Spark and is paused; the goal is
to continue it here, faster. The user's standing rules: **never start/resume/
stop training without their explicit go**, and **validate one thing end-to-end
before any batch/long run**.

## 1. Code

The repo is already checked out here. Get the current battle branch:

```bash
git fetch origin && git checkout battle && git pull origin battle
```

Everything below assumes the repo root as cwd.

## 2. Files to copy from the Spark (not in git)

Copy these from `evaughan@<spark-host>:~/sparkpack/ProtoMotions/` preserving
relative paths (~4.2 GB total):

| Path | Size | Why |
|---|---|---|
| `results/soma_battle_league_v4/` (whole dir) | 1.6 GB | checkpoint (`last.ckpt`), frozen configs (`resolved_configs*.pt`), league opponent pool (`lightning_logs/*/league/policy_*.ckpt`, 12 snapshots), env sampling state |
| `results/soma_gpc_prior_p2/` (whole dir, skip `lightning_logs/`) | 3.1 GB | the frozen GPC prior — the league loads BOTH `last.ckpt` AND the `resolved_configs*.pt` beside it (`load_resolved_configs_from_checkpoint`); the bare ckpt alone crashes at startup |
| `data/soma_combat_viewer.pt` | 60 MB | the merged 184-motion combat library (SEED curated + Reallusion kicks, boxing-weighted) |
| `data/pretrained_models/motion_tracker/soma_bones_fsq/` | 240 MB | FSQ tracker (safety: prior tokenization references it in some paths) |

```bash
rsync -avP evaughan@<spark-host>:~/sparkpack/ProtoMotions/results/soma_battle_league_v4 results/
rsync -avP --exclude lightning_logs evaughan@<spark-host>:~/sparkpack/ProtoMotions/results/soma_gpc_prior_p2 results/
rsync -avP evaughan@<spark-host>:~/sparkpack/ProtoMotions/data/soma_combat_viewer.pt data/
rsync -avP evaughan@<spark-host>:~/sparkpack/ProtoMotions/data/pretrained_models/motion_tracker/soma_bones_fsq data/pretrained_models/motion_tracker/
```

## 3. Environment gotchas (learned the hard way on the Spark)

- IsaacLab + IsaacSim required (x86 install here — do NOT reuse the Spark's
  aarch64 container image). `tensordict`, `wandb==0.23.0`, `cloudpickle`
  must be installed into the Isaac python.
- **NumPy must be < 2** (1.26.4). NumPy 2.x coexisting with Isaac's bundled
  1.x breaks the render extensions (annotator recursion) — blocks headless
  video recording; training may work but pin it anyway.
- If wandb is unavailable/unwanted, training runs fine without it.

## 4. Resume validation (do this FIRST, single GPU)

Resume auto-detects from `results/soma_battle_league_v4/last.ckpt` and
reloads the frozen configs — CLI config overrides are IGNORED on resume.
The reward set and env rules ride in `resolved_configs.pt` (already updated:
satisficing approach reward, whole-limb strikers, stomach targets).

```bash
python protomotions/train_agent.py \
    --robot-name soma23 --simulator isaaclab --headless \
    --motion-file data/soma_combat_viewer.pt \
    --experiment-path examples/experiments/battle/battle_league_prior_peft.py \
    --prior-checkpoint results/soma_gpc_prior_p2/last.ckpt \
    --checkpoint results/soma_battle_league_v4/last.ckpt \
    --num-envs 256 --batch-size 512 --training-max-steps 20000000 \
    --experiment-name soma_battle_league_v4
```

Success = it logs `RESUME: Found checkpoint`, restores the league
(`Restored league: 12/12 snapshots`), and completes an epoch (~100 s on the
Spark; expect faster here). Watch the first epochs for CUDA OOM — a 4090 has
24 GB vs the Spark's unified pool. If OOM: drop `--num-envs` to 128 and
`--batch-size` to 256 (league quality is robust to this; snapshots/pool are
unaffected).

Known resume trap: if the motion library file changes size/count later, delete
`results/soma_battle_league_v4/env_soma_combat_viewer.pt.ckpt` (stale
per-motion sampling state → "expanded size" crash). Not needed for this copy.

## 5. Multi-GPU (after single-GPU validates)

The trainer supports `--ngpu` (Lightning Fabric DDP):

```bash
python protomotions/train_agent.py ... --ngpu 3
```

Notes: effective batch scales with GPU count; each rank runs its own IsaacLab
sim instance (VRAM per card = sim + model + optimizer — if 256 envs/rank OOMs,
use 128/rank ≈ 384 total). Multi-GPU + league self-play has NOT been exercised
in this fork — treat the first multi-GPU run as an experiment: validate one
epoch, check `league/pool_size` telemetry still updates, and fall back to
single GPU if snapshot gating misbehaves.

## 6. Monitoring

- Log: `results/soma_battle_league_v4.log` (append `>>` on relaunch).
- Telemetry: TensorBoard scalars under `env/battle/*` — the numbers that
  matter: `facing_mean` (healthy ≈ 0.75), `dealt_hands_mean` vs
  `dealt_legs_mean` (kicking ratio — v4's whole point; it left off around
  15–20:1 with legs growing), `end_ko_mean`, `draw_mean` (should stay ≈ 0).
- `bones-seed/warden_check.py` (on the Spark) shows the health-verdict recipe
  if you want to replicate it.

## 7. What NOT to do

- Don't "fix" the reward/env config on resume — resume intentionally freezes
  configs; changes belong in `resolved_configs.pt` regeneration, and only
  with the user's explicit sign-off.
- Don't enable the stun-KO gate (`stun_gates_ko`) — it's built but
  deliberately off pending calibration.
- Don't start training without the user explicitly saying go.
