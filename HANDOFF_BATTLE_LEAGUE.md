# Handoff — SOMA Battle League on the 3×4090 box ("erics-deep-thought")

For the next AI session taking over. Written 2026-07-20. Repo:
`~/sparkpack/ProtoMotions`, branch `battle` (keep it rebased on
`origin/battle` — the Spark machine pushes Atlas-pipeline commits daily;
`git fetch && git rebase --autostash origin/battle` before pushing).

## Eric's standing rules (do not violate)

1. **Never start / resume / stop training without his explicit go.**
   Amendment for the current v5 run only: small corrective tweaks are
   pre-authorized if training goes badly (reward scales/refs, league gating
   params, batch sizes, crash-recovery restarts). Game-rule changes (speed
   gate, damage constants, stun/KO rules, step budget) always need his
   sign-off first.
2. **Validate one thing end-to-end before any long run.**
3. **No checkpoints in git.** Code only. `results/` is git-ignored; the
   modified `data/…/soma_bones_fsq/*` binaries stay uncommitted by policy.
4. **Burn few credits**: no polling monitors that wake the model on healthy
   progress; check telemetry only when asked; keep responses lean. Failure
   watchers (crash/stall) are fine — silent unless something breaks.

## What is running RIGHT NOW

- **`soma_battle_league_v5`** training on **GPU 0** (PID may change on
  restarts — find with `pgrep -f protomotions/train_agent`). Epoch ~4,560 of
  a **200M-step budget** (~24,400 epochs ≈ 1 week total; ~19% done).
  Warm-started from the v4 champion (`results/soma_battle_league_v4/last.ckpt`,
  epoch 2441). Log: `results/soma_battle_league_v5.log`. Checkpoints save
  every 10 epochs — stopping any time keeps the current fighter.
- **GPU 1 drives the display** — use it for recordings
  (`CUDA_VISIBLE_DEVICES=1`), never leave big jobs on it. **Only 2 of the
  supposed 3 4090s are visible** (unresolved hardware question).
- Relaunch command if it dies (get Eric's go first unless it's crash
  recovery):

```bash
source ~/sparkpack/.venv-isaacsim5/bin/activate && cd ~/sparkpack/ProtoMotions && \
OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 setsid python protomotions/train_agent.py \
  --robot-name soma23 --simulator isaaclab --headless \
  --motion-file data/soma_combat_viewer.pt \
  --experiment-path examples/experiments/battle/battle_league_prior_peft.py \
  --prior-checkpoint results/soma_gpc_prior_p2/last.ckpt \
  --checkpoint results/soma_battle_league_v4/last.ckpt \
  --num-envs 256 --batch-size 512 --training-max-steps 200000000 \
  --experiment-name soma_battle_league_v5 >> results/soma_battle_league_v5.log 2>&1
```

  NOTE: a mid-run restart RESUMES from `results/soma_battle_league_v5/last.ckpt`
  automatically (resume mode ignores CLI config args; frozen configs rule).

## The v5 ruleset (all committed; built this week)

Rules live in `examples/experiments/battle/battle_league_prior_peft.py` and
`protomotions/envs/battle/{control,hit_state}.py`:

- **Damage = kinetic energy**: HP loss = `0.005/J × ½·m_limb·v_impact² ×
  region-mult (head 2×)`, deposited ONCE per contact event at onset, **zero
  below 2.5 m/s impact speed**, capped 25%/hit. Limb masses come from the sim
  (`Simulator.get_body_masses`). Never use contact-solver force magnitudes —
  they're impulse artifacts (that bug wiped 100%→0 HP in 3 s of guard-grinding).
- **Dense hit reward**: continuous, UNGATED `log1p(KE / 5 J)` per event —
  Eric's requirement: taps must still pay a small positive guide; only
  health/wins are speed-gated. (`ke_reward_ref` was 70; lowered to 5 after
  run-1 showed the tap gradient was invisible next to the facing reward.)
- **Stun-gated KO**: a downed fighter is KO'd inside the 2 s window only
  while concussed (stun > 0.4, deposited from gated KE, head-weighted).
- **5 s referee count-out**: down for ANY reason > 5 s loses, stun
  irrelevant (Eric's rule — canvas-camping is never safe).
- **Facing reward targets the opponent's CHEST** (boxer's gaze), weight 2.0.
  Facing feeds reward/telemetry only, never policy observations.
- **Win economy (2026-07-22, mid-run change at epoch ~9,440)**: win weight
  **500** (was 100 — it lost to ~1,300/ep of dense income, so fighters
  danced to timeout; 39% into budget there were zero KOs and the 70% gate
  had never fired). Decisive wins/losses additionally scale by
  `1 + early_finish_win_scale × time_left_frac` (config, default 1.0) — an
  early KO pays up to 2×, an early loss costs 2×; draws never scale.
  Patched into v5's frozen configs (backups `.pre_win500_bak`). Watch for:
  `end_ko_mean` lifting off, first `reason="gate"` snapshot, Elo detaching
  from 1000. If fights become RECKLESS brawls (all-offense, no defense),
  the lever is dialing `early_finish_win_scale` down, with Eric's go.

## How v5 is doing (last read: epoch ~4,560)

Early telemetry (epoch ~580) showed: decisive finishes exist
(`end_ko_mean` > 0 — includes count-outs, not only concussions), Elo moving
(not pinned at 1000 like run 1), legs occasionally landing, facing no longer
monopolizing learning. **The limp is gone** — the combat data contains 6
limping clips that caused a permanent stumble in all earlier eras; v5's
count-out pressure trained it out by ~snapshot 13. What to watch next:
`end_ko_mean` compounding, first `reason="gate"` snapshot (the 70%-win-rate
gate has probably fired by now given 92 snapshots — check
`grep 'reason' results/soma_battle_league_v5.log` or snapshot metadata),
hands/legs KE flow rising, Elo trending.

Telemetry one-liner:

```bash
cd ~/sparkpack/ProtoMotions && ~/sparkpack/.venv-isaacsim5/bin/python - <<'EOF'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
d = sorted(glob.glob('results/soma_battle_league_v5/lightning_logs/version_*'), key=lambda p:int(p.split('_')[-1]))[-1]
ea = EventAccumulator(d); ea.Reload(); tags = ea.Tags()['scalars']
for w in ['end_ko_mean','dealt_hands_mean','dealt_legs_mean','facing_mean','pool_size','agent_elo','pool_avg_win_rate']:
    m=[t for t in tags if w in t]
    if m: e=ea.Scalars(m[0]); print(w, f"{e[-1].value:.4f}@{e[-1].step}")
EOF
```

Archived comparison run: `results/soma_battle_league_v5_run1_ref70/` (1,622
epochs at ke_reward_ref=70 — the failed flat run; keep for A/B reference).

## Recording fights

```bash
CUDA_VISIBLE_DEVICES=1 scripts/record_fight.sh [bouts] [run] [A.ckpt] [B.ckpt]
```

Defaults: 3 bouts, `soma_battle_league_v5`, latest snapshot vs random pool
member. Output: `output/fight_videos/`. The script already sets the correct
decode (trained `prior_constraint`, stochastic — `--deterministic` +
nucleus fast-sampling makes fighters walk instead of fight; never re-add).
Viewing config `results/soma_battle_league_v5/resolved_configs_inference.pt`
has `prior_top_p=0.5` — **Eric prefers this** ("much more aggressive dynamic
fight"); 0.9 backup at `.pre_topp_bak`. v4 viewing config similarly patched
(raw damage + stun on; backup `.pre_rawdamage_bak`).

## Machine gotchas (each cost us real time — respect them)

- **Env**: `source ~/sparkpack/.venv-isaacsim5/bin/activate` (Python 3.11,
  Isaac Sim 5.x + IsaacLab 2.3.2 editable). NOT `.venv-isaacsim6` —
  IsaacLab 3.0 removed `PhysxCfg`. NumPy pinned 1.26.4.
- **`OMNI_KIT_ACCEPT_EULA=YES`** on every Isaac launch.
- **Kit hijacks Python logging**: after AppLauncher boots, `log.info` goes to
  `/tmp/isaaclab/logs/isaaclab_<date>.log`, NOT your stdout redirect. Check
  there before diagnosing a "silent crash". Also `battle_tournament.py` ends
  with `os._exit()` which discards buffered stdio — use `PYTHONUNBUFFERED=1`.
- **Launch trainers with `setsid`**: killing a wrapper shell wounds the
  python child (Lightning catches the signal, hangs forever in Kit teardown
  at 0% GPU — a silent stall that passes liveness checks). Pair a
  log-staleness stall detector (>5 min quiet) with any liveness watch.
- **One Isaac job per GPU**: a second launch OOMs against a finishing one's
  teardown; wait for VRAM to actually drop, not for the mp4 to appear.
- **Steady-state training VRAM is ~23/24 GB** — no headroom; don't raise
  num-envs.
- Bash tool loses cwd/venv between some calls — use absolute paths and
  re-`source` per command.

## Pending / open items

1. **v4 champion exists ONLY on this disk** (`results/soma_battle_league_v4/`,
   epoch 2441 + 50 league snapshots). The Elements USB drive has been
   disconnected since a machine crash on 2026-07-17 — when Eric replugs it,
   rsync the v4 (and v5) results into `v4_transfer/` on the drive.
2. **Third 4090 missing** from nvidia-smi — needs a hardware look.
3. **Eric's roadmap**: he is training Tracker+Prior for **Atlas** on the
   Spark now (clean data, no limping clips); will retrain the SOMA adapter
   without limping later. Atlas is robot #2 for the future multi-robot
   league.
4. **Future-work plans** (design docs, committed, no implementation):
   - `MULTI_ROBOT_LEAGUE_PLAN.md` — 3 GPUs / 3 robots / one shared opponent
     pool; phased (pool hygiene → shared pool → cross-prior opponent
     bundles → cross-morphology fights). Do Phase 0 (snapshot provenance,
     atomic writes) before any pool sharing.
   - `SKINNED_OVERLAY_PLAN.md` — drive a rigged character's UsdSkel skeleton
     from the SOMA articulation for pretty fight videos; reuses the
     SOMA23→BVH converter's joint math.
5. Claude-session memory (auto-loaded for Claude sessions; other AIs: read
   `~/.claude/projects/-home-bizon-sparkpack/memory/battle-league-4090-setup.md`)
   duplicates most of this file plus finer details.

## This week's history in one paragraph

Resumed v4 from the Spark's epoch-550 checkpoint (missing prior configs were
regenerated with `--create-config-only`), trained it to its 20M-step budget
(epoch 2441). Discovered via probes that the "champion" never threw a real
strike (all 77 measured contacts < 1.02 m/s — guard-grinding that the old
log-normalized damage model rewarded; light taps also one-shot bouts due to
contact-force impulse spikes). Rebuilt the damage/win economy around
per-event kinetic energy with a speed gate (Eric's "speed and mass" call),
added the concussion-gated KO and his 5 s count-out, retargeted the facing
reward to the chest, made the dense reward continuous (his requirement) and
recalibrated its reference after run 1 flatlined. v5 (current run) is the
first version whose fights have a working win/loss economy; the limp is
already trained out and fights look dramatically more aggressive under the
top_p=0.5 viewing decode Eric prefers. All of it is committed on `battle`
with detailed messages from `ad417fa` through `deddb5f`.
