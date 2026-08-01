# Multi-Robot Shared League — Design Plan

**Status: PHASES 0 + 1 SHIPPED (2026-08-01, commit 29fd807); Phases 2/2b/3/G
remain.** Investigated 2026-07-18 by a 4-way code survey (league/pool, battle
env, simulator, model bundles); every claim below carries a file:line
reference into the `battle` branch as of `e28746f` — line numbers predate the
2026-07-29 league refactor (see status update), but the behavioral claims
were re-verified 2026-07-30 unless marked otherwise.

## STATUS UPDATE 2026-08-01 (Phases 0 + 1 shipped)

`agents/league/pool_io.py` + both league homes (`full_model_league.py`,
`agent.py`) now implement everything in Phase 0 and Phase 1:
atomic writes, `policy_{run_id}_{counter}.ckpt` names, full provenance
(run_id/robot/architecture/weight-shape fingerprint/action_dim/
`game_rules_version="ke1.5-stunko-win500-kick"`/schema v2), loud
loader-side refusal (`SnapshotIncompatible`), embedded-time ordering,
`league.shared_pool_dir` (CLI `--shared-pool-dir` on all three league
experiments), mid-run re-scan every `pool_rescan_epochs`, own-family-only
gating and stat resets, per-family quotas with family-aware eviction, and
own-family-aware seeding. ase_hlc snapshots pin their frozen LLC
(path + mtime + reload count) and the latent contract — the Phase 2b
bundle wrinkle is pre-solved. Validated by a 22-check unit harness plus
two concurrent live 128-env HLC leagues sharing one pool dir (both
directions ingested mid-run; no collisions or torn reads).
Open decision 1 → resolved as recommended (own-family gating). Open
decision 5 → rules-era stamp exists; PEFT snapshots now stamped too.
NOT built (deliberately): file GC for the shared dir (quotas govern pool
membership, files accumulate), and the optional ratings.jsonl sidecar
(open decision 2).

## STATUS UPDATE 2026-07-30 (what has landed since the survey)

Groundwork that changes this plan's baseline — none of it was planned here,
but several phases got cheaper:

- **League machinery is now shared code.** The ASE league's orchestration was
  extracted into `agents/league/full_model_league.py`
  (`FullModelLeagueMixin`): pool, full-model snapshots, lanes, Elo, ckpt
  persistence, with subclass hooks. `LeagueASEAgent` and the new
  `LeagueASEHLCAgent` both ride it; the PEFT league (`agents/league/agent.py`)
  is untouched — so Phase 0/1 changes now have ONE extra home, not two.
  (Opponent resampling/match accounting were also vectorized — one GPU sync
  per batch instead of per ended match; matters at 4096 concurrent matches.)
- **Phase 2b's ASE arm is BUILT and in production.** The paper-faithful
  frozen-LLC + HLC league (`examples/experiments/ase/battle_league_ase_hlc.py`,
  `agents/league/ase_hlc_agent.py`) is training atlas-vs-atlas at 8192 envs
  as of 2026-07-30. Its snapshots are self-contained HLC weights (~12 MB) and
  — partial Phase 0 item 3 — full-model-league snapshots now carry
  `architecture` ("ase" / "ase_hlc") and `robot` provenance keys. PEFT
  snapshots remain anonymous (re-verified 2026-07-30).
- **LLC hot-reload exists** (`hlc.llc_hot_reload`): the HLC league re-loads
  its frozen LLC when the pretrain saves a new checkpoint (PFSP stats reset
  on reload). This is the live counterexample to "pools never refresh
  mid-run" — the *base model* refreshes; the pool-file re-scan of Phase 1 is
  still unimplemented. It also sharpens the Phase 2b bundle question: an
  ASE-HLC bundle is {tiny HLC weights + LLC checkpoint ref}, and the LLC
  under a bundle can move.
- **Atlas is confirmed as robot #2 in practice** (open decision 3): per-robot
  battle body tables exist (`envs/battle/robot_tables.py` — soma23 pinned,
  t800/atlas derived from semantic naming; kick-bonus feet included as of
  2026-07-29), atlas has a v6 corpus, a 2-day ASE LLC pretrain (still
  improving), and the production HLC league. The Phase 3 "BattleBodyTables
  per side" item now reduces to making the EXISTING per-robot tables
  per-SIDE.
- **Robot multi-asset spawning has a precedent**: the tournament's
  `--opponent-asset` spawns the opponent half of paired envs from a variant
  robot USD via `MultiAssetSpawnerCfg` with `replicate_physics=False`
  (`simulator/isaaclab/utils/scene.py`). Same morphology (visual variant
  only) — but it exercises the exact spawn path Phase 3's two-entity scene
  needs, including the instancing pitfalls (material opinions inside
  instances don't render; bind above or de-instance).
- **Scale/hardware reality shifted**: leagues now run 8192 envs (4096
  matches, ~20.6 GB, 18-25 s/epoch after profiling work) — Phase 2's VRAM
  tax (~1.6 GB/foreign prior family) still fits, but the "decode latency"
  cost model should be re-checked against the HLC league (tiny MLPs, no
  autoregressive decode: foreign ASE/HLC families are nearly free; foreign
  GPC priors remain the expensive case). GPU 1 currently runs the LLC
  pretrain alongside GPU 0's league (the two-trainer workflow) — the
  display-reservation note below is stale in practice; the third 4090 is
  still missing.

## The idea

Run three concurrent league trainings, one per 4090, each training a
**different robot** (different tracker + GPC prior + adapters), all publishing
to and drawing opponents from **one shared pool** — so every robot can fight
snapshots of itself *and* of the other two robots.

## Verdict up front

| Tier | Scope | Feasibility | Size |
|---|---|---|---|
| 0 | Pool hygiene (prereq for everything) | straightforward | days |
| 1 | 3 runs, same robot+prior, shared pool | mostly plumbing | ~1 week |
| 2 | Same robot, different priors/trackers | real feature (opponent bundles) | ~2–3 weeks |
| 3 | Different robot morphologies fighting | simulator + env re-architecture | ~4–8 weeks |
| G | "Ghost sparring" shortcut (no contact physics) | cheap partial alternative | ~1 week |

Hardware note (updated 2026-07-30, per Eric): plan for **2-3 4090s** — the
third card has been invisible to `nvidia-smi` since 2026-07-16 but may
return. **The display is driven by GPU 0 and training on it is fine** (no
reserved card). Current allocation: GPU 0 league + GPU 1 LLC pretrain. A
live viewer still needs a few GB carved out of whichever card it renders
on, so plan viewer sessions around the ~20-23 GB training footprints.

### Does any of this slow training down?

The plan document itself: no — nothing changes until a phase is built. Per
phase, the runtime cost to a training run that adopts it:

- **Phase 0 (hygiene)**: ~zero. Atomic renames, provenance keys, and
  load-time validation are all off the hot path (snapshot events happen
  every few hundred epochs).
- **Phase 1 (shared pool)**: negligible. A directory glob every M epochs +
  occasional snapshot loads; the same cost the league already pays when its
  own gate fires. No per-step cost.
- **Phase 2 (foreign GPC priors as opponents)**: the real one. Each live
  foreign GPC family costs ~1.6 GB VRAM and its own 8-step autoregressive
  decode per opponent action — in the launch-bound regime that can multiply
  opponent-inference time by the number of live foreign families. Mitigate
  by capping live foreign members, fp16 foreign priors, or the
  single-foreign-member exploiter mode. Foreign **ase/ase_hlc** families
  (Phase 2b) are nearly free — tiny MLP forwards, negligible VRAM.
- **Phase 3 (cross-morphology)**: moderate, structural. Two articulation
  views + zero-padded state/action tensors + per-side table lookups add
  per-step work; expect single-digit-percent overhead if done with padded
  batched ops, worse if any per-side Python loops slip in (see the
  2026-07-30 league profiling: Python-per-event costs dominate fast at
  4096+ matches).
- **Phase G (ghost sparring)**: ~free at runtime (replayed trajectories,
  no second policy, no contact solve).

---

## What exists today (the load-bearing facts)

**Snapshots are adapter-only and anonymous.** A league `policy_*.ckpt` is
`{model: adapter_state, epoch, rating, reason, time}` — 552 K params of
LoRA/DoRA/FiLM deltas plus the task head and its obs-normalizer running stats
(`agents/league/agent.py:217-228`, `agents/peft/utils/adapter_state.py:27-59`).
The frozen prior is **not** in the file and the file carries **no identity**
(robot, prior, obs dims). An adapter is a delta on one specific prior's
weights; its logits index that prior's FSQ vocabulary (9^5 = 59049 tokens) and
decode only through that prior's tracker decoder. Loading a foreign adapter
either fails the strict shape check or — if two priors coincidentally match
shapes — **loads silently and plays garbage** (`adapter_state.py:62-93`).

**Opponents are ego-clones.** `OpponentLanes` deep-copies the ego model and
re-points every non-adapter tensor at the ego's frozen prior storage
(`agents/league/lanes.py:55-82`); N lanes cost ~1× base + N × adapter. The
lane `act()` loop routes per member independently
(`lanes.py:132-163`) — it does not require lanes to share an architecture,
which is the door Phase 2 walks through.

**The pool never refreshes mid-run.** `role=main` globs `league_dir` once at
startup and afterwards grows only from its own snapshots
(`agent.py:144-188, 410-430`). Only `main_exploiter` re-polls a (foreign)
dir — every 5 epochs, latest snapshot only, single member, no PFSP
(`agent.py:367-405`). It is the working template for cross-run pool watching.

**Concurrent writers are destructive today.** Snapshot names are
`policy_{counter}.ckpt` with per-process counters seeded from `len(dir)`;
writes are non-atomic `torch.save` to the final path; ordering is `st_mtime`;
there is no locking anywhere (`agent.py:166-188, 218-228`). Three writers in
one dir = silent overwrites, torn reads, counter collisions.

**Ratings/stats are per-process.** Elo is updated in RAM and stamped into the
snapshot file only at creation (`agent.py:348-360`); PFSP stats live in the
run's own `last.ckpt`. Three trainers would hold three divergent rating views
of the same shared file — fine for PFSP (per-robot views are legitimate), but
a *comparable* cross-robot ladder needs a shared rating store.

**The env and simulator are single-morphology to the bone.** One
`ArticulationCfg` named "robot" spans all envs; every state getter reshapes to
`(num_envs, num_bodies, ...)`; one `DataConversionMapping`, one PD-gain
vector, one flat `[2N, action_dim]` action tensor spliced by
`torch.cat([ego, opp])` (`simulator/isaaclab/utils/scene.py:128-195`,
`base_simulator/simulator.py:137-266`, `agents/league/self_play_env.py:107`).
The hit FSM sizes its buffers by one shared damage-body table and — sharpest
edge — looks up the **opponent's** strike surfaces/masses with the **ego's**
body indices (`envs/battle/hit_state.py:190-214`). PhysX hard constraint:
articulation views must be homogeneous; two morphologies require two separate
scene entities/views, so **an env slot's robot type is frozen at scene
creation** (`omni.physics.tensors` api; IsaacLab `MultiAssetSpawnerCfg`
exists and is already used for scene objects).

**Two genuinely portable pieces.** (1) The opponent observation kernel is
shape-stable across morphologies: width = 20 + 6K where K = key-body count —
standardize K=5 (head/hands/feet) across robots and `task_obs` is identical
league-wide with zero kernel changes (`envs/battle/obs.py:18-63`). (2) The KE
damage model reads limb masses from the sim per-env
(`get_body_masses`, already `[num_envs, num_bodies]`), so cross-robot damage
physics mostly "just works" — only the `masses.mean(dim=0)` collapse and the
SOMA-calibrated gates need per-robot treatment (`envs/battle/control.py:489-506`).

---

## Phase 0 — Shared-pool hygiene (do first, benefits everything)

1. **Atomic snapshot writes**: `torch.save` to `*.tmp` + `os.replace` in
   `_take_snapshot` (~5 lines). Kills torn reads.
2. **Collision-proof names**: `policy_{run_id}_{counter}.ckpt`; stop seeding
   counters from `len(dir)`.
3. **Provenance metadata** in the snapshot dict (backward compatible — readers
   use `meta.get`): `robot`, `prior_checkpoint` + weight fingerprint (hash a
   few prior tensors), `obs_dims`, `action_dim`, `game_rules_version` (the KE/
   stun calibration era — snapshots from different rule eras are not
   comparable), `schema_version`, `run_id`.
4. **Loader-side validation**: refuse (loudly) to assign a snapshot whose
   fingerprint/dims don't match the hosting lane. Closes the silent-garbage
   hole even for today's single-robot workflows.
5. **Ordering by embedded `time` field**, not `st_mtime`.

## Phase 1 — Shared pool, same robot + same prior (3 seeds/configs)

Everything loads everywhere already; this is coordination only.

- `LeagueParams.shared_pool_dir` (CLI-plumbed like `--league-opponent-dir`).
- **Mid-run re-scan**: in `post_epoch_logging`, every M epochs, glob for
  foreign `policy_*.ckpt` (by `run_id` prefix ≠ own) and `pool.add` new ones.
  The exploiter's `_refresh_exploiter_opponent` is the template, generalized
  to all-new-files + PFSP membership instead of latest-only/member-0.
- **Gate semantics decision**: `gate_win_rate=0.7` currently averages over
  *all* pool members. In a shared pool that gates robot A's growth on beating
  B and C. Recommend: gate on win-rate vs **own-family** members only; track
  cross-family win rate as telemetry. Also: stop `pool.reset_stats()` wiping
  foreign-member stats on every own-snapshot (`agent.py:229-239`).
- **Eviction/window**: restore keeps last `max_members` by time — a
  fast-snapshotting run crowds others out. Per-family quotas
  (e.g. 32 total = ~11/family, `max_members` per family) and actual file GC.
- **Elo**: leave per-process (PFSP only needs relative-to-me numbers). If a
  true cross-run ladder is wanted, add a tiny append-only `ratings.jsonl`
  sidecar (single-writer-per-line is atomic enough) — optional.
- Validation: 2 concurrent runs on one GPU at small `num_envs`, assert both
  pools ingest each other's snapshots, no name collisions, no torn loads,
  gates fire independently.

## Phase 2 — Different priors/trackers per run (same robot)

The pool now contains adapter families that need different frozen bases.
Same robot ⇒ env obs/actions are identical; only the **model hosting** changes.

- **Self-contained opponent bundles**: one `torch.save` file =
  `{model_config (with tracker decoder embedded via the existing
  PretrainedModelConfig.module_config mechanism), full prior state dict,
  adapter state, PEFT actor config, obs spec (actor in_keys + prior
  context_in_keys + dims + action_dim), provenance}`. Every ingredient exists
  piecemeal (`prepare_inference_config_for_save`,
  `agents/common/pretrained.py:86-124`, `prior_setup.py:146-187`); the packer
  and a `load_opponent_bundle()` are new. Publish bundles (~0.8 GB) *once per
  family per era*; per-snapshot files stay slim adapters referencing a bundle.
- **Per-family lane factories**: replace the single `model_factory` deepcopy
  with a registry keyed by family fingerprint; `share_frozen_base` aliasing
  per family (one non-aliasable ~0.8 GB fp32 prior copy per foreign family).
  Lane `act()` routing is already per-member — unchanged.
- **Cost model** (per hosting run, 2 foreign families live): +~1.6 GB VRAM
  weights + up to 3× opponent decode latency in the launch-bound regime
  (each family runs its own 8-step autoregressive chain). Mitigations: cap
  concurrent foreign members per episode batch; fp16 foreign priors;
  `exploiter`-style single-foreign-member mode as the cheap fallback.
- **Compatibility checks**: validate bundle obs spec against the hosting
  env's obs keys via `resolve_frozen_prior_input_keys`
  (`frozen_prior_contract.py:38-51`) at assign time.
- Validation: host a v4-prior opponent inside a v5-prior run (we already have
  two prior eras on disk — `soma_gpc_prior_p2` and any retrained successor —
  they make a perfect testbed without any new robots).

## Phase 3 — Cross-morphology fights (SOMA vs. Atlas vs. X)

The big one. Block-partitioned design (general per-env mixing is strictly
harder for no benefit given the pairing structure):

- **Scene**: two `ArticulationCfg` entities (`robot_a` for envs `[0..N)`,
  `robot_b` for `[N..2N)`), robot-scoped contact-sensor names (both SOMA and
  Atlas have a "Head" — names collide today, `scene.py:206-216`). Two spawn
  options: subset prim-path spawning with `replicate_physics=False`, or
  both-robots-every-env + `park_envs` the inactive twin (simpler, 2× actors).
- **Simulator**: `MultiRobotIsaacLabSimulator` overriding the ~12 abstract
  `_get_simulator_*/_apply_simulator_*` funnels with per-block view dispatch;
  per-robot `DataConversionMapping` + gains; state/action tensors padded to
  `max(num_bodies)/max(num_dofs)` so downstream `[2N, ...]` contracts hold
  (zero-padded DOFs must be masked out of PD application). Re-run joint-limit
  verification per view.
- **Env**: `BattleBodyTables` dataclass (strike/damage/key/head/facing/gaze/
  multipliers/stun-weights) resolved **per side**; hit FSM with per-side D×S
  (pad + mask); per-side scalar rules (knockdown height by stature); per-side
  motion libraries and reference-pose reset in `BaseEnv`; per-robot hit-reward
  normalization (`_e0` is currently one global EMA — heavier robots would
  compress lighter robots' reward scale); opponent-identity embedding appended
  to `task_obs` (obs width change — new config era).
- **League**: opponent slots are morphology-partitioned at scene creation
  (PhysX constraint), so the scheduler assigns snapshots only to matching
  slots; each trainer's scene must include every robot family it may fight.
- **Calibration**: the 2.5 m/s gate / 70 J refs / force_on=20 N /
  proximity 0.35 m were measured on SOMA. Re-run the probe per robot;
  KE physics itself is morphology-agnostic (mass comes from the sim).
- Validation ladder: (1) two-entity scene with the SAME robot on both sides
  (isolates simulator work from body-table work); (2) SOMA-vs-SOMA via the
  new per-side tables; (3) SOMA-vs-Atlas exhibition; (4) training.

## Phase 2b — Architecture-agnostic bundles (ASE x GPC, added 2026-07-22)

Define the opponent bundle one level up: not "prior + adapter" but a
SELF-CONTAINED POLICY — {obs spec, action spec, model builder, weights,
provenance(robot, architecture family, rules era)}. A GPC bundle wraps
prior+adapter; an ASE bundle (see ASE_BASELINE_PLAN.md) wraps its
high-level+low-level composed. Lanes' per-member routing is already
architecture-agnostic, so same-robot ASE-vs-GPC is the EASIEST cross-family
case (identical obs/actions, no foreign frozen base to host). This makes
architecture and morphology orthogonal axes: Phase 2 = cross-prior,
Phase 2b = cross-architecture, Phase 3 = cross-morphology; bundles carry
all three. Bonus: mixed TRAINING leagues — an ASE snapshot in a GPC pool
(or an ASE main-exploiter seat: ~10x cheaper per step, stylistically
alien) is a natural hole-finder.

*2026-07-30 amendment*: the ASE side now has TWO snapshot species —
full-weight (`architecture: "ase"`, battle_league_ase.py) and HLC-only
(`architecture: "ase_hlc"`, the production league). An ase_hlc bundle is
{HLC weights + LLC checkpoint reference + latent contract (64-dim
hypersphere, decision_interval)} — the frozen LLC plays the same role as
the GPC prior in Phase 2's cost model but is ~2.3 M params (negligible
VRAM, no autoregressive decode), so hosting foreign ase_hlc families is
nearly free. One new wrinkle hot-reload introduces: an HLC snapshot's
behavior depends on WHICH LLC checkpoint it executes under — bundles
should pin the LLC by fingerprint, not by live path, or accept the drift
(a snapshot replayed under a newer LLC of the same lineage degrades
gracefully; see the pool-staleness discussion in the HLC league notes).

## Phase G — Ghost sparring (cheap cross-robot exposure, anytime)

`VirtualOpponentControl` proves BattleContext can be fed an opponent that is
not physically in the sim (`envs/battle/virtual_opponent.py:163-208`). Replay
another robot's recorded root+key-body trajectories as the "opponent" — no
contact physics (no damage exchange), but the policy learns the other robot's
movement patterns, spacing, and timing. Days of work, usable before Phase 3,
and standardizing K=5 key bodies makes it plug-compatible league-wide.

## Cross-cutting concerns

- **Rules-era stamping**: pool snapshots trained under different
  damage/stun/count-out rules are different games. `game_rules_version` in
  provenance; pools should partition or GC across era changes.
- **Pickle fragility**: bundles embed pickled dataclasses; class-path renames
  break old bundles (`config_utils.py:40-46`). Keep bundle configs minimal
  and version-stamped.
- **Hardware**: 3rd 4090 currently absent; GPU 1 shares with the display.
  Phase 2's VRAM tax (~1.6 GB + decode latency) fits alongside the ~23 GB
  training footprint only if `num_envs` drops or foreign members are capped.
- **Disk**: shared dir grows unbounded today (eviction never deletes files);
  add GC with per-family quotas in Phase 0/1.

## Open decisions (for Eric)

1. Phase 1 gate semantics: own-family-only gating (recommended) vs. mixed?
2. Shared Elo ladder: worth a sidecar store, or is per-run PFSP enough?
3. ~~Phase 3 second robot: Atlas — confirm as the target morphology?~~
   **Resolved in practice (2026-07-30)**: Atlas is robot #2 — battle tables,
   v6 corpus, LLC pretrain, and a production HLC league all exist.
4. Budget check: Phase 2 buys "my robot trains against differently-brained
   opponents of the same body"; Phase 3 buys true mixed-morphology fights.
   Decide whether Phase 2 alone captures enough of the value first.
   *(2026-07-30 note: with the HLC league live, the cheapest high-value next
   step is Phase 2b same-robot ASE-vs-GPC — for SOMA it needs a soma HLC/ASE
   seat; for atlas it needs a GPC prior for atlas. The atlas-vs-atlas
   HLC-vs-HLC case is already covered by the running league's own pool.)*
5. NEW: rules-era stamp for the current game rules (KE damage + 1.5 m/s gate
   + stun-gated KO + win-500 + kick bonus) — Phase 0 item 3 remains undone
   for PEFT snapshots and un-fingerprinted everywhere; the v4/v5/v6 +
   atlas-league pools now span at least three rule eras on disk.
