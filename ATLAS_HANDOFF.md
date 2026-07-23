# Atlas GPC Pipeline — Handoff Notes

State of the Boston Dynamics Atlas onboarding into ProtoMotions (GPC stack),
written 2026-07-19 for whichever AI/human picks this up. The parallel SOMA
battle-league work lives on the 3×4090 machine (see `RESUME_V4_ON_4090.md`).

## Hard rules (from Eric, do not violate)

- **Only the `battle` Docker container exists.** Never reference or create
  `battle-shakedown`. All sim/training/USD work runs inside `battle`
  (`docker exec battle ...`, repo mounted at `/workspace/sparkpack`).
- **Never start/resume/stop training unless explicitly told, right then.**
- Give Eric plain `python ...` commands (he runs them inside the container
  himself), not docker-exec-wrapped ones.
- Validate ONE item through the real entrypoint before any batch job.
- Ask before committing to any repo other than the one the request is about.
- We are on usage credits: no periodic status polling. Use crash-only
  monitors (alert only when a process dies).

## What is running right now

- **`atlas_tracker_v5`** — FSQ motion tracker, training in the `battle`
  container. Epoch ~15,600 as of this writing, ~5 s/epoch at 512 envs.
  - Launch: `train_agent.py --robot-name atlas --simulator isaaclab
    --motion-file data/atlas_tracker_stage4.pt
    --experiment-path examples/experiments/atlas/tracker_fsq.py
    --num-envs 512 --batch-size 1024 --experiment-name atlas_tracker_v5
    --checkpoint results/atlas_tracker_v4/last.ckpt`
  - Log: `results/atlas_tracker_v5.log`. Checkpoints: rolling `last.ckpt`
    every 10 epochs, archived `epoch_NNNN.ckpt` every 1000 (framework
    defaults, no override needed).
  - Weight lineage: v1 (combat+drunken only) → v2 (full corpus) → v3
    (quality-filtered) → v4 (recovered clips, wide-limit robot) → v5
    (elevated clips removed). Warm-started each time.
  - **v1–v3 checkpoints no longer run in inference**: the USD/MJCF on disk
    now has widened joint limits and `_verify_joint_limits` correctly
    rejects the mismatch. v4+ match the current asset. To view:
    `python protomotions/inference_agent.py --checkpoint
    results/atlas_tracker_v5/last.ckpt --simulator isaaclab`

## The robot asset (regenerable end to end)

Chain: GMR `assets/atlas_mujoco/atlas.xml` (source of truth)
→ `data/scripts/retune_atlas_mjcf.py` (mass 68.04 kg/150 lb, EngineAI-class
torques, ankle restructure, root-quat bake `ATLAS_ROOT_BAKE_QUAT =
[0.67082, 0.74162, 0, 0]`) → GMR `atlas_physics.xml`
→ sed path-rewrite into `protomotions/data/assets/mjcf/atlas.xml`
(meshdir `../mesh/Atlas/`, drop viewer-only headlight/skybox)
→ `usd_convert/flatten_mjcf.py` → `usd_convert/convert_robot_mjcf_to_usda.py`
(in container) → `usd_convert/patch_atlas_usd_bindings.py`
→ verify with `usd_convert/inspect_atlas_joints.py`
(**must** print `revolute=30 d6=0 with_drive=30`).

Gotchas encoded in those scripts (do not "simplify" them away):
- IsaacLab's MJCF importer merges any multi-joint body into one D6 joint
  that never receives PD drives (robot collapses). Ankles must be CHAINED
  single-joint bodies (`Ankle_L/R` pitch → `Foot_L/R` roll). retune handles
  both ball-ankle and two-hinge-in-one-body inputs.
- The USD materials only expose `outputs:mdl:surface`, so every Omniverse
  renderer reads the OmniPBR-style inputs (`diffuse_texture`,
  `diffuse_color_constant`, `metallic_constant`...), NOT the
  UsdPreviewSurface inputs. patch_atlas_usd_bindings authors both, with
  layer-anchored `./materials/*.png` paths (the `./` prefix is mandatory —
  bare relative paths are USD *search* paths and fail).
- Bbody's color = `materials/Bbody_difussion.png`, a flat texture Eric
  tunes by hand at the GMR source (currently RGB 60). No albedo
  adjustments in the converter — the texture IS the color. Emission is an
  adjustable constant green (0.341, 0.906, 0.349) with glow enabled;
  Plastic keeps its (flat dark) texture.
- **Joint limits were widened 2026-07-18** (hips ±150°, knee −170..+5°,
  waist/backbone ±135°, ankle pitch ±80°, roll ±75°) so retargeting can do
  deep crouches and rolls. Joint/body ORDER is unchanged, so all motion
  libs remain valid.
- Robot config: `protomotions/robot_configs/atlas.py` (registered as
  `"atlas"`). 33 bodies / 30 DOF / `default_root_height=1.05`.

## Motion data pipeline

SOMA anything → BVH via `data/scripts/convert_soma23_motion_to_bvh.py`
(exact inverse of the BVH→SOMA converter, validated to 0.00000 m round-trip)
→ GMR retarget `GMR/scripts/retarget_headless.py --robot atlas_fists`
(host, `~/sparkpack/GMR/.venv`, 6 parallel shards)
→ `data/scripts/convert_gmr_pkl_to_proto.py` (**container** — needs
dm_control; handles 30-dof hinge + legacy 34-dof ball layouts, composes the
root bake quat, per-frame height fix offsets 0.056/0.076, and computes real
contact labels — the tracker's contact-match reward crashes on None).
→ package via MotionLib save, merge with
`data/scripts/merge_motion_libs.py` (`--skip-duplicates` dedups by
basename; `--match-weight-of .motion` gives added clips mean base weight).

Key libs in `data/` (gitignored, this machine only):
- `atlas_tracker_stage4.pt` — **current training set**: 8,364 motions,
  1.67M frames. = stage3 minus 362 elevated/stairs/ladder clips (clips
  that start in mid-air and descend imaginary steps — mocap was on real
  stairs; useless on flat terrain). Filter: per-frame lowest-body z
  (airborne fraction > threshold or elevated start/end), name families
  stairs/ladder/jump_off/jump_on; `come_up` get-up clips were false
  positives and were KEPT.
- `atlas_tracker_stage3.pt` 8,726 / stage2 8,756 / seed corpus 8,614 /
  combat_viewer 184 / reallusion 82 / drunken 20 / combat_stage1 102.
- `soma_combat_viewer.pt` — SOMA combat SFT lib, now 178 motions after
  removing 6 `choreography1_injured_*_leg` limping clips (Eric saw league
  humanoids limping on the 4090; backup `.pre_limp_filter_bak`). NOTE: the
  same 6 clips (Atlas-retargeted) are still inside atlas_combat_viewer /
  stage2-4 — harmless for the tracker, but FILTER THEM when building the
  Atlas prior/SFT dataset.

## Retarget quality: solved and open

Original scan (`data/scripts/scan_motion_lib_quality.py`) flagged 235/8,756
clips. Root causes found and fixed:
1. **Cold-start transient** — IK warm-starts frame-to-frame; clips starting
   away from origin/in deep poses "flew" to target (137 m/s observed).
   Fixed: 15 settle solves on frame 0 (GMR commit `dd2da47`). Rescued 39.
2. **Too-narrow joint limits** — knee was −120°, crouches/rolls pegged
   limits and thrashed the root. Fixed by the limit widening. Rescued 166.
3. **IK bistability (OPEN)** — arm flips 50–85 rad/s mid-clip while the
   human is smooth; two arm solutions reach the same hand target. 30 clips
   remain broken, listed with full history in
   `data/atlas_unconvertible_motions.txt` (all SEED: dance, jog_avoid_bump,
   ladder; ZERO reallusion/drunken remain). Likely fix: temporal
   continuity cost / per-frame rate limit in GMR's mink solver setup.
   Eric wants rolls/inversions working — they matter for fighting.

## Next steps (in rough order)

1. Let `atlas_tracker_v5` converge (metrics in tensorboard under
   `results/atlas_tracker_v5/`). This is the FSQ tracker = step BEFORE the
   prior in the GPC pipeline (Atlas has no pretrained tracker, unlike SOMA).
2. Then the Atlas **prior** (distill tracker codes), then combat SFT +
   battle league, mirroring the SOMA phases in `examples/experiments/gpc/`.
   Filter the 6 limping clips out of any Atlas SFT/prior dataset first.
3. Optional GMR solver work for the 30 bistability clips.
4. On the 4090 (SOMA league): Eric may re-run combat SFT on the filtered
   `soma_combat_viewer.pt` (copy it over) — resuming the league alone keeps
   the already-learned limp.

## Housekeeping

- Repos: ProtoMotions `battle` branch (github.com/Robokan/ProtoMotions,
  pushes clean as of `38e8b52`+), GMR master (Robokan/GMR_Grab, clean as of
  `dd2da47`). The 4090 machine also pushes to the battle branch — rebase
  before pushing.
- `assets/atlas_mujoco/**` in GMR is gitignored; anything the committed
  XMLs reference must be `git add -f`'d (already done for meshes, textures,
  `Bbody_difussion.png`, `Bbody_Metalic.png`).
- Memory watchdog: `scripts/memory_watchdog.sh battle 10` (re-arm with
  nohup after host reboot). Crash-only Monitor watches atlas_tracker_v5.
- Recurring friction: container-vs-host file ownership. Fix with
  `docker exec -u root battle chmod -R a+rwX <path>`.
- GMR viewer (`scripts/soma_bvh_to_robot.py --robot atlas_fists`) now has
  ambient headlight + gradient skybox; retargets follow the mocap exactly
  (no ground-height offset — Eric explicitly rejected artificial grounding).
