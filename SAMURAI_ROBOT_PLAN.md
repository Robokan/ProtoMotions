# Samurai Robot Plan — rig2mjcf pipeline (handoff document)

Goal: convert ANY rigged character (first test: the Red Samurai) into a
matching MuJoCo robot, retarget the same motion corpus the Atlas ASE uses
onto it with GMR, and train it with ASE. This gives a physics body whose
proportions exactly match the rendered character (overlay becomes identity —
no auto-fit / A-pose constants needed). The existing SOMA-mapping overlay
path stays; this is an additional capability.

Eric's constraints: never train on GPU 1 (display). Never use absolute paths
in commands handed to Eric (repo-relative from ProtoMotions root; inside
scripts/code, resolving to absolute is fine). Don't commit samurai binary
assets (USD/textures/FBX). Kill training processes with SIGKILL (they ignore
SIGTERM). ALWAYS cd to the repo root in shell commands — half the incidents
this week were cwd traps.

## Status (update as you go)
- [x] Plan written
- [x] rig2mjcf.py generator (data/scripts/rig2mjcf.py) — template-driven:
      reuses soma23_humanoid.xml joints/gains/actuators verbatim, swaps in
      character offsets + skin-fit capsules + 75 kg mass. Weapon-bone
      vertices excluded; per-body radius floors (Chest 0.10) and caps
      (hands 0.05 fist-size, head 0.12).
- [x] samurai.xml FK-validated: 23 bodies / 67 joints / 75.1 kg; zero pose
      is a T-pose, arms +-X, toes -Y, ALL body frames world-aligned to
      0.0 deg (soma convention holds). Drop test still pending (soma23
      template uses torque motors, no PD in-mjcf — do it via ProtoMotions
      sim or mujoco with position servo overrides).
- [x] GMR integration: samurai registered in params.py + retarget_headless
      choices; ik config ik_configs/soma_bvh_to_samurai.json generated.
- [~] IK config calibration IN PROGRESS — read this before continuing:
      * The soma-bvh loader's human body frames are NOT world-aligned and
        NOT constant-offset from the corpus conversion's frames: the two
        pipelines decompose LIMB TWIST differently, so no constant per-body
        rot offset exists for arms/forearms/hands (offset spread 100-175 deg,
        scaling with motion speed). Spine/legs/feet/head have quasi-constant
        offsets (spread 22-36 deg) — those were calibrated from
        hips-relative rotations on shadow_boxing_R_001__A359 and written
        into the config. Hips offset came out clean (10 deg residual).
      * KEYPOINT POSITIONS agree between loader and corpus to 0.01 cm at
        zero lag — positions are the reliable channel. Arm-chain rot
        weights are therefore ZEROED (position-driven arms).
      * DO NOT validate retargets by comparing FK against the soma corpus
        .motion ground truth — invalid oracle (twist conventions differ;
        atlas would fail it too). Validate with: (a) FK-vs-IK-target
        keypoint residuals, (b) the mined-range/±pi saturation scan,
        (c) VISUALLY via GMR's live viewer (the only trustworthy oracle):
          cd ~/sparkpack/GMR_Grab && .venv/bin/python \
            scripts/soma_bvh_to_robot.py --bvh_file <bvh> --robot samurai \
            --rate_limit
      * Zero-channel BVH is NOT a T-pose for this convention (skeleton
        lies flat along +X) — useless for calibration.
      * RESOLVED (2026-07-26): the garbled/tilted output was a wrong HIPS
        rot offset (calibrated from the degenerate zero pose). Final
        calibration recipe, now in the config: O_hips = mean over frames of
        conj(q_loader(Hips,t)) x q_corpus(Hips,t) — this ALSO pins GMR's
        output world to the corpus world (spread 12.7 deg = genuine
        constant). All other bodies: O_b = conj(N_t) x O_hips x M_t with
        N/M the hips-relative loader/corpus rotations. Arm-chain rot
        weights stay 0 (position-driven). Rendered frames confirm clean
        boxing poses (see scratchpad samurai_motion2_*.png).
- [x] Mirror bug fixed (robot left followed human right): O_hips must be
      calibrated in the LOADER world (insert Rz180 between loader and corpus
      quats) so orientation targets agree with position targets. Spread
      dropped to 0.0 deg. Eric visually approved the live retarget.
- [x] 142-stem v6 corpus retargeted (list: ~/sparkpack/output/samurai_v6.list,
      npz: ~/sparkpack/output/samurai_npz_v6, 142/142 ok).
- [x] Robot config: protomotions/robot_configs/samurai.py (+factory entry).
- [x] Converter: data/scripts/convert_gmr_npz_to_samurai.py. NOTE: uses a
      continuity-aware euler_xyz decomposition (both branches per frame,
      nearest-to-previous wins) — the library decomposer branch-flips at the
      y~90deg singularity (lying getup poses), which showed up as 188-376
      rad/s dof spikes. Also: joint_rot_mats includes the free root — skip
      index 0 or you emit 69 dofs.
- [x] Corpus: data/samurai_pretrain_corpus_v6.pt — 142 clips, v6 weights
      copied per stem from atlas v6, 15 getups at 2x. Quality gates: 6/142
      flagged, all the known genuinely-fast kicks (same set flags on
      atlas/t800); zero wrap spikes, zero saturated joints.
- [x] USD: protomotions/data/assets/usd/samurai/samurai.usda (IsaacLab
      MjcfConverter, headless).
- [ ] ASE pretrain launch (samurai_ase_pretrain_v6) — STAGED, awaiting
      Eric's go + GPU assignment:
      OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=<0|2> nohup python \
        protomotions/train_agent.py --robot-name samurai --simulator \
        isaaclab --headless --experiment-path examples/experiments/ase/mlp.py \
        --motion-file data/samurai_pretrain_corpus_v6.pt --num-envs 4096 \
        --batch-size 8192 --experiment-name samurai_ase_pretrain_v6 \
        >> results/samurai_ase_pretrain_v6.log 2>&1 &

## Inputs that already exist
- Character: protomotions/data/assets/overlay/red_samurai.usd — 12 meshes,
  ONE UsdSkel skeleton (192 joints, Epic/UE5 names), Z-up, faces -Y,
  centimeter-ish scale (pelvis z ~0.96 after fit scale 0.0105), ARMS BIND IN
  A-POSE (~55 deg down). Per-point skin weights (elementSize 1).
- Source FBXs: protomotions/data/assets/mesh/red_samurai/SKM_RS_*.FBX.
- Bone maps: protomotions/simulator/isaaclab/overlay_map.py —
  SOMA23_TO_UE (23 SOMA bodies -> Epic bones), UE_REST_REL (A-pose arm
  quats), SOMA23_TPOSE_POS (soma T-pose positions), SOMA23_PARENT_UE.
- GMR repo: ~/sparkpack/GMR_Grab (uv venv .venv). Robots defined in
  general_motion_retargeting/params.py (xml paths, ik config paths, root
  body, height). Pattern to copy: atlas_fists / t800 entries and
  ik_configs/soma_bvh_to_atlas_fists.json.
- Retarget runner: GMR_Grab/scripts/retarget_headless.py
  (--bvh_list --shard i/N --robot X --out_dir). BVH list:
  ~/sparkpack/output/all_bvh.list (8,730 unique stems). The v6 corpus is a
  142-stem subset; stems = basenames of motion_files in
  data/atlas_pretrain_corpus_v6.pt.
- npz -> proto converter: data/scripts/convert_gmr_pkl_to_proto.py (check
  its robot assumptions; it consumed atlas npz with robot_type field — may
  need a samurai branch pointing at the new MJCF for FK).
- Corpus recipe/weights: rebuild like build_v6 did — Reallusion 2x,
  shadow_boxing 1x (Eric downweighted late), dodges 2x, locomotion 1x,
  getups 1x with 2x SPEED (halve motion_dt & motion_lengths, double
  gvs/gavs/dvs) and kip-ups NOT sped. Verify against
  data/atlas_pretrain_corpus_v6.pt weights (motion_weights per stem).
- MJCF zero-pose convention: soma23_humanoid.xml zero = T-pose, ALL body
  frames world-aligned, arms +-X, faces -Y (SOMA23_TPOSE_POS docstring).
  The samurai robot must match this convention (see step 2).

## Design decisions (made deliberately)
1. The robot uses EXACTLY the 23 bones of SOMA23_TO_UE (pelvis, spine_02,
   spine_04, spine_05, neck_01, neck_02, head, clavicles, upperarms,
   lowerarms, hands, thighs, calves, feet, balls). Twist/corrective/cloth/
   finger bones are NOT articulated: their skinned vertices are reassigned
   to the nearest articulated ancestor for geometry fitting.
   Rationale: mirrors soma23's structure; GMR ik config and later tooling
   map 1:1; DOF count stays RL-sized.
2. Robot bodies are NAMED with the SOMA names (Hips, Spine1, ... RightToeBase)
   not the Epic names, so every existing tool (overlay maps, converters,
   robot-config patterns) applies verbatim. The Epic name is kept as a
   comment per body in the MJCF.
3. MJCF zero pose = T-POSE with world-aligned frames (soma convention),
   NOT the A-pose bind. rig2mjcf re-poses the arm chain bind to T using
   conj(UE_REST_REL[body]) before emitting local frames/geoms. Validation:
   FK at qpos0 must give identity rel-to-hips rotations and arm along +-X.
4. Joints: every non-root body gets a 3-hinge stack (x,y,z) like soma23,
   with limits from a per-SOMA-name table (knee/elbow tighter, spine
   moderate). Copy soma23_humanoid.xml's actuator/gain pattern; scale
   effort by downstream mass share.
5. Scale: emit in METERS. Character units are cm-ish: measure hips height
   from bind (pelvis z) and scale all geometry by (0.94 / pelvis_bind_z)
   ... actual: use the auto-fit scale already computed for the overlay
   (~0.0105) — recompute in-script from SOMA23_TPOSE_POS similarity fit.
6. Collision: capsules fit per articulated bone from its dominant-weight
   vertices (in bone-local T-posed frame): axis = bone->child direction,
   radius = 80th percentile radial distance (cap at sane values); feet get
   boxes. Self-collision: contype/conaffinity pattern copied from soma23.
   Cloth/armor bones (skirt etc.) contribute NO collision geometry.

## Pipeline steps in detail
1. data/scripts/rig2mjcf.py (Blender-free; needs pxr + numpy — use
   ~/anaconda3/bin/python, NOT the isaacsim venv (no pxr) and NOT system
   python). Inputs: character usd, bone map (SOMA23_TO_UE), rest-rel
   (UE_REST_REL), soma tpose (SOMA23_TPOSE_POS), output mjcf path
   protomotions/data/assets/mjcf/samurai.xml.
   Steps: parse skeleton/bind/weights -> reassign vertices to articulated
   ancestors -> compute fit scale -> re-pose arms to T -> per-bone local
   frames (world-aligned at T, translation = scaled bind offsets) ->
   capsule fit -> masses (density 985 kg/m3 human-ish; total target ~75kg,
   renormalize) -> joints+limits+actuators -> write MJCF.
2. Validate with mujoco (GMR_Grab venv has mujoco):
   a. FK at qpos0: body world quats ~identity rel to Hips; arm chain +-X;
      toes -Y; total mass 60-90 kg; hips height ~0.94.
   b. Drop test: 500 steps with position actuators holding qpos0; robot
      must remain standing (root z within 15% of start).
3. GMR: add 'samurai' to GMR_Grab params.py (ROBOT_XML_DICT ->
   assets/samurai/samurai.xml — copy the mjcf + meshes into GMR assets;
   IK_CONFIG_DICT soma_bvh -> ik_configs/soma_bvh_to_samurai.json;
   ROBOT_BASE_DICT 'Hips'; height table entry). Create the ik config by
   copying soma_bvh_to_atlas_fists.json and renaming robot bodies to the
   soma-named samurai bodies (mapping is IDENTITY on soma names!); zero out
   atlas-specific rotation offsets (samurai frames are world-aligned at T,
   same as the soma bvh convention, so offsets should be identity to start;
   head/hand offsets tunable later). Sanity: single-clip retarget of a
   shadow_boxing bvh; inspect dof ranges (no stuck-at-pi, see the arm
   branch-flip incident in t800_retarget_repair_list.txt notes).
4. Retarget the 142 stems: filter ~/sparkpack/output/all_bvh.list to the
   stems of data/atlas_pretrain_corpus_v6.pt -> 16-shard
   retarget_headless.py --robot samurai fleet (CPU-only, minutes).
5. ProtoMotions: protomotions/robot_configs/samurai.py — copy soma23.py
   structure: body_names in MJCF order, dof names (x/y/z per joint),
   control gains per mass class, contact bodies (feet/hands/head/torso),
   trackable subset (Hips, Head, feet, hands), default_root_height,
   asset file names (mjcf/samurai.xml, usd once converted). Register in
   robot_configs/factory.py the same way soma23/atlas are.
6. Convert + package: extend convert_gmr_pkl_to_proto.py (or a sibling
   script) to accept the samurai MJCF; emit data/motions/samurai_v6/;
   package MotionLib with the v6 weight recipe (copy weights per stem from
   atlas v6); apply the 2x getup speedup; save
   data/samurai_pretrain_corpus_v6.pt. Quality gates:
   scan_motion_lib_quality.py (--dof-spike-max 50 --body-speed-max 15
   --sat-joints-max 12) + the stuck-at-pi arm sweep.
7. MJCF -> USD: use the same conversion used for atlas
   (usd_asset_file_name; see Elements RETARGET_ON_4090.md 'code/usd_convert'
   and protomotions/data/assets/usd/atlas as the output pattern). For a
   first training run this can be deferred: IsaacLab can also load MJCF
   robots — check robot_config.asset usage for soma23 (it trains from
   mjcf usd... check). If a USD is required, the visual can be capsules
   first; the skinned overlay provides the pretty rendering anyway
   (identity mapping: drive red_samurai.usd from the samurai robot state
   with joint_map = {soma_name: epic_bone} = SOMA23_TO_UE — already usable
   in the viewer).
8. ASE: same experiment as atlas — examples/experiments/ase/mlp.py,
   --robot-name samurai --motion-file data/samurai_pretrain_corpus_v6.pt
   --num-envs 4096 --batch-size 8192 --experiment-name
   samurai_ase_pretrain_v6. GPU 0 or 2 only. Expect IsaacLab to need the
   robot registered for sim (actuator/config sanity) — mirror soma23's
   simulation_params.

## Known pitfalls (learned this week, the hard way)
- Motion-lib grs quats are XYZW; IsaacLab native is WXYZ; the common state
  layer converts to xyzw. Every "upside down" bug this week was a layout
  misread. rig2mjcf touches only bind matrices (no quats from motion data).
- Blender's USD export writes UsdPreviewSurface only -> renders BLACK in
  RTX. data/scripts/add_mdl_context.py fixes that (already applied to the
  overlay USDs).
- The A-pose: any place that assumes character bind == robot T-pose must
  use UE_REST_REL. rig2mjcf sidesteps this by re-posing to T at generation.
- retarget_headless fleet: pgrep patterns match your own monitor shells —
  count with care; workers need SIGKILL sometimes.
- The machine has crashed 3 nights running under full load (power
  suspected, NVMe symptom treated with a GRUB APST flag that may not be
  applied yet — check /proc/cmdline). Checkpoints save every 10 epochs;
  0-byte last.ckpt after a crash means restore from epoch_N000 snapshot.
- Trainings that were running before the last crash are PAUSED and Eric
  has not yet said to resume: t800_ase_pretrain_v6 (resume from last.ckpt
  == epoch_14000 restore), atlas_ase_pretrain_v6 (last.ckpt, epoch 13770),
  soma_sft_combat_v6 (last.ckpt, epoch 2370). GPU assignments Eric gave:
  SFT gpu0, atlas gpu1 (yes, display — his explicit call), t800 gpu2.
- Also still open: bake overlay defaults into record_fight.sh /
  battle_tournament.py (red vs gray samurai, ambient 500, ring lights) and
  a keypad-5 show/hide-robot toggle for inference_agent (the kinematic
  viewer has one; simulator custom_key_handlers must be passed at
  construction — see motion_libs_visualizer.py:486 and
  isaaclab/simulator.py:349).

## OPEN ISSUE (2026-07-26 end of session): skin hands vs robot hands
Eric reports the samurai skin's hands don't line up with the robot's hand
capsules (everything else aligns after SAMURAI_TPOSE_POS fit + head trim).
Facts for the next session:
- Chain math says positions should coincide (offset formulas identical on
  both sides; rest==bind verified; scales match). So the mismatch is
  probably ORIENTATION, not translation: UE_REST_REL sets
  c(Hand) := c(ForeArm) (direction-derived, no wrist twist). If the
  character's bind wrist carries roll relative to the forearm, the gauntlet
  pivots around the wrist and its volume sits centimeters off the capsule
  while the wrist joint itself is aligned. Fix candidate: derive c(Hand)
  from the hand's BIND ORIENTATION (not the forearm segment direction) —
  e.g. c_hand = rotation taking the robot-T hand frame to the char bind
  hand frame projected to remove the arm-lowering component; or simplest:
  calibrate visually with a small roll offset knob.
- A runtime FK diagnostic exists in overlay.py sync (env OVERLAY_DIAG=1)
  but ITS OWN FK IS WRONG (printed 100m deltas while the render is close):
  fix its accumulation before trusting it (suspect: mixing skel-space bind
  quats with local rest quats in W, or the translations cache).
- Robot hand capsules are fist-sized (r=0.05) along +-X from the wrist;
  the skin gauntlet is bigger — a few cm of visual mismatch is inherent;
  ask Eric how large the offset actually looks before over-engineering.

## NEXT TASK (Eric, 2026-07-26): kick-attempt shaping reward for the league
Problem: league v6 shows no kicks (RL prunes risky moves; punch meta).
Eric's design — implement in the battle env dense rewards:
- Event: a foot rises above a height threshold (suggest foot z > 0.75 m,
  i.e. waist-ish; knee-lifts don't count), with hysteresis: the foot must
  drop below ~0.4 m before it can score again.
- Reward: small fixed bonus per event (start ~0.5, tune vs KE hit rewards),
  capped at 3 events PER FOOT per episode (LeftFoot and RightFoot counted
  separately -> max 6 bonuses/episode). Counters reset on env reset.
- Scope: dense reward component (so it anneals with --dense-reward-scale
  as the league matures, like the other shaping terms).
- Files: protomotions/envs/battle/factories.py
  (default_battle_reward_components) + wherever per-episode state lives
  (battle control resets); foot bodies via
  robot_config common naming all_left/right_foot_bodies.
- Optional refinement if lifted-leg cheese appears: require foot speed
  > 2 m/s at event time so only dynamic lifts (actual kick attempts) pay.

## NEXT TASK (Eric, priority): paper-faithful ASE battle league (frozen LLC + HLC)
Decision: rebuild the ASE league to match the ASE paper (Peng 2022), which
also mirrors the GPC league's frozen-prior structure.
- Stage 1 (exists): examples/experiments/ase/mlp.py pretrains the
  latent-conditioned low-level controller (LLC). atlas/t800 v6 pretrains
  are these LLCs; they can keep training and be swapped under old HLCs.
- Stage 2 (BUILD THIS): new experiment examples/experiments/ase/
  battle_league_ase_hlc.py:
  * Load LLC policy from --llc-checkpoint (PretrainedModelConfig, like
    battle_league_prior_peft.py loads --prior-checkpoint). FREEZE it
    (requires_grad False, eval mode, no optimizer params).
  * High-level policy: small MLP (e.g. 3x512), inputs = battle task obs
    (+ self obs), outputs = 64-dim latent ACTION (continuous, tanh or
    normalized to the hypersphere like sample_latents does — check
    protomotions/agents/ase/agent.py store_latents/sample_latents for the
    latent normalization convention and reuse it).
  * Each control step: HLC emits z (optionally at a slower rate, e.g.
    every 2-5 steps like the paper); LLC(z, proprio) emits joint actions.
  * League machinery: reuse agents/league/ase_agent.py lanes but
    snapshot/restore ONLY the HLC weights (LLC shared+frozen across all
    snapshots — that is what share_frozen_base_with was scaffolded for).
  * Rewards: unchanged battle rewards (incl. kick_attempt_bonus).
    AMP/discriminator terms NOT needed in stage 2 (style is guaranteed by
    the frozen LLC) — matches both the paper and the GPC league.
  * Warm start: none needed; HLC trains from scratch (small, fast).
- Old full-weight league (battle_league_ase.py) stays for comparison.
- Launch pattern once built:
    python protomotions/train_agent.py --robot-name atlas ... \
      --experiment-path examples/experiments/ase/battle_league_ase_hlc.py \
      --motion-file data/atlas_pretrain_corpus_v6.pt \
      --llc-checkpoint results/atlas_ase_pretrain_v6/last.ckpt \
      --num-envs 256 --batch-size 512 \
      --experiment-name atlas_ase_battle_hlc_v1
- Payoff: LLC pretraining can continue independently; a deeper LLC can be
  swapped under an existing HLC (works, degraded, retrainable) — Eric's
  requested workflow; and HLC league snapshots are tiny.

### STATUS 2026-07-29: BUILT + SMOKE-PASSED
Implemented as examples/experiments/ase/battle_league_ase_hlc.py +
protomotions/agents/league/ase_hlc_agent.py (league orchestration shared via
agents/league/full_model_league.py mixin; ase_agent.py refactored onto it,
PEFT league untouched). Atlas smoke (64 envs, 19 epochs): frozen LLC loaded
from atlas_ase_pretrain_v6, league seeded, 184 matches vs seed at ~50 % win
rate, snapshots are HLC-only (~12 MB, architecture "ase_hlc"), clean exit +
checkpoint resume verified. BattleTournament works unchanged (action_hold
holds the latent while the LLC re-runs each step = the paper's slow-HLC
cadence). Fixed in passing: robot_tables derives kick-bonus foot bodies for
non-SMPL robots. Real run launch = the command in the file's docstring.
