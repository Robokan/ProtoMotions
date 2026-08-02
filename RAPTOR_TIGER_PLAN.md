# Raptor & Tiger: FBX creatures as ASE-trainable robots

**Eric's directive (2026-08-01 evening)**: generalize the samurai
FBX-skeleton→robot approach beyond humanoids so the Raptor and Tiger in
`~/sparkpack/UnrealExportedAssets` become robots that can be retargeted
and trained with ASE — battle integration later. GPU 1 runs Eric's
`t800_ase_pretrain_v8` (untouchable); GPU 0 free for smokes.

## Status

- [x] `go2-training` merged into `battle` (commit a18f559 + test fix
  a47df24): quadruped robots (go2/anymal_d/dog_v2), quadruped env/pose
  semantics, motion tools. Battle smoke passed post-merge (27 epochs,
  mixed-arena league).
- [x] Asset survey complete (trees in the session scratchpad:
  `raptor_tree.txt`, `tiger_tree.txt`).
- [x] `data/scripts/fbx2robot.py` SHIPPED (commit 771d905): raptor
  27 bodies/78 dof/40 kg (hips 0.51 m); tiger 29 bodies/84 dof/200 kg
  (pelvis 0.86 m; its bind has a -1.15 m Y offset — harmless, root
  motion comes from anims). MJCFs at protomotions/data/assets/mjcf/
  {raptor,tiger}.xml. NOTE: tiger up_axis="y" (explicit in ROBOTS spec).
- [ ] NEXT: robot_configs/raptor.py + tiger.py (dog_v2 pattern:
  semantic naming front-feet="hand", trackable subset, ControlConfig
  gains from actuator efforts, default_root_height 0.51/0.86,
  contact_bodies) + factory registration.
- [ ] NEXT: data/scripts/fbx_anim_to_motion.py — same-skeleton FK
  converter (plan item 3; world-rotation copy for kept joints with
  skipped-bone folding; qpos euler_xyz; use convert_gmr_pkl_to_proto's
  fk_from_transforms_with_velocities pattern with each robot's
  kinematic_info; contact labels; cm->m + up-axis like fbx2robot).
- [ ] Raptor corpus from Animations/RootMotion (~dozens of clips);
  tiger animation FBXs still to inventory (find dir).
- [ ] ASE boot smoke per robot on GPU 0 (mlp.py, few epochs).

## Assets

- Raptor: `UnrealExportedAssets/Raptor/Game/RaptorDinosaur/`
  - `Model/Raptor_Gameplay.FBX` — skeleton + skinned mesh (LOD0-2),
    114 bones, cm units, hips at ~51 cm.
  - `Animations/RootMotion/*.FBX` — DOZENS of clips on the SAME skeleton:
    attacks (BackBite, RKick, StrafeLeft, Kick...), locomotion
    (walk/run/dive/swim/climb), knockdowns, idles, `bindpose.FBX`.
- Tiger: `UnrealExportedAssets/Tiger/Game/Animalia/Tiger_M/`
  - `Meshes/Tiger_M.FBX` (mesh+skeleton, LOD0-4, pelvis ~86 cm up);
    `Tiger_M_Bones.FBX`; animation FBXs to inventory (check
    `Animations/` sibling dirs).

## Key design decisions

1. **Topology comes FROM the FBX skeleton** (unlike rig2mjcf.py, which
   maps onto the fixed soma23 humanoid template). A curated KEEP-list
   per robot selects the articulated subset; skipped intermediate bones
   are collapsed (their transforms folded into the kept chain).
2. **No GMR needed**: the packs' animation FBXs share the robot's own
   skeleton — "retargeting" is direct FK: evaluate each frame's local
   rotations for kept joints → qpos (euler per hinge triplet, framework
   `euler_xyz` convention) → `fk_from_transforms_with_velocities` →
   `.motion`. Root motion from the Hips/RigPelvis world track (cm→m).
3. **ufbx landmines** (hard-won): one scene per process, module-level
   scene KEEPALIVE list, fetch `scene.anim` once, access ONE attribute
   family per pass (mixing n.bone/n.mesh/anim_stacks in one loop
   segfaults), `gc.disable()`, `os._exit(0)`.
4. **MJCF conventions** (from dog_v2/t800/atlas experience): 3 hinge
   joints (x,y,z) per articulated body — NEVER multi-joint D6 merges
   (IsaacLab importer breaks PD); identity root quat (bake if the rig
   root is rotated — see ATLAS_ROOT_BAKE_QUAT pattern); capsule geoms
   along parent→child bone axis, radius heuristic from bone length
   (skin-weight fitting like rig2mjcf is the v2 refinement); explicit
   density → target mass (raptor ~40 kg, tiger ~200 kg); EngineAI-class
   strength-to-weight actuators (~4 Nm/kg primaries, retune_atlas
   JOINT_SPEC pattern); generous ±90° hinge ranges v1, mine真 ranges
   from the anim corpus later (rig2mjcf DOF_RANGES approach).
5. **Robot config**: follow `dog_v2.py` — quadruped semantic naming
   (front feet = "hand" bodies for the battle tables), trackable subset
   = trunk + chain ends, `apply_default_visual_material=False` later
   when the skinned overlay lands.
6. **USD**: `usd_convert/flatten_mjcf.py` →
   `convert_robot_mjcf_to_usda.py` (nomesh capsules render fine for v1;
   skeletal-mesh overlay via SkinnedOverlay is the follow-up — needs an
   overlay_map for each creature like SOMA23_TO_UE).

## Articulation keep-lists (from the surveyed trees)

Raptor (~27 bodies): Hips; Spine, Spine1; Neck, Neck1, Neck3 (collapse
Neck2); Head; Jaw (bite = strike body); Tail1, Tail3, Tail5 (collapse
evens, drop 6+); per side: UpLeg, Leg, Foot, ToeBase; Shoulder, Arm,
ForeArm, Hand. PRUNE: cameras, FootIK, LodGroup/LODs, toe/finger
phalanx chains (Index/Middle/Ring/Thumb*), eyes/lids, Tongue*, Snout,
Throat, Belly, Dupa, PreySocket, props, ToeBaseEND.

Tiger (~28, tree confirmed): RigPelvis; RigSpine1, RigSpine3 (collapse
Spine2), RigChest; RigNeck1, RigNeck3 (collapse 2, fold 4 into head
offset); RigHead; RigJaw1 (bite); RigTail1, RigTail3, RigTail5
(collapse evens, drop 6+); hind per side: RigLBLeg1, RigLBLeg2,
RigLBLeg3, RigLBLegAnkle (digitigrade foot); front per side:
RigLFLegCollarbone, RigLFLeg1, RigLFLeg2, RigLFLeg3, RigLFLegAnkle.
PRUNE: Digit*/Claw chains (Ankle = foot contact body), ShoulderBlade*,
ears/eyes/eyelids/whiskers/nose/tongue, LODs; RigRoot folds into root.
Trees saved: ~/sparkpack/output/{raptor,tiger}_tree.txt.

## Pipeline to build (fbx2robot.py + fbx_anim_to_motion.py)

1. `fbx2robot.py --fbx <mesh fbx> --keep <robot>.keep.yaml --out
   <mjcf>` : bind pose (evaluate_transform at t0 of bindpose/mesh FBX)
   → kept-joint tree with collapsed offsets → MJCF (conventions above)
   → verify with mujoco (mass, dof count, no locked joints).
2. Generate `protomotions/robot_configs/raptor.py` / `tiger.py`
   (dog_v2 pattern), register in factory.
3. `fbx_anim_to_motion.py --robot raptor --anim-dir .../RootMotion` :
   per-clip FK → `.motion` (contact labels via
   compute_contact_labels_from_pos_and_vel; height fix vs ground).
4. Recipe → corpus → `mlp.py` ASE pretrain boot smoke on GPU 0
   (few epochs only, then free the GPU).

## Gotchas expected

- UE cm units + axis conventions: raptor local offsets are cm; world
  FK must land Z-up meters (check against hips ~0.51 m / pelvis
  ~0.86 m standing).
- The mesh FBX's `Unreal Take` anim duration is 0.00 s — bind pose
  only; use it (or bindpose.FBX) for the bind, animation FBXs for
  motion.
- Collapsed-bone rotation folding: when skipping Neck2/Tail2 etc., the
  kept child's bind offset = product of skipped local transforms.
  During anim conversion the skipped bones' ROTATIONS must also be
  folded into the kept joint (world-rotation copy of the kept bone —
  compute kept-joint LOCAL rotation as inv(parent_kept_world) @
  kept_world, exactly like retarget_to_soma's world-copy trick).
- Tail/jaw dof ranges: bite needs a real jaw range (~25°); tail ±45°.
- `_derive_generic` in `envs/battle/robot_tables.py` expects humanoid
  semantic names — battle tables for creatures come LATER (Eric said
  training first, battle after).
