# Dog model rebuild plan (resume notes)

## RESOLVED #2 (2026-06-15): correct+fast decomposition (analytic_xyz)
The full library popped on ALL clips because the fast `euler_xyz` method used the
reversed (IsaacGym) convention Rz*Ry*Rx, but MuJoCo composes the 3 stacked x,y,z
hinges as R = Rx*Ry*Rz -> 168 deg FK error = garbage. `sequential` was correct but
~17h for 102 clips (per-frame Python coordinate descent).
FIX: new `analytic_xyz` multi-DOF method in pose_lib.py — closed-form extraction
for R = Rx*Ry*Rz (y=asin(R[0,2]), z=atan2(-R[0,1],R[0,0]), x=atan2(-R[1,2],R[2,2]),
gimbal branch when cos y ~ 0), vectorized, with a torch.cumsum phase-unwrap. Raises
to `sequential` fallback unless the body's 3 hinges are exactly identity X,Y,Z.
RESULT: FK residual 0.0003-0.0005 deg (vs 168), full 102-clip library in ~8 s.
Convert the dog with `--multi-dof-method analytic_xyz`. dog_full.pt regenerated
(102 motions, 21 bodies, 60 dof). Viewer confirmed clean.

## RESOLVED (2026-06-15): BVH-matched cylinder dog works
The dm_control-framed approaches (surgical scapula/elbow fixes) never fully fixed
the shaking/arch. The fix was the full rebuild the owner directed: a NEW skeleton
matching the BVH mocap 1:1 (generate_dog_mjcf.py: root 'trunk'=Hips, 21 bodies,
3 ORTHOGONAL hinges X/Y/Z per body, capsule "bones", offsets = BVH*CM_TO_M), with
an IDENTITY retarget (retarget_bvh_to_dog.py copies BVH local quaternions verbatim;
only the root is rotated Y-up->Z-up). This gives a level back (mocap's own 2-joint
spine, no 7-lumbar over-distribution) and no shaking (orthogonal axes).
KEY LAST FIX: hinges must be `limited="false"` (unlimited). The quaternion->3-hinge
euler decomposition occasionally wraps an angle past +-pi (an IDENTICAL rotation);
a +-pi LIMIT would clamp the wrapped value and pop the leg for one frame. Unlimited
hinges => no clamp => popping gone (owner confirmed visually). Max single-frame jump
dropped 360deg -> 97deg; ~6 residual gimbal frames are not visually significant.
robot_configs/dog_v2.py updated to the BVH body/joint names (anchor 'trunk', feet =
Left/RightFoot + Left/RightHand, CONTROL_OVERRIDES cover <Body>_[xyz]).
NOTE: SMPL/humanoid in this repo also use 3 hinges (not ball joints) with +-180
limited and work — they just don't drive joints to the wrap boundary like the dog's
BVH does. Ball joints are NOT needed.
Viewer: env_kinematic_playback.py gained N/next-clip and P/prev-clip keys.
NEXT: regenerate full 102-clip library (in progress), smoke-train dog on Newton,
then bone meshes (task #10) + spine/tail cosmetic subdivision.

---


## Why we are rebuilding
The current dog (`protomotions/data/assets/mjcf/dog_v2_nomesh.xml`, derived from
dm_control dog_v2) does NOT match the BVH mocap's DOF structure, which makes
retargeting fail:

- **Scapula 3-DOF axes are non-orthogonal** (off-axis dot 0.29 ≈ 73°, not 90°).
  Sequential hinge decomposition of a rotation into skewed axes is
  ill-conditioned → the twist DOF flips up to ~112°/frame → front legs/feet
  shake. (Other multi-DOF bodies — lumbar, cervical, upper_leg, upper_arm —
  are orthogonal.)
- **Elbow/wrist are 1-DOF** but the BVH forearm/hand carry full 3-DOF rotation,
  so the off-axis part can't be represented → more flipping.
- **Spine rest geometry** gives an unnatural head-up, pelvis-high topline
  (the lumbar axes are orthogonal, but the bone offsets/frames arch the back).

Proven facts (measured):
- Source BVH is smooth (arm joints <9°/frame).
- Retarget NPY output is smooth (scapula local <5.3°/frame) — **the retarget
  logic is correct**.
- The converter's `sequential` multi-DOF decomposition is where smooth input
  becomes jumping joint angles (112°/frame) whose FK diverges 108° from the
  smooth target. Temporal-continuity and limit-clamping both had ZERO effect
  (byte-identical) — the problem is the model's joint frames, not the converter.

## Rebuild design (owner's direction)
Build a NEW dog skeleton that matches the BVH mocap's DOF exactly, then add
cosmetic extra DOF in spine/tail for realism.

### Source BVH skeleton (`/home/bizon/eric/Mode Adaptive/mocap/*.bvh`, 60fps, Y-up, cm)
21 joints + endsites. Hierarchy (DOG mocap — humanoid joint names):
```
Hips (root)
  Spine -> Spine1
    Neck -> Head
    LeftShoulder -> LeftArm -> LeftForeArm -> LeftHand   (FRONT-LEFT leg)
    RightShoulder-> RightArm-> RightForeArm-> RightHand   (FRONT-RIGHT leg)
  LeftUpLeg -> LeftLeg -> LeftFoot     (HIND-LEFT leg)
  RightUpLeg-> RightLeg-> RightFoot    (HIND-RIGHT leg)
  Tail -> Tail1
```
Each non-root BVH joint carries a 3-DOF (euler) rotation. Offsets are in the
BVH HIERARCHY block (read with data/scripts/poselib_vendor bvh parser:
`parse_bvh_file` / `load_bvh_zup`). Standing Hips height ≈ 46.8 cm.

### Target MJCF (new, replaces dog_v2_nomesh.xml)
- **Body tree = BVH tree, 1:1** (root `trunk`=Hips, then spine/spine1, neck/head,
  4 legs, tail). Rename to match the BVH joints so the retarget is identity.
- **Per non-root body: 3 ORTHOGONAL hinges** in a fixed known order (e.g. X then
  Y then Z, axes = world/identity in the body's rest frame). Orthogonal +
  fixed-order = the sequential decomposition is exact and stable. (Alternative:
  a single ball/free joint per body — only if the ProtoMotions pose_lib + PD
  action path supports ball-joint qpos cleanly; default to 3 orthogonal hinges,
  which the existing pipeline already handles for orthogonal cases.)
- **Body offsets** = BVH OFFSET values scaled cm->m by the standing-height ratio
  (≈0.01018, same as the retargeter uses), so FK reproduces the mocap exactly.
- **Colliders**: capsule per bone segment (parent->child offset), spheres at
  paws, box at trunk. Visual only need not be physical-accurate.
- **Actuators**: one position actuator per hinge (PD), gains in a sane range
  (start from current dog_v2.py values; tune later — sim-only).
- **Realism extras (after the core works)**: subdivide the Spine->Spine1 path and
  Tail->Tail1 path into N intermediate bodies that each receive a FRACTION of the
  parent joint rotation (proportional/cosmetic), purely for looks. These are not
  driven by distinct mocap DOF.

### Retarget (huge simplification)
With the target matching the BVH DOF, `retarget_bvh_to_dog.py` becomes nearly
identity:
1. Parse BVH, Y-up->Z-up ([0.7071,0.7071,0,0]), scale root cm->m.
2. For each body, take the BVH joint's LOCAL rotation directly as the target
   local rotation (no chain slerp distribution, no reference-pose delta, no IK).
3. Root trajectory from Hips, scaled.
4. Mirror as before (L/R swap) for `--mirror`.
The cosmetic spine/tail subdivision bodies each get `parent_rotation ** (1/N)`
(slerp from identity) so the bend distributes smoothly.

### Conversion
`convert_quadruped_poselib_to_proto.py --multi-dof-method sequential
--temporal-continuity` should now produce SMOOTH, FAITHFUL joint angles because
the axes are orthogonal and ordered. Verify residual ≈ 0.

### Bone meshes (visualization)
Re-map the 162 anatomical STLs (already copied to
`protomotions/data/assets/mesh/dog_v2/`, originally from dm_control
suite/dog_assets) onto the new body tree by anatomical region (skull->head,
vertebrae->spine/neck/tail segments, limb bones->leg segments, scapula/humerus/
radius/ulna->front-leg segments, femur/tibia->hind-leg segments). Attach as
visual-only geoms (contype=0 conaffinity=0). Approximate placement is fine; the
point is a readable skeleton in the Newton viewer.

## Verification (do at each step, with SCREENSHOTS — the metrics lied before)
1. `compare_dog_retarget.py`: per-paw contact, gait-phase corr.
2. Jitter: per-hinge mean|2nd diff| and max single-frame jump on the front-leg
   joints — must be near the body median (~0.001-0.01), no 100°+ jumps.
3. Decomposition residual ≈ 0 (FK of extracted dof matches target rotation).
4. **Newton bone viewer screenshot** (the decisive check): level topline,
   natural head, feet planted, no shaking. Capture via:
   `DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority import -window <Newton Viewer wid> out.png`
   then read the PNG. The viewer driver:
   `examples/env_kinematic_playback.py --robot-name dog_v2 --simulator newton
   --num-envs 1 --motion-file <walk.pt> --experiment-path
   examples/experiments/mimic/quadruped_mlp.py` (NO --headless; needs pyglet,
   already installed). Pin one clip with
   `--overrides "env.motion_manager.subset_method=[<id>]"`.

## What is saved on disk (branch go2-training, NOT committed)
- retarget_bvh_to_dog.py, generate_dog_mjcf.py (bones + widened limits),
  dog_v2_nomesh.xml, convert_quadruped_poselib_to_proto.py (temporal-continuity
  + limit-aware), protomotions/robot_configs/dog_v2.py (+factory entry),
  data/scripts/poselib_vendor/, compare_dog_retarget.py, verify_dog_limits.py,
  verify_dog_retarget.py, RETARGETER_NOTES.md, 162 bone STLs in
  protomotions/data/assets/mesh/dog_v2/.
- The Go2/ANYmal-D work + motion-support terrain is COMMITTED+pushed to fork
  Robokan/ProtoMotions branch go2-training (commits up to 73a2334). The dog work
  is all uncommitted working-tree changes.

## Resume entry point
Rebuild generate_dog_mjcf.py to emit the BVH-matched skeleton (3 orthogonal
hinges/body), simplify retarget_bvh_to_dog.py to identity local-copy, regenerate
clip 37, screenshot the bone viewer. Iterate on spine/tail cosmetic segments
once the core gait is clean.

## UPDATE (2026-06-13): bone-alignment problem + corrected approach

### Current messy working-tree state (after a half-done full rebuild)
- `generate_dog_mjcf.py` was OVERWRITTEN to build a NEW 21-body BVH-matched
  skeleton (root `trunk`, 60 hinges + free root). Loads in MuJoCo + Newton.
  **The bone-mesh attach code is GONE from it** (the previous bone version was
  untracked and is overwritten — must be reconstructed).
- `dog_v2_nomesh.xml` is now that new BVH-matched skeleton: BVH proportions,
  NO bone meshes.
- `retarget_bvh_to_dog.py` is still the OLD (delta + chain-slerp + IK) version —
  NOT yet simplified to identity.
- The 162 bone STLs are still safe in `protomotions/data/assets/mesh/dog_v2/`.

### The bone-alignment finding (why full rebuild is risky)
Bones lined up perfectly on the OLD dm_control-framed model (proven in
screenshots) because they were authored for those body frames/offsets. A full
BVH-proportioned rebuild changes segment lengths, so the 162 STLs no longer span
their segments — they'd need per-bone refit (scale to segment length, orient
along bone axis, reposition). Approximate + laborious.

### Decomposition bug is LOCALIZED (key insight)
Measured: the SPINE decomposition was already smooth (lumbar/cervical axes
orthogonal). Only the SCAPULA (non-orthogonal 3-DOF axes, off-dot 0.29) and the
ELBOW/WRIST (1-DOF can't hold the BVH forearm/hand 3-DOF) flip. So a full rebuild
is overkill.

### RECOMMENDED narrower approach (preserves bone alignment by construction)
Invariant: NEVER change a segment's length or which body a bone attaches to —
only joint AXES and REST ANGLES. Then bones stay aligned (they rotate with their
body as attached children).
1. Start from the OLD dm_control-framed model (the one WITH bones — must be
   reconstructed since the generator was overwritten; the bone-attach logic:
   parse dm_control suite/dog.xml `class="bone"` mesh geoms per body, emit as
   visual-only contype=0 conaffinity=0, meshdir ../mesh/dog_v2/).
2. Orthogonalize the scapula_L/R 3 hinge axes (make supinate/abduct/extend
   mutually perpendicular) -> fixes front-leg shaking. Axes only; bones unmoved.
3. Add a supinate DOF to elbow/wrist (or otherwise let the forearm/hand twist be
   representable) -> removes the remaining front-leg flips. Axes only.
4. Adjust spine/neck REST angles (default joint positions) to flatten the
   head-up/pelvis-high topline. Bodies rotate, bones rotate with them, stay
   aligned.
5. Re-verify: residual≈0, jitter near zero, and SCREENSHOT the Newton bone
   viewer (level topline, planted feet, no shake).

### Alternative (owner's original idea): keep the new BVH-matched skeleton
If chosen, finish the identity retarget rewrite AND add per-bone refitting:
for each new segment, scale/orient/position the corresponding STL to span it.
More work, approximate bones, but a maximally clean retarget. Decide before
proceeding.

## UPDATE (2026-06-13, second pass): surgical scapula+elbow/wrist fix DONE

Chose the NARROW approach (preserve bone alignment by construction). All changes
are axis/DOF-level only; no body offset, no bone attachment moved.

### What was rebuilt / changed
- `data/scripts/generate_dog_mjcf.py` REWRITTEN to emit the dm_control-framed dog
  again. It now transforms `.../dm_control/suite/dog.xml` (which already carries
  the bone mesh geoms + collision primitives + joint default classes): drops the
  ball/floor/walls/cameras/lights, WELDS the 4 anchor bodies (foot_anchor_L/R,
  hand_anchor_L/R — jointless) into their parents, sets bone geoms visual-only
  (contype=0 conaffinity=0 group=2, meshdir ../mesh/dog_v2/), widens hinge ranges,
  applies the two fixes below, and adds one PD position actuator per hinge.
  Result: nbody=58 (57 bodies + world, root 'torso'), nq=84, nu=77,
  nmesh=162 (all bone meshes present), mass 10.24 kg, rest torso z 0.4152 m.
  Compiles in MuJoCo and imports in Newton (body_count=57, dof=83).
- `protomotions/data/assets/mjcf/dog_v2_nomesh.xml` regenerated accordingly.
- `protomotions/robot_configs/dog_v2.py`: added CONTROL_OVERRIDES regex for the
  two new twist DOFs (elbow_[LR]_supinate, wrist_[LR]_supinate). Everything else
  (root 'torso', feet foot_L/R + hand_L/R, factory entry 'dog_v2') unchanged.
  Config loads: 57 bodies, 77 dofs.

### FIX 1 — scapula axes orthogonalized (the real shake source) — SOLVED
dm_control scapula extend axis was "0.3 1 0" (dot ~0.29 with abduct "1 0 0").
Replaced the three scapula_L/R axes with an orthonormal frame spanning the same
3-DOF space: supinate "0 0 1", abduct "1 0 0", extend "0 1 0" (R mirror-signed).
Result: scapula_{L,R}_supinate dof jitter dropped from the old ~112 deg/frame
flips to mean|2nd diff| 0.0035 rad / max 4.5 deg/frame — at the body median.

### FIX 2 — elbow/wrist twist DOF added — PARTIAL
Added one supinate hinge (axis "0 0 1", along the segment long axis) to
lower_arm_L/R (elbow) and hand_L/R (wrist), + matching actuators. Also
orthogonalized the elbow BEND axis "0 1 0.2" -> "0 1 0" (perp to the twist).

### Metrics (clip 37, --no-foot-ik, sequential + temporal-continuity)
- Decomposition/FK residual (FK of extracted dof vs target body rotations):
  0.006 deg mean, 0.079 deg max  (<< 2 deg target) — the dog reproduces the
  smooth mocac EXACTLY. The front-leg BONES (grs FK output) move smoothly:
  scapula 4.5, upper_arm 11, lower_arm/hand ~26 deg/frame MAX (natural swing,
  mean ~4 deg/frame); paws move <=7.5 cm/frame. NO bone shaking.
- scapula_{L,R}_supinate jitter: 0.0035/0.0041 rad, max jump 4.5/4.0 deg/frame.
- elbow_R/wrist_R: clean (13.5/14.8 deg/frame max jump).
- compare_dog_retarget gait-phase corr: front_L +0.80 front_R +0.51
  hind_L +0.64 hind_R +0.77 (all in phase). Range over-lifts vs source
  (expected, foot IK off).

### KNOWN RESIDUAL ARTIFACT (not a bone shake) — TODO
elbow_L (156 deg/frame) and wrist_L (119 deg/frame) still show large
DOF-SPACE jumps, while the R side is clean. Verified these are GIMBAL-EQUIVALENT
branch relabels, NOT motion: the target lower_arm_L local rotation is smooth
(<=24.6 deg/frame, mean 22 deg total), and FK(dof) residual is 0.006 deg — the
same smooth rotation is just expressed as (bend=-160,twist=180) vs (bend=-3,
twist=0) on adjacent frames. The bones therefore do NOT shake in kinematic
playback (grs is smooth). It only matters for BUILT_IN_PD training (PD target
would jump). Root cause is `_sequential_hinge_decomposition`'s frame-0 cold
solve picking the wrapped branch for the L 2-DOF bend+twist body; the per-frame
warm-start then tracks it. MJCF joint RANGES have NO effect here — this code
path (no --limit-aware) does not clamp, so widening/narrowing ranges gives
byte-identical output (confirmed). FIX requires a converter change (seed frame-0
toward the small-bend branch, or limit-aware clamp during decomposition), which
the plan says to DEFER. RECOMMENDED next step: run the converter WITH
--limit-aware (it clamps + plants to the now-finite elbow/wrist ranges and
re-runs the branch resolver) and re-measure; if still flipping, add a frame-0
branch-preference seed to `extract_qpos_from_transforms` for 2-DOF bodies.

### Still DEFERRED
- Spine topline rest-angle fix (#3).
- Foot-IK / refit / 102-clip batch.

### Artifacts left for the parent to screenshot
- /tmp/dogsurg/walk.pt  (clip 37, 856 frames, the bone viewer input)
- /tmp/dogsurg/npy/37.npy, /tmp/dogsurg/clips/37.motion
- /tmp/dogsurg/verify_surg.py  (jitter + residual measurement)
Nothing committed.

## UPDATE (2026-06-13, FINAL): FIX 2 changed to swing/twist projection — SOLVED

The "add a twist DOF to elbow/wrist" version of FIX 2 (above) made the new
wrist_*_supinate / elbow_*_supinate hinges the worst-popping joints (the bend+
twist 2-DOF pair gimbal-flipped ~157 deg/frame on the L side as it tried to
track the NOISY BVH forearm/hand twist). REVERTED. New FIX 2:

- elbow_L/R and wrist_L/R are back to their single 1-DOF dm_control BEND hinges
  (original axes kept: elbow "0 1 0.2", wrist "0 -1 0"). nq/nu drop back: now
  nbody=58 (57+world), nq=80, nu=73.
- The noisy off-axis forearm/hand twist is DISCARDED at conversion time by a new
  clean swing/twist projection: `protomotions/utils/rotations.py
  twist_angle_about_axis()` (quaternion swing-twist: theta = 2*atan2(v·a, w)),
  wired into the 1-DOF branch of `pose_lib.extract_qpos_from_transforms`
  (replacing the trace-based `angle_from_matrix_axis`, which LEAKED off-axis
  content into the bend angle and caused the pop). It is behavior-preserving for
  go2 / anymal_d (their 1-DOF leg rotations are already about-axis, so they get
  the SAME angle to numerical precision).
- elbow/wrist BEND range bounded to -140..60 so the temporal unwrap can reject a
  wrapped branch.

### FINAL metrics (clip 37, --no-foot-ik, sequential + temporal-continuity)
Per-hinge dof jitter mean|2nd diff| (rad) / max single-frame jump (deg):
  scapula_L_supinate 0.0035 / 4.5    scapula_R_supinate 0.0041 / 4.0
  scapula_L_extend   0.0039 / 4.1    scapula_R_extend   0.0033 / 3.4
  elbow_L 0.0194 / 20.7   elbow_R 0.0094 / 5.7
  wrist_L 0.0259 / 16.9   wrist_R 0.0179 / 14.6
  body-wide median 0.0009 rad / 0.73 deg; body-wide MAX max jump 20.7 deg
  (elbow_L). NO >30 deg/frame jumps anywhere (was 360 with the twist-hinge ver).
Decomposition/FK residual (FK of extracted dof vs target body rot):
  0.006 deg mean, 0.079 deg max  (<< 2 deg). Front-leg BONES (grs) smooth:
  scapula 4.5, upper_arm ~11, lower_arm/hand ~26 deg/frame MAX (natural swing).
compare_dog_retarget gait-phase corr: front_L +0.80 front_R +0.51
  hind_L +0.64 hind_R +0.77.
Loads: MuJoCo nbody 58 + 162 bone mesh geoms (162 assets); Newton body_count 57
  joint_count 57 dof 79; config 57 bodies / 73 dofs / root torso.

### Net of the two fixes
FIX 1 (scapula orthogonalized) removed the genuine 112 deg/frame scapula twist
flip. FIX 2 (1-DOF elbow/wrist + swing/twist projection) removed the 157/119
deg/frame elbow/wrist pops. Body geometry and all 162 bone attachments are
untouched, so the bones stay aligned. Spine topline rest-angle (#3), foot-IK,
refit, and the 102-clip batch remain DEFERRED.
