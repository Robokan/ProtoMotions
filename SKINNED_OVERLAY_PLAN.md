# Skinned Character Overlay — Design Plan

**Status: PLAN ONLY.** Goal: render fights with a rigged character's skinned
mesh instead of capsules — the character's skeleton is placed "on top of" the
SOMA articulation and its bones are driven, every frame, from the SOMA's
joint states. Physics stays 100% SOMA; only the visuals change.

## Why this is feasible

- The simulator already exposes per-frame world transforms for all 23 SOMA
  bodies in a stable, named order (`rigid_body_pos/rot` via
  `_get_simulator_bodies_state`, common-order remapped).
- Isaac/Omniverse renders UsdSkel skinned characters natively; what's missing
  is only the *coupling* — there is no built-in PhysX-articulation → UsdSkel
  binding, so we write a small per-frame sync (the whole feature is that glue).
- The hard math is already solved in-repo: the **SOMA23→BVH converter**
  (`45024de`, exact inverse of `convert_soma23_bvh_to_proto`) encodes the
  canonical joint mapping, `change_tpose` rest-pose offsets, and rot1/rot2
  frame conjugations, validated to 0.00000 m round-trip error. The overlay's
  joint-space conversion is the same transform chain applied at runtime
  instead of at export time.
- Recording renders at ~5 fps (launch-bound), so a Python per-frame sync is
  free in the only path that matters first.

## Architecture

```
sim step
  └─ rigid_body_pos/rot [2N, 23]  (world, common order)      ← already exists
       └─ OverlayDriver (new, per rendered arena)
            ├─ select fighter rows (ego_id, opp_id)           ← recorder knows these
            ├─ SOMA world xforms → character joint LOCALS
            │    joint map + bind-pose offsets (from BVH-converter math)
            └─ write UsdSkelAnimation prim (translations/rotations arrays)
                 └─ UsdSkel skins the character mesh; capsules hidden
```

Key design choices:

1. **Drive joint-local transforms via a `SkelAnimation` prim**, not per-prim
   xform writes — that's the UsdSkel-native path and keeps skinning on the
   renderer.
2. **Root + rotations only, bones don't stretch**: character is scaled to
   SOMA proportions once at import; per-frame we write root translation and
   per-joint rotations. Mismatched limb lengths then show up as (small) hand/
   foot offsets rather than mesh shear — matching proportions at prep time is
   the quality knob.
3. **Unmapped bones** (fingers, face, twist bones) hold bind pose — same
   policy as the BVH exporter's "template frame-0 locals for unmapped joints".
4. **Two characters per recording** (the followed arena's ego + opponent);
   other arenas keep capsules.

## Work plan

### Phase A — Asset prep (no code)
- Pick a rigged character (Reallusion CC or Mixamo — both FBX; the repo
  already has Reallusion FBX experience). Import to USD (SkelRoot + skinned
  mesh + skeleton) via Omniverse FBX importer.
- Scale/proportion pass: uniform scale to SOMA height, then limb-length
  check against `soma23` (the closer the proportions, the less clinch
  interpenetration and end-effector drift).
- Deliverable: `data/assets/overlay/<character>.usd` + a T-pose screenshot
  next to SOMA's rest pose.

### Phase B — Joint map + rest-pose calibration (the real work)
- `overlay_map_soma23.py`: 23-entry table SOMA body → character bone, plus
  per-joint bind-pose offset quaternions. **Derive from the BVH converter's
  mapping/offsets** rather than hand-authoring; SOMA is SMPL-family and the
  SOMASkeleton77 conventions are already inverted there.
- Static validation (no sim): load a packed SOMA motion clip, drive the
  character kinematically frame-by-frame, render side-by-side with the
  capsule playback (`motion_libs_visualizer` is the harness). Iterate until
  poses visually match; assert end-effector world-position error < ~5 cm.

### Phase C — Runtime driver
- `protomotions/simulator/isaaclab/overlay.py`: `SkinnedOverlay` class —
  `attach(character_usd, joint_map, fighter_env_ids)`, `sync(body_pos,
  body_rot)` writing the SkelAnimation arrays; hide the capsule visuals for
  overlaid fighters (visibility attr, not deletion).
- Hook: the recording path (`RecordingMixin` / `record_pairing`) calls
  `sync()` once per rendered frame — recorder already knows the followed
  arena's env ids. Config-gated (`--overlay <character>` on
  `battle_tournament.py` / `record_fight.sh` pass-through), default off.
- Fabric caveat: physics state is read through the tensor API (already how
  `rigid_body_pos/rot` arrive), so we never scrape USD for transforms; we
  only *write* the animation prim. If writes don't propagate under the
  recording render mode, fall back to `UsdSkelSkeleton` joint xform writes.

### Phase D — Polish (optional, later)
- Second character so the two fighters look different (corner colors).
- Real-time viewer support (move sync to a C++/Fabric extension only if
  Python-rate becomes the bottleneck in live viewing).
- Simple cloth/hair via UsdSkel blendshapes or engine-side — out of scope.

## Validation ladder
1. Bind-pose overlay screenshot: character standing inside/over SOMA rest pose.
2. Single-joint wiggle (scripted elbow/knee sweep) — checks joint-orient signs.
3. Kinematic clip playback side-by-side (Phase B gate).
4. One recorded bout with overlay on: `record_fight.sh 1 ... --overlay`.
5. Clinch stress test: grind-range bout, judge interpenetration acceptability.

## Risks / gotchas
- **Joint-orient conventions** are the classic sink (SOMA spherical joints vs
  character bone orients). Mitigated by reusing the BVH converter math and by
  validation steps 2–3 before any sim integration.
- **Bind-pose mismatch** (character A-pose vs SOMA T-pose): must bake the
  offset at calibration; symptom is a constant limb rotation error.
- **Clinch interpenetration**: visual mesh ≠ collision capsules; fights are
  close-range. Proportion matching reduces it; it cannot be eliminated.
- **Units/scale**: FBX cm vs USD m — check once at import.
- **Skel writes under headless offscreen rendering**: verify SkelAnimation
  updates are picked up per frame in the `enable_cameras` recording mode
  (fallback path noted in Phase C).

## Effort
- Phase A: ~half a day (asset wrangling).
- Phase B: 1–2 days (mapping + calibration harness; the BVH math is the head start).
- Phase C: 1–2 days (driver + recorder hook + config plumbing).
- Total: **< 1 week** to first overlaid fight video, independent of training.

## Relationship to alternatives
- The offline **BVH → Unreal retarget** path (already viable today via
  `45024de`) remains the max-quality route for cinematic renders.
- This overlay makes the *default* `record_fight.sh` output pretty with zero
  per-video manual steps — the two are complementary, sharing the same joint
  math.
- Atlas note: the same `OverlayDriver` works for any robot given a joint map,
  so an Atlas character skin is a Phase-B-only add once Atlas fights exist.
