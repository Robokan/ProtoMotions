# MJCF → USD robot conversion

The MJCF is the source of truth for every robot we author (Atlas, T800, tiger,
raptor, utahraptor). Isaac needs USD. This directory turns one into the other.

**The `usd/` trees are gitignored**, so a fresh checkout — or a new machine like
the DGX Spark — has the MJCF and none of the USD, and has to rebuild. That is
what this doc and `build_robot_usd.sh` exist to make survivable.

## Just build it

```bash
usd_convert/build_robot_usd.sh atlas
```

That runs all six steps in the required order, with the right paths, and
verifies the result. `--list` shows buildable robots; `--dry-run` prints the
commands without executing them.

Use the manual steps below only when debugging one stage.

## Environment

```bash
source ~/sparkpack/.venv-isaacsim5/bin/activate     # Isaac Sim 5 + IsaacLab 2.3.2
export OMNI_KIT_ACCEPT_EULA=YES
```

**Not `.venv-isaacsim6`.** IsaacLab 3.0 removed `PhysxCfg`, which the converter
imports. NumPy is pinned to 1.26.4. `.venv-isaacsim6` is for *running* Lab 3 /
Newton, not for building assets.

(Older docstrings in this directory still say `source env_isaaclab/bin/activate`.
That is the pre-venv container name and is stale.)

## The six steps

Run from the repo root. `<robot>` is the MJCF stem, e.g. `atlas`.

### 1. Flatten the MJCF

```bash
python usd_convert/flatten_mjcf.py protomotions/data/assets/mjcf/atlas.xml
```

Resolves `<default class=...>` inheritance, rewrites `<freejoint>` as
`<joint type="free">`, names unnamed mesh geoms, marks ranged joints
`limited="true"`. Writes `atlas_flat.xml` and re-loads both files in MuJoCo to
assert the compiled models are identical. **The converter only accepts flattened
input** — it does not understand MuJoCo's class inheritance and will silently
lose whatever the defaults carried.

### 2. Convert

```bash
python usd_convert/convert_robot_mjcf_to_usda.py \
    protomotions/data/assets/mjcf/atlas_flat.xml \
    --output-dir protomotions/data/assets/usd/atlas
```

Strips `<contact>/<sensor>/<tendon>` into a temp `_cleaned.xml`, inlines
material rgba onto geoms, shells out to `convert_mjcf_to_usd.py` (the IsaacLab
`MjcfConverter`), then re-adds visual meshes the converter dropped
(`patch_usd_visual_meshes.py` — the converter keeps only the *first* visual mesh
geom per body).

> **`--output-dir` is mandatory.** Its default is
> `usd/<mjcf-stem>/`, and the stem of a flattened file is `atlas_flat`, so the
> default writes `usd/atlas_flat/`. Every robot config reads
> `usd/atlas/atlas_flat.usda` — directory without `_flat`, file with it. Get this
> wrong and the converter still succeeds; nothing checks that the output landed
> where the config points, so **training quietly keeps loading the old body.**
>
> This has happened. `protomotions/data/assets/usd/config.yaml` is the fossil: a
> provenance file stranded at the root of `usd/` because someone passed
> `--output-dir .../usd` instead of `.../usd/raptor`. It and `usd/.asset_hash`
> are orphans — leave them or delete them, they belong to no robot.

### 3. Material / visual patches (robot-specific)

```bash
python usd_convert/patch_atlas_usd_bindings.py          # atlas
python usd_convert/patch_t800_usd_bindings.py           # t800
python usd_convert/hide_t800_collision_visuals.py       # t800
```

These hardcode their target paths. tiger/raptor/utahraptor need none.

### 4. Re-apply contact excludes — required

```bash
python usd_convert/apply_mjcf_contact_excludes.py \
    --mjcf protomotions/data/assets/mjcf/atlas.xml \
    --usd  protomotions/data/assets/usd/atlas/configuration/atlas_flat_physics.usd
```

**The IsaacLab MJCF converter silently drops MuJoCo `<contact><exclude>` pairs.**
Body pairs the MJCF declares as never-colliding will collide in Isaac. On T800
that is 36 excludes; on Atlas it is the difference between a working robot and
one that fights itself. Reads the excludes from the *original* MJCF and writes
them into the physics layer as filtered pairs.

### 5. Strip the duplicate articulation root — required

```bash
python usd_convert/strip_worldbody_articulation.py \
    --usd protomotions/data/assets/usd/atlas/configuration/atlas_flat_physics.usd
```

The converter puts `ArticulationRootAPI` on two prims. Isaac then refuses to
load:

```
RuntimeError: Failed to find a single articulation when resolving
'/World/envs/env_0/Robot'. Found multiple
'[.../Robot/worldBody, .../Robot/RigPelvis/RigPelvis]'
```

Both 4 and 5 operate on the **physics** layer, are idempotent, and must run
**after** the material patches on **every** regeneration.

### 6. Verify

```bash
python usd_convert/inspect_atlas_joints.py     # must print revolute=30 d6=0 with_drive=30
python data/scripts/check_self_collisions.py   # MuJoCo-side audit
```

`build_robot_usd.sh` also asserts that the file named by
`robot_configs/<robot>.py:usd_asset_file_name` actually exists — the direct
guard against the `--output-dir` trap.

## Optional: rescale before converting

```bash
python data/scripts/scale_robot_mjcf.py \
    --in-mjcf protomotions/data/assets/mjcf/raptor.xml \
    --out-mjcf protomotions/data/assets/mjcf/utahraptor.xml \
    --target-mass 200 --model-name utahraptor
```

Scaling is **not** one multiplier: length `s¹`, mass `s³`, torque `s⁴`,
armature `s⁵`, angles `s⁰`. This is how utahraptor was derived from raptor.
Run it between steps 1 and 2.

## Authoring rules for new MJCFs

- Three hinge joints (x, y, z) per articulated body. **Never** multi-joint D6
  merges — the IsaacLab importer breaks PD control on them.
- Declare self-collision exclusions as `<contact><exclude>` in the MJCF, not by
  hand in the USD. Step 4 carries them across; hand edits are lost on regen.

## Other conversion paths (don't confuse these)

- **`protomotions/simulator/isaaclab/utils/mjcf_to_usd.py`** — a *runtime*
  MJCF→USD conversion under Isaac Lab 3, cached by MJCF path + options. It
  produced `usd/atlas/lab3/`. Not this pipeline; see that directory's
  `SOURCE.txt`.
- **`collision_baking.py`** — convex-hull/decomposition collision variants
  (`*.collision_ch.usd` etc.), generated at runtime, verified by
  `scripts/verify_collision_baking.py`. Separate concern from MJCF→USD.
- **`convert_objects_to_usd.py`**, `scripts/convert_obj_scenes_to_usd.py` —
  scene meshes, not robots.
- **`data/scripts/build_creature_overlay_usd.py`** and friends — the skinned
  visual overlay, a separate track from the physics robot.

## Known state

| Robot | Converted | `self_collision` in USD |
|---|---|---|
| atlas | 2026-08-01 | **false** |
| t800 | 2026-08-03 | true |
| tiger | 2026-08-07 | true |
| raptor | 2026-08-05 | true |
| utahraptor | 2026-08-11 | true |

Atlas's on-disk USD predates commit `f0be48f` (2026-08-02), which set
`self_collision=True` in the converter because the IsaacLab default was
overriding the articulation flag. `robot_configs/atlas.py` doesn't set
`self_collisions`, so it inherits `base.py`'s default of `True` while the asset
says `false`. Worth resolving by rebuilding Atlas — which is now one command.

anymal_d and go2 have no MJCF here; they were imported as USD and are not built
by this pipeline.
