# Setting up ProtoMotions on the DGX Spark

Read this first. It is the whole bring-up.

Two sources: **git** carries the code and most assets, the **WD Elements drive**
carries what git cannot (checkpoints, corpora, retargeted clips, Atlas's USD).
You need both.

## 1. Code

```bash
git clone https://github.com/Robokan/ProtoMotions.git
cd ProtoMotions
git checkout battle          # NOT main -- battle is the live branch
```

## 2. Data from the Elements drive

Mount the drive, then copy from `ProtoMotionsData/`:

```bash
DRIVE=/path/to/Elements/ProtoMotionsData
rsync -rlt --inplace --no-perms --no-owner --no-group --modify-window=1 \
      "$DRIVE/data/"                       data/
rsync -rlt --inplace --no-perms --no-owner --no-group --modify-window=1 \
      "$DRIVE/protomotions/data/assets/"   protomotions/data/assets/
rsync -rlt --inplace --no-perms --no-owner --no-group --modify-window=1 \
      "$DRIVE/results/"                    results/
```

`--inplace` is not optional: the drive is NTFS and its `data/` directory has
damaged entries that fail every rename with `Input/output error`. Use it and
those failures do not occur.

**Known gap:** tiger and utahraptor corpora are NOT on the drive — their
entries on the NTFS volume are corrupt (they appear in `ls` but `stat` and
`read` fail with EIO). Atlas, ANYmal, Go2, T800 and raptor data are all
complete. If you need tiger/utahraptor, the drive cannot supply them; they must
be re-copied from the 4090 after the filesystem is repaired.

What comes from where:

| | git | drive |
|---|---|---|
| code | yes | — |
| MJCF, meshes, URDF | yes | yes |
| USD for t800/tiger/raptor/utahraptor/g1/h1 | yes | yes |
| **USD for atlas** | **no** (gitignored) | yes |
| corpora (`data/*.pt`) | 5 tiny samples only | yes |
| retargeted clips (`data/motions/`) | no | yes |
| checkpoints | no | yes (`last.ckpt` + `epoch_1000.ckpt` per run) |

Only `last.ckpt` and `epoch_1000.ckpt` are backed up per run — the
`epoch_N.ckpt` series stays on the 4090. `last.ckpt` is the current policy;
`epoch_1000.ckpt` exists because it is the discriminator/critic donor for the
warm-start splice.

## 3. Atlas USD

If step 2 copied `protomotions/data/assets/usd/atlas/`, you are done — it is
the built asset, post-processing already applied.

If you must rebuild it from MJCF:

```bash
usd_convert/build_robot_usd.sh atlas
```

Do **not** hand-run the individual converter. The chain is six steps and three
of them are easy to miss and fail like physics bugs rather than build errors
(dropped contact excludes, duplicate articulation root, wrong `--output-dir`
silently leaving the old body in place). See `usd_convert/README.md`.

## 4. Environments

Two venvs, not interchangeable:

- **`.venv-isaacsim5`** — Isaac Sim 5 / IsaacLab 2.3.2, Python 3.11, NumPy
  pinned 1.26.4. Used for **asset conversion** only. IsaacLab 3.0 removed
  `PhysxCfg`, which the converter imports.
- **`.venv-isaacsim6`** — Isaac Sim 6 / IsaacLab 3. Used for **training and
  inference**, selected with `--physics physx|newton`.

`export OMNI_KIT_ACCEPT_EULA=YES` for every Isaac launch.

## 5. Train

Lab 3 + PhysX is the working stack.

```bash
PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 \
python protomotions/train_agent.py \
  --robot-name atlas --simulator isaaclab --physics physx --headless \
  --motion-file data/atlas_pretrain_corpus_v17.pt \
  --experiment-path examples/experiments/ase/mlp_template_tuned.py \
  --experiment-name atlas_spark_v1 \
  --num-envs 8192 --batch-size 16384
```

To continue an existing run instead, restore its directory under `results/` and
add `--resume`. **Resume loads the pickled `resolved_configs.pt`** — it ignores
CLI flags and never reads the experiment file, so config changes must go
through `--overrides` or by patching the pickle.

## 6. Things that will bite you

- **Newton is eval-only.** MJWarp pre-allocates a contact buffer and *silently
  drops* contacts past it, which causes interpenetration and then divergence.
  Atlas at 2048 envs asks for a single 8 GB contact array; ANYmal at 8192 hits
  Warp's signed-32-bit shape limit. Train on PhysX.
- **Viewer env counts**: 5 for an isaaclab viewer, 50 for newton. Higher will
  not open.
- **Warm starts must pass `--freeze-actor-obs-norm`.** The observation
  normalizer is an EMA; without the freeze it rewrites itself over the first
  epochs and destroys the policy's input distribution while the weights sit
  there looking fine.
- `mlp_template_tuned.py` is mandatory for Atlas — the plain template plateaus
  around 0.19 style reward.
