<div align="center">

# ProtoMotions 3

**A GPU-Accelerated Framework for Simulated Humanoids**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
[![Documentation](https://img.shields.io/badge/docs-online-green.svg)](https://protomotions.github.io/)

[![Newton](https://img.shields.io/badge/Newton-e7a737c-brightgreen.svg)](https://github.com/newton-physics/newton/commit/e7a737c)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.0-blue.svg)](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.0)
[![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview_4-blue.svg)](https://developer.nvidia.com/isaac-gym)
[![Genesis](https://img.shields.io/badge/Genesis-untested-lightgrey.svg)](https://github.com/Genesis-Embodied-AI/Genesis)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.0+-orange.svg)](https://github.com/google-deepmind/mujoco)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVlabs/ProtoMotions) (unverified AI generation)

</div>

---

## Overview

**ProtoMotions3** is a GPU-accelerated simulation and learning framework for training physically simulated digital humans and humanoid robots. Our mission is to provide a **fast prototyping platform** for various simulated humanoid learning tasks and environments—for researchers and practitioners in **animation**, **robotics**, and **reinforcement learning**—bridging efforts across communities.

**Modularity**, **extensibility**, and **scalability** are at the core of ProtoMotions3. It is **community-driven** and permissively licensed under the [Apache-2.0 license](LICENSE.md).

Also check out **[MimicKit](https://github.com/xbpeng/MimicKit/tree/main)**, our sibling repository for a lightweight framework for motion imitation learning.

<table>
<tr>
<td align="center"><img src="data/static/vault.gif" height="180"/></td>
<td align="center"><img src="data/static/g1_tracker.gif" height="180"/></td>
<td align="center"><img src="data/static/soma_regen.gif" height="180"/></td>
</tr>
<tr>
<td align="center"><img src="data/static/wineglass.gif" height="180"/></td>
<td align="center"><img src="data/static/real_robot.gif" height="180"/></td>
<td align="center"><img src="data/static/real_robot_3.gif" height="180"/></td>
</tr>
</table>

---

## What You Can Do with ProtoMotions3

### 🏃 Large-Scale Motion Learning

Train your fully physically simulated character to learn motion skills from the entire public [**AMASS**](https://amass.is.tue.mpg.de/) human animation dataset (**40+ hours**) within **12 hours** on 4 A100s.

<p align="center">
  <img src="data/static/smpl_mlp_094132.gif" alt="SMPL motion 1" height="180">
  <img src="data/static/smpl_mlp_094428.gif" alt="SMPL motion 2" height="180">
  <img src="data/static/smpl_mlp_095344.gif" alt="SMPL motion 3" height="180">
  <img src="data/static/smpl_mlp_095848.gif" alt="SMPL motion 4" height="180">
  <img src="data/static/smpl_mlp_095746.gif" alt="SMPL motion 5" height="180">
</p>

### 📈 Scalable Multi-GPU Training

Scale training to even larger datasets with each GPU handling a subset of motions. For example, we have trained with **24 A100s** with **13K motions** on each GPU with the [**BONES**](https://huggingface.co/datasets/bones-studio/seed) dataset in [**SOMA**](https://github.com/NVlabs/SOMA-X) skeleton format. Check out [Quick Start](https://protomotions.github.io/getting_started/quickstart.html) and [SEED BVH Data Preparation](https://protomotions.github.io/getting_started/seed_bvh_preparation.html) to play around with the dataset and pre-trained models today.

<p align="center">
  <img src="data/static/soma_regen_markers.gif" height="180">
  <img src="data/static/soma_regen_2.gif" height="180">
  <img src="data/static/soma_regen_3.gif" height="180">
  <img src="data/static/soma_regen_4.gif" height="180">
  <img src="data/static/soma_regen_5.gif" height="180">
</p>

### 🔄 One-Command Retargeting

Transfer (retarget) the entire [AMASS](https://amass.is.tue.mpg.de/) dataset to your favorite robot with the built-in [**PyRoki**](https://github.com/chungmin99/pyroki)-based optimizer—in one command.

> **Note:** As of v3, we use [PyRoki](https://github.com/chungmin99/pyroki) for retargeting. Earlier versions used [Mink](https://github.com/kevinzakka/mink).

<p align="center">
  <img src="data/static/retargeting-g1.gif" alt="G1 retargeting" height="280">
</p>

### 🤖 Train Any Robot

Train your robot to perform AMASS motor skills in **12 hours**, by just changing one command argument:  
`--robot-name=smpl` → `--robot-name=h1_2` and preparing retargeted motions (see [here](https://protomotions.github.io/tutorials/workflows/retargeting_pyroki.html))

<p align="center">
  <img src="data/static/h1_2_gym.gif" alt="H1_2 AMASS training" height="280">
</p>

### 🐕 Quadruped Motion Imitation (Go2, ANYmal-D, dog)

Train quadrupeds to imitate retargeted mocap across all engines (IsaacGym, IsaacLab, Newton). Three robots are registered out of the box:

| Robot | `--robot-name` | Asset | Source motions |
|-------|----------------|-------|----------------|
| Unitree **Go2** | `go2` | `mjcf/go2.xml` (+ USD) | poselib NPY (MANN/ASE) |
| **ANYmal-D** | `anymal_d` | `mjcf/anymal_d.xml` | poselib NPY |
| dm_control **dog** | `dog_v2` | `mjcf/dog_v2_nomesh.xml` (BVH-matched cylinder skeleton) | MANN BVH library |

**Quick launch.** Once the motion libraries exist under `data/motions/` (steps 1–4 below), each robot has a one-command launcher for training (headless) and one for running the trained policy in a windowed viewer:

| Robot | Train | Run trained policy |
|-------|-------|--------------------|
| Go2 (IsaacLab) | `scripts/train_go2_tracker.sh` | `scripts/run_go2_tracker.sh` |
| ANYmal-D (IsaacLab) | `scripts/train_anymal_tracker.sh` | `scripts/run_anymal_tracker.sh` |
| ANYmal-D MaskedMimic (IsaacLab) | `scripts/train_anymal_masked_mimic.sh` | `scripts/run_anymal_masked_mimic.sh` |
| dog (Newton) | `scripts/train_dog_tracker.sh` | `scripts/run_dog_tracker.sh` |

All launchers accept env-var overrides and pass extra args through to the underlying script:

```bash
GPU=1 scripts/train_anymal_tracker.sh                # pin a GPU; auto-resumes results/$EXP/last.ckpt
GPU=1 NO_RESUME=1 EXP=anymal_v2 scripts/train_anymal_tracker.sh   # force a fresh run under a new name
NUM_ENVS=4096 BATCH_SIZE=16384 scripts/train_go2_tracker.sh       # shrink for a smaller GPU
CKPT=results/go2_tracker/last.ckpt scripts/run_go2_tracker.sh     # view a specific checkpoint
scripts/run_anymal_tracker.sh --full-eval                          # extra args pass through
```

Notes:
- The IsaacLab launchers activate `../.venv-isaacsim5`; the dog (Newton) launchers activate `../.venv-protomotions-newton`. Adjust if your envs live elsewhere.
- `train_anymal_masked_mimic.sh` **distills a trained tracker** — train `scripts/train_anymal_tracker.sh` first, then point `EXPERT=results/anymal_flat_v1/last.ckpt` at it. It is heavier per env (transformer student + frozen expert): 4096 envs fits a 24 GB card.
- The dog is a sim-only skeletal model (no deployable variant); for a sim2real-deployable Go2/ANYmal tracker see the BeyondMimic config below.

**1. Prepare motions.** Convert retargeted poselib clips to a packed MotionLib `.pt`, or retarget the dm_control dog directly from BVH:

```bash
# Go2 / ANYmal-D — poselib NPY → ProtoMotions .pt
python data/scripts/convert_quadruped_poselib_to_proto.py \
    --yaml-file /path/to/full_set.yaml --motion-dir /path/to/clips/ \
    --robot-name go2 --output data/motions/go2/go2_full.pt

# dm_control dog — identity retarget from the MANN BVH library (Y-up→Z-up)
python data/scripts/retarget_bvh_to_dog.py --clips /path/to/bvh/ --mirror
```

**2. (Optional) Motion-support terrain.** Some clips climb onto blocks/platforms. The scanner flags those clips, builds per-clip support structures, and spawns the flagged clips on them so the reference motion lines up. A clip needs support only when **all four feet leave the floor onto a sustained, elevated, non-falling surface** — pure jumps (free-fall), rearing/sitting (a foot stays down), and lie-downs (surface too low) are correctly left flat.

```bash
python data/scripts/scan_clip_support_geometry.py \
    --clips-dir data/motions/anymal_d/clips \
    --motion-lib data/motions/anymal_d/anymal_d_full.pt \
    --output data/motions/anymal_d/support_manifest.yaml --standing-height 0.6
```

**3. (Optional) Split mixed clips.** A clip that walks on flat ground *and* climbs a platform is split at the airborne-event boundaries — the climb (+1s of ground each side) becomes a support sub-clip on terrain, the rest trains on flat ground, and no jump is ever cut in half:

```bash
python data/scripts/split_support_clips.py \
    --motion-lib data/motions/anymal_d/anymal_d_full.pt \
    --manifest   data/motions/anymal_d/support_manifest.yaml \
    --out-lib    data/motions/anymal_d/anymal_d_split.pt \
    --out-manifest data/motions/anymal_d/support_manifest_split.yaml
```

**4. (Optional) Weight by uniqueness.** Re-weight `motion_weights` so rare behaviours (jumps, climbs, backward/sideways gaits) are sampled more and redundant forward walks less — training (and the viewer) draw clips via `multinomial(motion_weights)`:

```bash
python data/scripts/compute_uniqueness_weights.py \
    --motion-lib data/motions/anymal_d/anymal_d_split.pt --feet 4 8 12 16
```

**5. Train** with the quadruped experiment, on any engine:

```bash
python protomotions/train_agent.py \
    --robot-name anymal_d --simulator isaaclab \
    --experiment-path examples/experiments/mimic/quadruped_mlp.py \
    --experiment-name anymal_split_terrain \
    --motion-file data/motions/anymal_d/anymal_d_split.pt \
    --num-envs 12288 --batch-size 49152 \
    --overrides terrain.motion_support_manifest=data/motions/anymal_d/support_manifest_split.yaml \
                terrain.motion_support_motion_lib=data/motions/anymal_d/anymal_d_split.pt
```

**Deployable (sim2real) training for Go2 / ANYmal-D.** Use the BeyondMimic "bones deploy" config — an asymmetric actor-critic where the **actor sees only on-board signals** (reduced-coords joint proprioception, projected gravity, local angular velocity, the reference as a root-relative future trajectory; **no root height, no root linear velocity, no global position**), while the privileged critic gets full state. Adds L2C2 noise regularization and domain randomization, and exports to ONNX with observation computation baked in. (The sim-only dog is not deployable and uses `quadruped_mlp.py`.)

```bash
python protomotions/train_agent.py \
    --robot-name go2 --simulator isaaclab \
    --experiment-path examples/experiments/mimic/quadruped_bm_deploy.py \
    --experiment-name go2_bm_deploy \
    --motion-file data/motions/go2/go2_full.pt \
    --num-envs 8192 --batch-size 32768

# export the trained tracker to ONNX (obs baked in) for hardware deployment
python deployment/export_bm_tracker_onnx.py --checkpoint results/go2_bm_deploy/last.ckpt
```

**Preview the reference motions** before training (`N`/`=` next clip, `P`/`-` previous; weighted random when one finishes):

```bash
python examples/env_kinematic_playback.py \
    --robot-name dog_v2 --simulator newton --num-envs 1 \
    --motion-file data/motions/dog_v2/dog_full.pt \
    --experiment-path examples/experiments/mimic/quadruped_mlp.py
```

### 🔬 Sim2Sim Testing

One-click test (`--simulator=isaacgym` → `--simulator=newton` → `--simulator=mujoco`) of robot control policies on **H1_2** or **G1** in different physics engines (NVIDIA Newton, MuJoCo CPU). Policies shown below only use observations you could actually get from real hardware.

<p align="center">
  <img src="data/static/h12-g1-newton-sim2sim.gif" alt="H1_2/G1 sim2sim" height="280">
</p>

### 🤖 From Sim to Real

Train in simulation, deploy on real hardware. ProtoMotions trains one General Tracking Policy on entire [**BONES-SEED**](https://huggingface.co/datasets/bones-studio/seed) dataset (~142K motions) and transfers directly to the Unitree G1 humanoid robot zero-shot.

<p align="center">
  <img src="data/static/g1_deploy_1.gif" alt="G1 deployment 1" height="240">
  <img src="data/static/g1_deploy_2.gif" alt="G1 deployment 2" height="240">
  <img src="data/static/real_robot_2.gif" alt="G1 real robot" height="240">
</p>

Our deployment pipeline exports a single ONNX model (with observation computation baked in), so deployment frameworks only need to provide raw sensor signals — no need to rewrite obs functions or match training internals. We tested on the Unitree G1 via the brilliant [**RoboJuDo**](https://github.com/HansZ8/RoboJuDo) framework, adding just one policy file with no mandatory changes to RoboJuDo core.

📖 [**Full Deployment Tutorial**](https://protomotions.github.io/tutorials/workflows/g1_deployment.html) — from data preparation to real robot, fully reproducible.

### 🎨 High-Fidelity Rendering

Test your policy in [**IsaacSim 5.0+**](https://developer.nvidia.com/isaac-sim), which allows you to load beautifully rendered Gaussian splatting backgrounds (with [**Omniverse NuRec**](https://developer.nvidia.com/blog/reconstruct-a-scene-in-nvidia-isaac-sim-using-only-a-smartphone/) — this rendered scene is not physically interact-able yet).

<p align="center">
  <img src="data/static/g1-neurc.gif" alt="G1 NeuRec" height="280">
</p>

### 🎬 Motion Authoring with Kimodo

With [**Kimodo**](https://research.nvidia.com/labs/sil/projects/kimodo/) (NVIDIA's text-to-motion generation model), generate any motion from a text prompt and use ProtoMotions to train a physics-based policy that performs the motion — for both the SOMA animation character and the Unitree G1 robot. Policies trained this way can be deployed directly on real hardware.

See [Kimodo Data Preparation](https://protomotions.github.io/getting_started/kimodo_preparation.html) for how to convert Kimodo outputs to ProtoMotions format.

<p align="center">
  <img src="data/static/aibm-vaulting.gif" alt="Vaulting" height="240">
  <img src="data/static/g1_robot_walking.gif" alt="G1 robot walking" height="240">
</p>

> *Image Credit: [NVIDIA Human Motion Modeling Research](https://research.nvidia.com/labs/sil/human_motion_modeling/)*



### 🏗️ Procedural Scene Generation

Procedurally generate many scenes for scalable **Synthetic Data Generation (SDG)**: start from a seed motion set, use RL to adapt motions to augmented scenes.

<p align="center">
  <img src="data/static/augmented_combined.gif" alt="Augmented Scenes and Motions" height="280">
</p>

### 🎭 Generative Policies

Train a generative policy (e.g., [**MaskedMimic**](https://research.nvidia.com/labs/par/maskedmimic/)) that can autonomously choose its "move" to finish the task. For reusable discrete latent priors and PEFT task adapters, see the [**GPC and PEFT guide**](https://protomotions.github.io/user_guide/gpc.html).

<table align="center">
<tr>
<td align="center"><img src="data/static/maskedmimic_093152.gif" alt="MaskedMimic 1" height="180"/></td>
<td align="center"><img src="data/static/maskedmimic_093229.gif" alt="MaskedMimic 2" height="180"/></td>
<td align="center"><img src="data/static/maskedmimic_093313.gif" alt="MaskedMimic 3" height="180"/></td>
</tr>
<tr>
<td align="center"><img src="data/static/maskedmimic_093430.gif" alt="MaskedMimic 4" height="180"/></td>
<td align="center"><img src="data/static/maskedmimic_093406.gif" alt="MaskedMimic 5" height="180"/></td>
<td align="center"><img src="data/static/maskedmimic_093349.gif" alt="MaskedMimic 6" height="180"/></td>
</tr>
</table>

### 🥊 Two-Character Combat — SOMA Fight Club

Train two characters to **fight each other** with natural, human-like combat
motion, using tournament-style self-play on top of a GPC motor foundation.
This is a fork addition (branch `battle`); the design lives in
[`SOMA_GPC_COMBAT_PLAN.md`](SOMA_GPC_COMBAT_PLAN.md).

**How it works — three stages, each building on the last:**

1. **GPC prior** (motor foundation): a generative prior over the frozen FSQ
   tracker's motion tokens, trained on a combat-weighted
   [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) library.
   The fight policy acts in *token space*, so every action decodes to a
   physically-executable, human-looking movement — no AMP discriminator, no
   GAN instability.
2. **Combat SFT**: a DoRA/PEFT adapter that biases the prior toward fighting,
   conditioned on an *opponent observation*, so the policy engages an
   opponent instead of just idling.
3. **Battle league**: PPO self-play across many paired arenas (env `i` fights
   env `i+N`), with a PFSP opponent pool, snapshots, and Elo. Naturalness is
   enforced structurally by prior-constrained sampling, so the reward only has
   to define *winning*.

**The battle environment** (`protomotions/envs/battle/`): two characters per
match in a shared arena, contact-based hit scoring (per-region damage
multipliers, closing-velocity gating), knockdown/get-up with a grace window,
and a deliberately thin reward — sparse win/loss + an IsaacLabASE-style
approach term (velocity toward the opponent, attenuating inside fighting
range) + gaze facing + a kickboxing diversity bonus (rewards the under-used
limb group so fighters punch *and* kick). Ring-outs resolve as a points
decision, not an instant loss.

```bash
# 1. Train the GPC prior on the combat-weighted SEED library
#    (see scripts/run_seed_prior_pipeline.sh for the full data pipeline)
python protomotions/train_agent.py \
    --robot-name soma23 --simulator isaaclab --headless \
    --motion-file data/soma_seed_curated.pt \
    --experiment-path examples/experiments/gpc/prior.py \
    --tracker-checkpoint data/pretrained_models/motion_tracker/soma_bones_fsq/last.ckpt \
    --experiment-name soma_gpc_prior

# 2. Combat SFT: bias the prior toward fighting (warm-starts the league)
python protomotions/train_agent.py \
    --robot-name soma23 --simulator isaaclab --headless \
    --motion-file data/soma_combat_viewer.pt \
    --experiment-path examples/experiments/gpc/sft_combat_prior_peft.py \
    --prior-checkpoint results/soma_gpc_prior/last.ckpt \
    --experiment-name soma_sft_combat

# 3. Battle league (tournament self-play)
python protomotions/train_agent.py \
    --robot-name soma23 --simulator isaaclab --headless \
    --motion-file data/soma_combat_viewer.pt \
    --experiment-path examples/experiments/battle/battle_league_prior_peft.py \
    --prior-checkpoint results/soma_gpc_prior/last.ckpt \
    --checkpoint results/soma_sft_combat/last.ckpt \
    --num-envs 256 --batch-size 512 \
    --experiment-name soma_battle_league
```

**Watch and evaluate** with `protomotions/battle_tournament.py`:

```bash
# Exhibition: two checkpoints fight in a windowed viewer (arena ring drawn).
# Press O to follow the camera, R to restart.
python protomotions/battle_tournament.py \
    --resolved-configs results/soma_battle_league/resolved_configs_inference.pt \
    --exhibition results/soma_battle_league/inference_last.ckpt \
                 results/soma_battle_league/lightning_logs/version_0/league/policy_0.ckpt \
    --num-envs 2 --matches-per-pairing 4

# Record fights to mp4 — real IsaacSim render, offscreen (NO display needed).
# Records whole bouts (KO / ring-out / timeout); --bouts strings several into
# one clip. Simplest: the wrapper (latest vs a random pool member by default):
scripts/record_fight.sh 3
# Or the underlying CLI, to pick the two fighters explicitly:
python protomotions/battle_tournament.py \
    --resolved-configs results/soma_battle_league/resolved_configs_inference.pt \
    --exhibition <ckpt_a> <ckpt_b> \
    --record output/fight_videos/a_vs_b.mp4 --bouts 3 --num-envs 2 --deterministic

# Round-robin tournament over league snapshots -> Elo ladder + head-to-head JSON
python protomotions/battle_tournament.py \
    --resolved-configs results/soma_battle_league/resolved_configs_inference.pt \
    --adapters "results/soma_battle_league/lightning_logs/version_0/league/policy_0.ckpt,..." \
    --matches-per-pairing 16 --num-envs 32 --headless \
    --output results/soma_battle_league/tournament.json

# Diagnostic: print opponent-observation and engagement stats (no matches)
python protomotions/battle_tournament.py \
    --resolved-configs results/soma_battle_league/resolved_configs_inference.pt \
    --exhibition <ckpt_a> <ckpt_b> --headless --probe-steps 120
```

**Auto-gallery over time (optional).** `scripts/battle_video_daemon.sh` polls a
league run for new `policy_*.ckpt` snapshots and records each new one against a
fixed baseline (the earliest snapshot by default), building a chronological
gallery in `output/fight_videos/`. It runs headless and self-throttles on host
memory (recording is a second Isaac process; on a unified-memory box it skips a
tick if free RAM is below `MIN_GB`, default 25):

```bash
# run_name  interval_sec  container
scripts/battle_video_daemon.sh soma_battle_league_v3 3600 battle
```

> **NumPy pin:** offscreen recording needs NumPy 1.x — the canonical Spark
> image ships NumPy 2, which breaks Isaac's render extensions (the annotator
> recurses in `np.all`). `Dockerfile.spark` pins `numpy==1.26.4`; rebuild the
> image (`docker build -f Dockerfile.spark -t protomotions:spark .`) so
> `--record` works in the `battle` container.

> **Note:** the combat data is derived from the gated BONES-SEED dataset and
> is not shipped. Build the motion libraries with
> `scripts/run_seed_prior_pipeline.sh` after accepting the dataset license.
> On a unified-memory box (e.g. DGX Spark), run `scripts/memory_watchdog.sh`
> alongside training and don't open a viewer while a league trains.

### ⛰️ Terrain Navigation

Train your robot to hike challenging terrains!

<p align="center">
  <img src="data/static/smpl_terrain.gif" alt="SMPL Terrain" height="280">
</p>

### 🎯 Custom Environments

Have a new task? Build it from modular components — no monolithic env class needed. Here's how the **steering** task is composed:

| Layer | File | What it does |
|-------|------|-------------|
| **Control** | [`steering_control.py`](protomotions/envs/control/steering_control.py) | Manages task state (target direction, speed, facing). Periodically samples new heading targets. |
| **Observation** | [`obs/steering.py`](protomotions/envs/obs/steering.py) | Pure tensor kernel — transforms targets to robot-local frame → 5D feature vector. |
| **Reward** | [`rewards/task.py`](protomotions/envs/rewards/task.py) | `compute_heading_velocity_rew` — blends direction-matching (0.7) and facing-matching (0.3) rewards. |
| **Experiment** | [`steering/mlp.py`](examples/experiments/steering/mlp.py) | Wires components together as `MdpComponent` instances via context paths. |

Each piece is a standalone function or class — the experiment config binds them into a complete task using [`MdpComponent`](protomotions/envs/mdp_component.py) and [`FieldPath`](protomotions/envs/context_views.py) descriptors.

<p align="center">
  <img src="data/static/g1_steering.gif" alt="G1 Steering" height="280">
</p>


### 🧪 New RL Algorithms

Want to try a new RL algorithm? Implement algorithms like **ADD** in ProtoMotions in ~50 lines of code, utilizing our modularized design:

📄 [`protomotions/agents/mimic/agent_add.py`](protomotions/agents/mimic/agent_add.py)

### 🔧 Custom Simulators

Would like to use your own simulator? Implement these APIs interfacing among different simulators:

📄 [`protomotions/simulator/base_simulator/`](protomotions/simulator/base_simulator/)

Refer to this community-contributed example:

📄 [`protomotions/simulator/genesis/`](protomotions/simulator/genesis/)

### 🤖 Add Your Own Robot

Want to add your own robot? Follow these steps:

1. Add your `.xml` MuJoCo spec file to [`protomotions/data/assets/mjcf/`](protomotions/data/assets/mjcf/)
2. Fill in config fields (see examples like [`protomotions/robot_configs/g1.py`](protomotions/robot_configs/g1.py))
3. Register in [`protomotions/robot_configs/factory.py`](protomotions/robot_configs/factory.py)

And you're good to go!

---

## Documentation

📚 **[Full Documentation](https://protomotions.github.io/)**

- [Installation Guide](https://protomotions.github.io/getting_started/installation.html)
- [Quick Start](https://protomotions.github.io/getting_started/quickstart.html)
- [GPC and PEFT](https://protomotions.github.io/user_guide/gpc.html)
- [AMASS Data Preparation](https://protomotions.github.io/getting_started/amass_preparation.html)
- [PHUMA Data Preparation](https://protomotions.github.io/getting_started/phuma_preparation.html)
- [SEED BVH Data Preparation](https://protomotions.github.io/getting_started/seed_bvh_preparation.html)
- [SEED G1 CSV Data Preparation](https://protomotions.github.io/getting_started/seed_g1_csv_preparation.html)
- [Kimodo Data Preparation](https://protomotions.github.io/getting_started/kimodo_preparation.html)
- [Tutorials](https://protomotions.github.io/tutorials/)
- [API Reference](https://protomotions.github.io/api_reference/)
- [G1 Deployment: Data to Real Robot](https://protomotions.github.io/tutorials/workflows/g1_deployment.html)

---

## Contributing

We welcome contributions! Please read our [**Contributing Guide**](CONTRIBUTING.md) before submitting pull requests.

## License

ProtoMotions3 is released under the [**Apache-2.0 License**](LICENSE.md).

Third-party software and bundled asset notices are listed in [legal/](legal/), including Unitree, BeyondMimic, Isaac Lab, and SMPL/SMPL-H attribution and license notices.

---

## Citation

If you use ProtoMotions3 in your research, please cite:

```bibtex
@misc{ProtoMotions,
  title = {ProtoMotions3: An Open-source Framework for Humanoid Simulation and Control},
  author = {Tessler*, Chen and Jiang*, Yifeng and Peng, Xue Bin and Coumans, Erwin and Shi, Yi and Zhang, Haotian and Rempe, Davis and Chechik†, Gal and Fidler†, Sanja},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NVLabs/ProtoMotions/}},
}
```
