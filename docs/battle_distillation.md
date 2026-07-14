# Distilling a battle champion into a fast MLP (viewer + Thor)

The league policy acts in **token space**: each control step runs 8 sequential
autoregressive prior forwards. At the batch size of a single match the GPU is
badly underutilized (~50–100 ms/step; see the perf notes in the README battle
section), so the viewer tops out around 5 fps and on-robot deployment would be
latency-bound.

A **distilled feedforward student** removes the autoregression entirely:
observations → joint PD targets in one MLP forward (sub-millisecond). That is
fast enough for a smooth viewer, and it is the realistic **Jetson Thor**
deployment path (ONNX → TensorRT/fp8, running at control rate with headroom).

This uses the framework's built-in supervised-distillation agent
(`SupervisedAgent`) — the same machinery the SFT step uses, with an MSE action
loss instead of token cross-entropy.

## Pipeline

1. **Train a champion** (the teacher) — the league run, e.g.
   `results/soma_battle_league_v3`. Distillation quality is capped by the
   teacher, so distill a *mature* champion, not an early snapshot.

2. **Distill** with `examples/experiments/battle/distill_battle_mlp.py`
   (DAgger: the MLP student drives rollouts in the battle env, the champion
   labels each visited state with its action, the student regresses to match):

   ```bash
   python protomotions/train_agent.py \
       --robot-name soma23 --simulator isaaclab --headless \
       --motion-file data/soma_combat_viewer.pt \
       --experiment-path examples/experiments/battle/distill_battle_mlp.py \
       --expert results/soma_battle_league_v3 \
       --num-envs 512 --experiment-name soma_battle_distill
   ```

   > **Status:** the experiment is a reviewed scaffold; it needs one GPU
   > validation pass. The block to confirm live is the student model config
   > (its action output key must match `loss.prediction_key`) — see the
   > `TODO(validation)` note in the experiment file.

3. **Watch the student** — a plain MLP, so the viewer runs at hundreds of fps
   (no `--fast-sampling` needed; there is no autoregression). Exhibition
   support for a feedforward student checkpoint is the remaining viewer-side
   wiring (the tournament currently assumes the PEFT model); the student can
   also be driven directly through `inference_agent.py`.

4. **Export for Thor** — the deploy trackers' ONNX exporter
   (`deployment/export_bm_tracker_onnx.py`, from the quadruped work) is the
   template: an MLP actor with observation computation baked in exports
   cleanly to ONNX, then TensorRT with fp8 on Thor's Blackwell tensor cores.

## Why the student is enough

The champion's fighting behavior is, at deployment, a *fixed* policy — the
generative token structure earns its keep during *training* (natural motion,
league diversity), not at inference. A student that reproduces the champion's
state→action mapping captures the behavior at a fraction of the cost. This is
the standard route for putting expensive generative policies on hardware.

Caveats: validate that the student preserves fight quality (spot-check
exhibitions vs. the teacher), and that sim2real transfer holds separately —
distillation does not fix a teacher trained without deployment robustness
(domain randomization, action-rate penalties; the `g1-bones-deploy` tracker is
the template for a deployable motor foundation on a real robot).
