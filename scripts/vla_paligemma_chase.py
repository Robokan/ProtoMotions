# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune PaliGemma on a chase recording, then grade what it says and does.

A VLA test for the go2 chase: shown the dog's 224x224 eye crop and the
request, can a ~3B vision-language model ANSWER ("yes, I see a green ball"
-- and "no" when only a red one is in view) and ACT (the torso target,
heading and next gaze the demonstrator chose)? -- or, told "sit" or "beg",
produce the MaskedMimic targets that put the dog in that pose?

    prefix : "is there a green ball? | asked 2.1s ago | said: let me look | turned 140 | turning +60 | lean left | gaze +0.10 -0.05 2.0"
    text   : "yes, I see a green ball"                (generated, short)
    action : the MaskedMimic conditioning -- per body position, orientation,
             on/off, the lead time -- and the next gaze (regression head, one pass)

Three choices the first version taught:

* **Memory in the prompt.** An answer is sticky (once "yes", still "yes"
  after the ball slips out of view) and "no" comes only after a full turn,
  so a single frame cannot say either. The prefix carries what the dog last
  said and how far it has turned since the throw (the gyro's yaw rate,
  integrated -- a real go2 has it). Graded with the model's OWN previous
  answer, episode by episode, as it would run.
* **A clock in the prompt.** A pose is a timed sequence -- stand, sit down,
  hold, get up -- and nothing in one frame says which part is due. Without
  the seconds since the request, the model sent each part ~0.5-1 s late and
  the dog missed the pose (paligemma_pose_v2, closed loop). A robot knows
  when it was told.
* **Lean in the prompt.** Which way to sweep follows the foot loads; without
  them the heading is a coin flip.
* **Actions from a head, not text.** ~150 numbers as generated tokens
  would cost seconds a step. A small MLP on the last prefix token's hidden state
  gives them in the same forward pass that reads the image; only the short
  answer is generated -- and at run time only when it changes.

Reads the recorder's staging layout (parquet + mp4 per episode), holds out
whole episodes, trains LoRA on the language model plus the head (vision
tower frozen), writes adapter/, head.pt and eval.json. Run in the lerobot
venv:

    ~/sparkpack/lerobot/.venv/bin/python scripts/vla_paligemma_chase.py \\
        --data output/datasets/go2_qa_v1 --out output/vla/paligemma_qa_v2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, nargs="+", required=True,
                   help="One or more recorder staging dirs (e.g. an open-ground and a "
                        "warehouse recording). Each is held out from and graded separately.")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Use at most this many episodes from EACH recording.")
    p.add_argument("--out", type=Path, required=True, help="Where the adapter and eval go.")
    p.add_argument("--model", default="google/paligemma2-3b-pt-224")
    p.add_argument("--holdout", type=float, default=0.15, help="Share of episodes held out.")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--action-weight", type=float, default=1.0,
                   help="Weight of the action loss against the answer loss.")
    p.add_argument("--flat-action-loss", action="store_true",
                   help="Average the action loss over every number equally, instead of "
                        "over its parts (see action_groups).")
    p.add_argument("--eval-frames", type=int, default=600,
                   help="Grade whole held-out episodes until this many frames.")
    p.add_argument("--prev-noise", type=float, default=0.2,
                   help="In training, replace the previous answer with another answer "
                        "possible for that request this often. Taught only on its correct "
                        "past, a model copies its first mistake forever at run time.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-only", action="store_true", help="Load --out's model and grade.")
    return p.parse_args()


# The action's columns, as the recordings name them (lerobot_recorder.
# masked_target_names, then the gaze). Set by load_episodes; every recording
# must agree.
ACTION_NAMES: list = []
NOTHING_SAID = "nothing yet"


def action_groups(names) -> dict:
    """The action's parts, as column indices: each counts equally in the loss.

    Averaged over all ~150 numbers equally, the dozen that steer a chase --
    where the trunk goes and which way it faces -- are a sliver of the loss;
    the other ~130 are leg targets and bits, zero whenever a ball is wanted.
    Measured with the flat loss (paligemma_pose_v1): ball-frame trunk error
    41 cm / 33 deg against the 16 cm / 18 deg of a model taught only those.
    """
    trunk = {f"base_link_{f}" for f in ("x", "y", "z")}
    turn = {f"base_link_{f}" for f in ("fwd_x", "fwd_y", "fwd_z", "up_x", "up_y", "up_z")}
    groups = {"trunk_pos": [], "trunk_rot": [], "legs_pos": [], "legs_rot": [],
              "bits": [], "seconds": [], "gaze": []}
    for i, c in enumerate(names):
        if c in trunk:
            groups["trunk_pos"].append(i)
        elif c in turn:
            groups["trunk_rot"].append(i)
        elif c.endswith("_on"):
            groups["bits"].append(i)
        elif c == "seconds":
            groups["seconds"].append(i)
        elif c.startswith("gaze_"):
            groups["gaze"].append(i)
        elif c.rsplit("_", 1)[-1] in ("x", "y", "z") and "_fwd_" not in c and "_up_" not in c:
            groups["legs_pos"].append(i)
        else:
            groups["legs_rot"].append(i)
    return {k: v for k, v in groups.items() if v}


def trunk_heading(action, names=None) -> float:
    """The trunk's commanded heading (egocentric, rad) from an action vector."""
    names = names or ACTION_NAMES
    fx, fy = action[names.index("base_link_fwd_x")], action[names.index("base_link_fwd_y")]
    return float(math.atan2(fy, fx)) if abs(fx) + abs(fy) > 1e-6 else 0.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_jsonl(path: Path, key: str):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r[f"{key}_index"]: r[key] for r in rows}


def read_video(path: Path) -> np.ndarray:
    import av

    with av.open(str(path)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


def load_episodes(root: Path, max_episodes=None):
    """Every recorded frame as a dict, grouped by episode.

    Adds what a robot would carry forward itself: the previous answer, the
    yaw turned since the throw (|gyro z| integrated at the recording rate)
    and which side the weight is on.
    """
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text())
    tasks = read_jsonl(root / "meta" / "tasks.jsonl", "task")
    responses = read_jsonl(root / "meta" / "responses.jsonl", "response")
    names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    if ACTION_NAMES and ACTION_NAMES != action_names:
        raise ValueError(f"{root}: its action columns differ from the other recordings'")
    ACTION_NAMES[:] = action_names
    dt = 1.0 / float(info["fps"])
    cam = next(k for k, v in info["features"].items() if v["dtype"] == "video")
    gaze = [names.index(n) for n in ("gaze_u", "gaze_v", "gaze_zoom")]
    yaw_rate = names.index("ang_vel_z")
    loads = {n: i for i, n in enumerate(names) if n.endswith(".load")}
    left = [i for n, i in loads.items() if len(n) > 1 and n[1] in "Ll"]
    right = [i for n, i in loads.items() if len(n) > 1 and n[1] in "Rr"]
    episodes = []
    for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
        index = json.loads(line)["episode_index"]
        table = pq.read_table(root / "data" / "chunk-000" / f"episode_{index:06d}.parquet").to_pydict()
        frames = read_video(root / "videos" / "chunk-000" / cam / f"episode_{index:06d}.mp4")
        n = min(len(frames), len(table["action"]))
        turned, prev, prev_heading, episode = 0.0, NOTHING_SAID, 0.0, []
        # A VLA-driven (DAgger) recording carries what the driver itself
        # had: its own last answer and heading. Use those, not the labels.
        driven = "input_prev_heading" in table
        for i in range(n):
            if driven:
                k = table["input_prev_response_index"][i]
                prev = responses.get(k, NOTHING_SAID) if k >= 0 else NOTHING_SAID
                prev_heading = float(table["input_prev_heading"][i])
            state = table["observation.state"][i]
            action = np.asarray(table["action"][i], dtype=np.float32)
            weight = sum(state[j] for j in left) - sum(state[j] for j in right)
            said = responses[table["response_index"][i]]
            episode.append({
                "image": frames[i],
                "prompt": tasks[table["task_index"][i]],
                # An episode starts at its throw (the recorder cuts there).
                "elapsed": i * dt,
                "response": said,
                "prev": prev,
                "prev_heading": prev_heading,
                "turned": turned,
                "lean": "left" if weight >= 0 else "right",
                "gaze": [state[g] for g in gaze],
                "action": action,
                # Unique across recordings, and says which scene it came from.
                "episode": f"{root.name}/{index}",
                "source": root.name,
            })
            turned += abs(state[yaw_rate]) * dt
            prev = said
            prev_heading = trunk_heading(action)
        episodes.append(episode)
    return episodes


def prefix_text(s, prev=None, prev_heading=None) -> str:
    """The request plus what the dog carries forward itself.

    `turning` is its own last commanded heading: a search sweeps 60-90 deg
    one way or the other, and without knowing which way it was already
    going a squared-error head averages left and right into standing still.
    """
    u, v, z = s["gaze"]
    said = s["prev"] if prev is None else prev
    turn = s.get("prev_heading", 0.0) if prev_heading is None else prev_heading
    return (f"{s['prompt']} | asked {s.get('elapsed', 0.0):.1f}s ago | said: {said} "
            f"| turned {math.degrees(s['turned']):.0f} "
            f"| turning {math.degrees(turn):+.0f} | lean {s['lean']} "
            f"| gaze {u:+.2f} {v:+.2f} {z:.1f}")


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

COLORS = ("red", "green", "blue", "yellow")
# The recorder's pose phrasings (LeRobotRecorderConfig.prompts pose_*).
POSE_PROMPTS = {"sit", "sit down", "sit!", "beg", "sit up and beg", "beg for it",
                "lie down", "down", "lay down"}


def answer_kind(text: str):
    """What an answer commits to: ('yes'|'no'|'pending', colour or None)."""
    t = text.lower()
    colour = next((c for c in COLORS if c in t), None)
    if t.startswith("no") or "can't find" in t:
        return "no", colour
    # "getting the green ball", "I'm sitting": found it / done it.
    if t.startswith("yes") or t.startswith("getting the") or t.startswith("i'm"):
        return "yes", colour
    return "pending", colour


def colours_in_view(image: np.ndarray):
    """Which ball colours have a clear blob in the crop (a pixel count)."""
    f = image.astype(int)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    masks = {
        "red": (r > 140) & (g < 90) & (b < 90),
        "green": (g > 120) & (r < 90) & (b < 100),
        "blue": (b > 140) & (r < 90) & (b - g > 60),
    }
    return {c for c, m in masks.items() if int(m.sum()) >= 8}


def asked_colour(prompt: str):
    return next((c for c in COLORS if c in prompt.lower()), None)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ActionHead(nn.Module):
    """The normalised action vector from one hidden state."""

    def __init__(self, hidden: int, mean, std):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, 512), nn.GELU(), nn.Linear(512, len(mean)))
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, h):
        return self.net(h.float())

    def denorm(self, y):
        return y * self.std + self.mean


def last_prefix_state(hidden, token_type_ids, attention_mask):
    """Hidden state of each row's last prefix token (image + prompt).

    The prefix attends only to itself, so this is the same whether or not
    an answer follows -- training and inference read the same vector.
    """
    prefix = (token_type_ids == 0) & (attention_mask == 1)
    positions = torch.arange(prefix.shape[1], device=prefix.device).expand_as(prefix)
    last = torch.where(prefix, positions, torch.full_like(positions, -1)).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


def load_model(name: str, device: str):
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    processor = PaliGemmaProcessor.from_pretrained(name)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        name, torch_dtype=torch.bfloat16
    ).to(device)
    return processor, model


def add_lora(model, r: int):
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad_(False)
    config = LoraConfig(
        r=r,
        lora_alpha=2 * r,
        lora_dropout=0.05,
        # Language model only: the vision tower already sees colour; what is
        # new is reading the request and saying and doing the right thing.
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def corrupt_prev(batch, answers_by_prompt, p, rng):
    """Previous answers for a training batch, some deliberately wrong."""
    prevs = []
    for s in batch:
        if rng.random() < p:
            options = [a for a in answers_by_prompt[s["prompt"]] if a != s["prev"]]
            prevs.append(rng.choice(options) if options else s["prev"])
        else:
            prevs.append(s["prev"])
    return prevs


def encode(processor, samples, device, prevs=None, with_answer=True, prev_headings=None):
    from PIL import Image

    kwargs = dict(
        text=["<image>" + prefix_text(
                  s, None if prevs is None else prevs[i],
                  None if prev_headings is None else prev_headings[i])
              for i, s in enumerate(samples)],
        images=[Image.fromarray(s["image"]) for s in samples],
        return_tensors="pt",
        padding="longest",
    )
    if with_answer:
        kwargs["suffix"] = [s["response"] for s in samples]
    inputs = processor(**kwargs).to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    if "token_type_ids" not in inputs:
        inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    return inputs


def batches(samples, size, rng):
    order = list(range(len(samples)))
    while True:
        rng.shuffle(order)
        for i in range(0, len(order) - size + 1, size):
            yield [samples[j] for j in order[i : i + size]]


def train(model, head, processor, samples, args, device):
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    warm = max(args.steps // 20, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm)
        * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)),
    )
    rng = random.Random(args.seed)
    it = batches(samples, args.batch, rng)
    groups = [torch.tensor(v, device=device) for v in action_groups(ACTION_NAMES).values()]
    answers_by_prompt = {}
    for s in samples:
        answers_by_prompt.setdefault(s["prompt"], {NOTHING_SAID}).add(s["response"])
    answers_by_prompt = {k: sorted(v) for k, v in answers_by_prompt.items()}
    t0 = time.time()
    for step in range(args.steps):
        batch = next(it)
        prevs = corrupt_prev(batch, answers_by_prompt, args.prev_noise, rng)
        inputs = encode(processor, batch, device, prevs=prevs)
        out = model(**inputs, output_hidden_states=True)
        h = last_prefix_state(out.hidden_states[-1], inputs["token_type_ids"], inputs["attention_mask"])
        target = torch.as_tensor(np.stack([s["action"] for s in batch]), device=device)
        err = (head(h) - (target - head.mean) / head.std) ** 2
        if args.flat_action_loss:
            act_loss = err.mean()
        else:
            act_loss = torch.stack([err[:, g].mean() for g in groups]).mean()
        loss = out.loss + args.action_weight * act_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 25 == 0 or step == args.steps - 1:
            rate = (step + 1) / max(time.time() - t0, 1e-6)
            print(f"step {step:5d}  answer {out.loss.item():.4f}  action {act_loss.item():.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {rate:.2f} it/s", flush=True)
    model.eval()


@torch.no_grad()
def evaluate(model, head, processor, episodes, args, device):
    """Episode by episode, each step fed the model's OWN previous answer."""
    model.eval()
    if getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    rng = random.Random(args.seed + 1)
    by_source = {}
    for ep in episodes:
        by_source.setdefault(ep[0]["source"], []).append(ep)
    for eps in by_source.values():
        rng.shuffle(eps)
    # Alternate recordings so a frame budget covers every scene.
    order = [ep for group in zip(*by_source.values()) for ep in group]
    order += [ep for eps in by_source.values() for ep in eps[min(map(len, by_source.values())):]]
    rows, act_times, ans_times = [], [], []
    for episode in order:
        if len(rows) >= args.eval_frames:
            break
        prev, prev_heading = NOTHING_SAID, 0.0
        for s in episode:
            inputs = encode(processor, [s], device, prevs=[prev], with_answer=False,
                            prev_headings=[prev_heading])
            inputs.pop("labels", None)
            torch.cuda.synchronize()
            t0 = time.time()
            out = model(**inputs, output_hidden_states=True, logits_to_keep=1)
            h = last_prefix_state(out.hidden_states[-1], inputs["token_type_ids"], inputs["attention_mask"])
            action = head.denorm(head(h))[0].cpu().numpy()
            torch.cuda.synchronize()
            act_times.append(time.time() - t0)
            t0 = time.time()
            gen = model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            ans_times.append(time.time() - t0)
            said = processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            rows.append({
                "episode": s["episode"], "source": s["source"], "prompt": s["prompt"], "prev": prev,
                "truth": s["response"], "said": said,
                "action": action.tolist(), "truth_action": s["action"].tolist(),
                "in_view": sorted(colours_in_view(s["image"])),
            })
            prev = said
            prev_heading = trunk_heading(action)

    def rate(sel):
        sel = list(sel)
        if not sel:
            return None
        return sum(1 for r in sel if answer_kind(r["said"]) == answer_kind(r["truth"])) / len(sel)

    decided = [r for r in rows if answer_kind(r["truth"])[0] != "pending"]
    hard_no = [
        r for r in decided
        if answer_kind(r["truth"])[0] == "no"
        and asked_colour(r["prompt"]) is not None
        and any(c != asked_colour(r["prompt"]) for c in r["in_view"])
    ]
    def action_errors(sel):
        """Mean errors of the action's parts, each where the label switches it on."""
        sel = list(sel)
        if not sel:
            return {}
        n = ACTION_NAMES
        pred = np.array([r["action"] for r in sel])
        true = np.array([r["truth_action"] for r in sel])
        bodies = [c[: -len("_on")] for c in n if c.endswith("_on") and not c.endswith("_rot_on")]
        legs = [b for b in bodies if b != "base_link"]

        def pos_err(bs):
            errs = []
            for b in bs:
                on = true[:, n.index(f"{b}_on")] > 0.5
                if on.any():
                    cols = [n.index(f"{b}_{f}") for f in ("x", "y", "z")]
                    errs.append(np.linalg.norm(pred[on][:, cols] - true[on][:, cols], axis=-1))
            return round(float(np.concatenate(errs).mean()), 4) if errs else None

        def heading_err():
            on = true[:, n.index("base_link_rot_on")] > 0.5
            if not on.any():
                return None
            h = lambda a: np.arctan2(a[:, n.index("base_link_fwd_y")], a[:, n.index("base_link_fwd_x")])  # noqa: E731
            d = h(pred[on]) - h(true[on])
            return round(float(np.degrees(np.abs(np.arctan2(np.sin(d), np.cos(d)))).mean()), 2)

        bit_cols = [i for i, c in enumerate(n) if c.endswith("_on")]
        out = {
            "trunk_pos_m": pos_err(["base_link"]),
            "trunk_heading_deg": heading_err(),
            "legs_pos_m": pos_err(legs),
            "on_bits_acc": round(float(((pred[:, bit_cols] > 0.5) == (true[:, bit_cols] > 0.5)).mean()), 4),
            "seconds": round(float(np.abs(pred[:, n.index("seconds")] - true[:, n.index("seconds")]).mean()), 4),
        }
        for g in ("gaze_u", "gaze_v", "gaze_zoom"):
            if g in n:
                out[g] = round(float(np.abs(pred[:, n.index(g)] - true[:, n.index(g)]).mean()), 4)
        return out

    def summary(sel):
        sel = list(sel)
        dec = [r for r in sel if answer_kind(r["truth"])[0] != "pending"]
        return {
            "frames": len(sel),
            "answer_kind_all": rate(sel),
            "answer_kind_decided": rate(dec),
            "action": action_errors(sel),
        }

    def is_pose(r):
        return r["prompt"] in POSE_PROMPTS

    report = {
        "frames": len(rows),
        "episodes": len({r["episode"] for r in rows}),
        "exact_answer": sum(r["said"] == r["truth"] for r in rows) / len(rows),
        "answer_kind_all": rate(rows),
        "yes_frames": rate(r for r in decided if answer_kind(r["truth"])[0] == "yes"),
        "no_frames": rate(r for r in decided if answer_kind(r["truth"])[0] == "no"),
        "pending_frames": rate(r for r in rows if answer_kind(r["truth"])[0] == "pending"),
        "hard_no_other_colour_in_view": rate(hard_no),
        "hard_no_count": len(hard_no),
        "action": action_errors(rows),
        "poses": summary(r for r in rows if is_pose(r)),
        "balls": summary(r for r in rows if not is_pose(r)),
        "latency_action_s_median": float(np.median(act_times)),
        "latency_answer_s_median": float(np.median(ans_times)),
        "by_source": {
            src: summary(r for r in rows if r["source"] == src)
            for src in sorted({r["source"] for r in rows})
        },
    }
    return report, rows


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"
    episodes, held = [], set()
    rng = random.Random(args.seed)
    for root in args.data:
        eps = [ep for ep in load_episodes(root, args.max_episodes) if ep]
        order = list(range(len(eps)))
        rng.shuffle(order)
        n_hold = max(1, int(round(len(eps) * args.holdout)))
        held |= {eps[i][0]["episode"] for i in order[:n_hold]}
        print(f"{root}: {len(eps)} episodes, {n_hold} held out")
        episodes += eps
    train_s = [s for ep in episodes for s in ep if s["episode"] not in held]
    test_eps = [ep for ep in episodes if ep[0]["episode"] in held]
    print(f"{len(episodes)} episodes: {len(train_s)} train frames, "
          f"{sum(map(len, test_eps))} held-out frames from {len(held)} episodes")
    print("example:", prefix_text(train_s[5]), "=>", train_s[5]["response"], train_s[5]["action"].round(2))

    args.out.mkdir(parents=True, exist_ok=True)
    processor, model = load_model(args.model, device)
    hidden = model.config.text_config.hidden_size
    if args.eval_only:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.out / "adapter")
        state = torch.load(args.out / "head.pt", map_location=device)
        head = ActionHead(hidden, state["mean"], state["std"]).to(device)
        head.load_state_dict(state)
    else:
        acts = np.stack([s["action"] for s in train_s])
        head = ActionHead(hidden, acts.mean(0), acts.std(0) + 1e-3).to(device)
        model = add_lora(model, args.lora_r)
        train(model, head, processor, train_s, args, device)
        model.save_pretrained(args.out / "adapter")
        torch.save(head.state_dict(), args.out / "head.pt")
    # Merged for grading: the run-time model, not LoRA's extra matmuls.
    model = model.merge_and_unload()
    report, rows = evaluate(model, head, processor, test_eps, args, device)
    (args.out / "eval.json").write_text(json.dumps(report, indent=2))
    with open(args.out / "eval_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
