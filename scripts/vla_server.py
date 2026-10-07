# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Serve a fine-tuned chase VLA (vla_paligemma_chase.py) to a running sim.

The sim sends, per env, what the dog has: its 224x224 eye crop, the request,
how long ago it was asked, what it last said, how far it has turned since
the throw, which way it leans, and where the eye is looking. The server
answers with the action (the MaskedMimic conditioning in the robot's frame,
then the next gaze) and -- when asked -- the answer text. Same split as a
real deployment: the model on a 4090, the robot on the other end of a socket.

Plain multiprocessing.connection, so the lerobot venv (model) and the Isaac
venv (sim) need nothing in common but Python:

    CUDA_VISIBLE_DEVICES=1 ~/sparkpack/lerobot/.venv/bin/python scripts/vla_server.py \\
        --model-dir output/vla/paligemma_pose_v3

Then run the chase with --vla-server localhost:6010.
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vla_paligemma_chase as vla  # noqa: E402

AUTHKEY = b"protomotions-vla"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-dir", type=Path, required=True,
                   help="A --out of vla_paligemma_chase.py (adapter/ and head.pt).")
    p.add_argument("--base", default="google/paligemma2-3b-pt-224")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=6010)
    return p.parse_args()


def load(args, device):
    from peft import PeftModel

    processor, model = vla.load_model(args.base, device)
    hidden = model.config.text_config.hidden_size
    model = PeftModel.from_pretrained(model, args.model_dir / "adapter").merge_and_unload()
    model.eval()
    model.config.use_cache = True
    state = torch.load(args.model_dir / "head.pt", map_location=device)
    head = vla.ActionHead(hidden, state["mean"], state["std"]).to(device)
    head.load_state_dict(state)
    head.eval()
    return processor, model, head


@torch.no_grad()
def serve_one(request, processor, model, head, device):
    """[per-env dict] -> [per-env reply]. One batch through the model."""
    samples = [
        {
            "image": np.frombuffer(r["image"], dtype=np.uint8).reshape(r["shape"]),
            "prompt": r["prompt"],
            "prev": r["prev"],
            "prev_heading": float(r.get("prev_heading", 0.0)),
            "turned": float(r["turned"]),
            "elapsed": float(r.get("elapsed", 0.0)),
            "lean": r["lean"],
            "gaze": list(r["gaze"]),
        }
        for r in request["envs"]
    ]
    t0 = time.time()
    inputs = vla.encode(processor, samples, device, with_answer=False)
    inputs.pop("labels", None)
    # Only the hidden states are needed here; logits for all ~300 positions
    # over a 257k vocabulary are ~6 GB at a batch of 20 and are thrown away.
    out = model(**inputs, output_hidden_states=True, logits_to_keep=1)
    h = vla.last_prefix_state(out.hidden_states[-1], inputs["token_type_ids"], inputs["attention_mask"])
    actions = head.denorm(head(h)).cpu().numpy()
    act_ms = (time.time() - t0) * 1000.0
    answers = [None] * len(samples)
    ans_ms = 0.0
    if request.get("want_answer"):
        t0 = time.time()
        gen = model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True)
        n = inputs["input_ids"].shape[1]
        answers = [processor.decode(g[n:], skip_special_tokens=True).strip() for g in gen]
        ans_ms = (time.time() - t0) * 1000.0
    return {
        "actions": actions.tolist(),
        "answers": answers,
        "action_ms": act_ms,
        "answer_ms": ans_ms,
    }


def main():
    args = parse_args()
    device = "cuda"
    processor, model, head = load(args, device)
    print(f"[vla-server] {args.model_dir} loaded; listening on {args.host}:{args.port}", flush=True)
    with Listener((args.host, args.port), authkey=AUTHKEY) as listener:
        while True:
            conn = listener.accept()
            print(f"[vla-server] sim connected from {listener.last_accepted}", flush=True)
            try:
                while True:
                    request = conn.recv()
                    conn.send(serve_one(request, processor, model, head, device))
                    # Hand the batch's activations back: the sim shares this
                    # GPU, and PyTorch's cache otherwise keeps the peak forever.
                    torch.cuda.empty_cache()
            except (EOFError, ConnectionResetError, BrokenPipeError):
                print("[vla-server] sim disconnected; waiting for the next one", flush=True)
            finally:
                conn.close()


if __name__ == "__main__":
    main()
