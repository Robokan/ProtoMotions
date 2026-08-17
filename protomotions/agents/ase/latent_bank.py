# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Named skill latents encoded from motion clips ("press B to sit").

An ASE low-level controller already knows every skill in its pretrain corpus;
a latent is the handle. The MI encoder (which lives inside the discriminator
and emits ``mi_enc_output``) maps a motion window BACK to the latent that
produces it, so a clip name becomes a z the frozen LLC will execute -- no new
reward, no retraining. That is the whole trick behind the button override in
ASEHLCAgent.

Build a bank:

    python -m protomotions.agents.ase.latent_bank \\
        --llc-checkpoint results/anymal_ase_getup_v4/last.ckpt \\
        --motion-file data/motions/anymal_d/anymal_d_flat.pt \\
        --clip 12 --name sit --clip 40 --name jump \\
        --out data/anymal_latents.pt

CAVEAT: one fixed latent reproduces a SUSTAINED skill (a sit, a stance) far
better than a transient one (a jump: crouch, launch, land). If a button skill
looks mushy, that is the signal to train the button as a command with an
imitation reward instead of piping a constant z.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import torch
from tensordict import TensorDict
from torch import Tensor


@torch.no_grad()
def encode_clip(
    discriminator,
    motion_lib,
    motion_id: int,
    history_steps: int,
    dt: float,
    *,
    num_samples: int = 64,
    start_frac: float = 0.1,
    end_frac: float = 0.9,
    local_obs: bool = True,
    root_height_obs: bool = True,
    observe_contacts: bool = False,
    body_ids: Optional[List[int]] = None,
    device: str = "cpu",
) -> Tensor:
    """Mean encoder latent over a clip, projected to the unit hypersphere.

    Samples times across the clip's interior (the ends of a trimmed clip are
    usually transitions, not the skill) and averages the encoder's output.
    The mean of unit vectors is not a unit vector, hence the renormalize --
    the LLC only ever sees latents on the sphere (ASE.sample_latents).
    """
    from protomotions.envs.obs import compute_historical_max_coords_from_motion_lib

    length = float(motion_lib.get_motion_length(
        torch.tensor([motion_id], device=device)).item())
    times = torch.linspace(
        start_frac * length, end_frac * length, num_samples, device=device
    )
    ids = torch.full((num_samples,), int(motion_id), dtype=torch.long, device=device)
    hist = compute_historical_max_coords_from_motion_lib(
        motion_lib=motion_lib,
        motion_ids=ids,
        motion_times=times,
        num_state_history_steps=history_steps,
        dt=dt,
        local_obs=local_obs,
        root_height_obs=root_height_obs,
        observe_contacts=observe_contacts,
        body_ids=body_ids,
    )
    td = TensorDict({"historical_max_coords_obs": hist}, batch_size=hist.shape[0])
    td = discriminator(td)
    z = torch.nn.functional.normalize(td["mi_enc_output"], dim=-1)
    return torch.nn.functional.normalize(z.mean(dim=0), dim=-1)


def load_bank(path: str, device: str = "cpu") -> Dict[str, Tensor]:
    """Load {name: latent} and re-project (a hand-edited bank may drift)."""
    bank = torch.load(path, weights_only=False, map_location=device)
    return {
        k: torch.nn.functional.normalize(v.to(device).float(), dim=-1)
        for k, v in bank.items()
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llc-checkpoint", required=True)
    ap.add_argument("--motion-file", required=True)
    ap.add_argument("--robot-name", required=True)
    ap.add_argument("--clip", type=int, action="append", required=True,
                    help="Motion index to encode (repeatable, pairs with --name).")
    ap.add_argument("--name", action="append", required=True,
                    help="Skill name for the preceding --clip.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--history-steps", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-root-height", action="store_true",
                    help="Match an LLC trained with root_height_obs=False "
                         "(e.g. Atlas after 2026-08-17).")
    args = ap.parse_args()
    if len(args.clip) != len(args.name):
        raise SystemExit("--clip and --name must be given in pairs")

    from protomotions.agents.common.config import PretrainedModelConfig
    from protomotions.agents.common.pretrained import load_pretrained_model_module
    from protomotions.components.motion_lib import MotionLib, MotionLibConfig
    from protomotions.robot_configs.factory import robot_config

    rc = robot_config(args.robot_name)
    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=args.motion_file), device=args.device
    )
    # Same loader the HLC uses for its frozen LLC discriminator, so the
    # encoder here is byte-identical to the one running at inference.
    disc = load_pretrained_model_module(
        PretrainedModelConfig(
            checkpoint_path=args.llc_checkpoint, module_path="discriminator"
        ),
        device=torch.device(args.device),
    )
    disc.eval()

    dt = 1.0 / 30.0
    bank = {}
    for clip, name in zip(args.clip, args.name):
        bank[name] = encode_clip(
            disc, motion_lib, clip,
            history_steps=args.history_steps, dt=dt,
            root_height_obs=not args.no_root_height,
            body_ids=getattr(rc, "disc_bodies_subset", None) and None,
            device=args.device,
        )
        print(f"encoded '{name}' from clip {clip}: |z|={bank[name].norm():.3f}")
    torch.save(bank, args.out)
    print(f"wrote {len(bank)} latents to {args.out}")


if __name__ == "__main__":
    _main()
