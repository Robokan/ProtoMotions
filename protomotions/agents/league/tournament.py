# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Battle evaluation tournament (SOMA_GPC_COMBAT_PLAN Phase 7).

Round-robin (and single-pairing) evaluation over slim adapter checkpoints on
a paired BattleEnv, decoupled from training-time statistics:

- All parallel matches run one pairing at a time (adapter A on the ego half,
  adapter B on the opponent half) until the requested match count completes,
  with randomized spawns per match.
- Outcomes update Elo ratings; results aggregate into a ladder table and a
  head-to-head matrix (the head-to-head matrix is what exposes
  non-transitivity — the interesting result).
- ``regression_gate`` is the clean-statistics admission check the plan
  requires before a snapshot enters the league: the candidate must beat the
  previous snapshot at ``threshold`` (default 55%) over a fixed match count.
"""

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from protomotions.agents.league.elo import elo_update
from protomotions.agents.league.pfsp import DRAW_WEIGHT

log = logging.getLogger(__name__)


@dataclass
class PairingResult:
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0

    @property
    def matches(self) -> int:
        return self.wins_a + self.wins_b + self.draws

    def score_a(self) -> float:
        if self.matches == 0:
            return 0.5
        return (self.wins_a + DRAW_WEIGHT * self.draws) / self.matches


@dataclass
class TournamentReport:
    adapters: List[str] = field(default_factory=list)
    ratings: Dict[str, float] = field(default_factory=dict)
    head_to_head: Dict[str, Dict[str, dict]] = field(default_factory=dict)

    def ladder(self) -> List[Tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda kv: -kv[1])

    def to_json(self, path: str) -> None:
        payload = {
            "ladder": [
                {"adapter": name, "elo": round(rating, 1)}
                for name, rating in self.ladder()
            ],
            "head_to_head": self.head_to_head,
        }
        Path(path).write_text(json.dumps(payload, indent=2))
        log.info("Tournament report written to %s", path)


class BattleTournament:
    """Drives adapter-vs-adapter matches on a league agent's environment.

    Requires a fully-built league agent (see ``battle_tournament.py``); uses
    its model for the ego side, its opponent lanes for the other side, and
    its SelfPlayEnvAdapter for match accounting.
    """

    def __init__(
        self,
        agent,
        deterministic: bool = False,
        action_hold: int = 1,
        autocast_dtype: Optional[str] = None,
        sampling_mode: Optional[str] = None,
    ):
        self.agent = agent
        self.env = agent.env  # SelfPlayEnvAdapter
        self.device = agent.device
        self.deterministic = deterministic
        # Override the prior's decode sampling (None = keep trained mode).
        # "nucleus" skips the per-token reference (prior-constraint) forward,
        # halving the sequential decodes — the main launch-bound cost at the
        # low batch of the viewer. Trained under prior_constraint, so behavior
        # may shift slightly; inference/viewing only.
        self.sampling_mode = sampling_mode

        # Optional bf16/fp16 autocast around the prior forwards. The prior runs
        # in fp32; on Blackwell tensor cores bf16 matmuls are 2-4x faster, so
        # this can help even at batch 2 (where the fp32 path barely uses the
        # tensor cores). Weights stay fp32; only the matmuls run reduced-
        # precision, and the action output auto-casts back. Inference only.
        import contextlib

        if autocast_dtype in ("bf16", "bfloat16"):
            self._autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        elif autocast_dtype in ("fp16", "float16", "half"):
            self._autocast = lambda: torch.autocast("cuda", dtype=torch.float16)
        else:
            self._autocast = contextlib.nullcontext

        # Viewer smoothness: re-decode each fighter's policy only every
        # `action_hold` control steps, reusing the action between. Each decode
        # is 8 sequential autoregressive prior forwards (~100ms at batch 2),
        # so holding for 2-3 steps ~2-3x's the frame rate. Physics still steps
        # every frame, so motion stays smooth. Eval/scoring only; 1 = off.
        self.action_hold = max(1, int(action_hold))
        self._frame = 0
        self._held_opp = None

        # Take over the adapter's callbacks for evaluation
        self._outcomes: List[int] = []  # +1 A wins, -1 B wins, 0 draw
        self.env.set_match_end_callback(self._record)
        self.env.set_opponent_policy(self._opponent_policy)
        self._opponent_model = None

        # Make sure the league agent never engages its own training league
        agent._league_initialized = True

    # ------------------------------------------------------------------
    def _record(self, ego_ids, win, lose, draw) -> None:
        for i in range(len(ego_ids)):
            if win[i] > 0.5:
                self._outcomes.append(1)
            elif lose[i] > 0.5:
                self._outcomes.append(-1)
            else:
                self._outcomes.append(0)

    def _opponent_policy(self, opp_obs):
        # Reuse the held opponent action between decode frames (see action_hold)
        if self.action_hold > 1 and self._held_opp is not None and (
            self._frame % self.action_hold != 0
        ):
            return self._held_opp
        obs_td = self.agent._opponent_obs_td(opp_obs)
        with torch.no_grad(), self._autocast():
            out = self._opponent_model(obs_td)
        key = "mean_action" if self.deterministic and "mean_action" in out else "action"
        self._held_opp = out[key].float()
        return self._held_opp

    @staticmethod
    def _set_sampling_mode(model, mode: Optional[str]) -> None:
        """Override the prior's sampling mode (e.g. 'nucleus' at inference to
        skip the per-token reference forward — halves the sequential decodes)."""
        if mode is None:
            return
        actor = getattr(model, "_actor", model)
        pwp = getattr(actor, "prior_with_peft", None)
        if pwp is not None:
            pwp.sampling_mode = mode

    def _load_ego(self, adapter_path: str) -> None:
        self.agent.load_adapter_checkpoint(adapter_path)
        self._set_sampling_mode(self.agent._unwrapped_model(), self.sampling_mode)

    def _load_opponent(self, adapter_path: str) -> None:
        if self.agent._lanes is None:
            self.agent._build_lanes()
        lanes = self.agent._lanes
        state = torch.load(adapter_path, map_location=self.device, weights_only=False)
        adapter_state = state["model"] if "model" in state else state
        adapter_state = {k: v.to(self.device) for k, v in adapter_state.items()}
        lanes.lane_member = [None] * lanes.num_lanes
        lanes.assign(0, adapter_state)
        self._opponent_model = lanes.lanes[lanes._lane_of(0)]
        self._set_sampling_mode(self._opponent_model, self.sampling_mode)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_pairing(
        self, adapter_a: str, adapter_b: str, matches: int, max_steps: int = 200000
    ) -> PairingResult:
        """Run ``matches`` A-vs-B matches across all parallel arenas."""
        self._load_ego(adapter_a)
        self._load_opponent(adapter_b)
        self._outcomes = []

        obs, _ = self.env.reset()
        model = self.agent.model
        model.eval()

        steps = 0
        held_ego = None
        while len(self._outcomes) < matches and steps < max_steps:
            obs = self.agent.add_agent_info_to_obs(obs)
            # Re-decode ego only every action_hold frames (opponent held in
            # _opponent_policy, keyed off the same self._frame).
            if held_ego is None or self._frame % self.action_hold == 0:
                obs_td = self.agent.obs_dict_to_tensordict(obs)
                with self._autocast():
                    out = model(obs_td)
                key = (
                    "mean_action"
                    if self.deterministic and "mean_action" in out
                    else "action"
                )
                held_ego = out[key].float()
            obs, _, dones, _, _ = self.env.step(held_ego)
            self._frame += 1
            done_ids = dones.nonzero(as_tuple=False).flatten()
            if len(done_ids) > 0:
                obs, _ = self.env.reset(done_ids)
            steps += 1

        outcomes = self._outcomes[:matches]
        result = PairingResult(
            wins_a=sum(1 for o in outcomes if o == 1),
            wins_b=sum(1 for o in outcomes if o == -1),
            draws=sum(1 for o in outcomes if o == 0),
        )
        log.info(
            "%s vs %s: %d-%d-%d (W-L-D over %d matches)",
            Path(adapter_a).stem,
            Path(adapter_b).stem,
            result.wins_a,
            result.wins_b,
            result.draws,
            result.matches,
        )
        return result

    @torch.no_grad()
    def probe(self, adapter_a: str, adapter_b: str, steps: int = 120) -> None:
        """Diagnostic rollout: print opponent-obs and engagement stats.

        Verifies the evaluation path end-to-end: adapters loaded, battle
        context populated, task observations flowing, policies reacting.
        """
        import time

        self._load_ego(adapter_a)
        self._load_opponent(adapter_b)
        obs, _ = self.env.reset()
        model = self.agent.model
        model.eval()

        inner = getattr(self.env, "inner", self.env)
        # Per-phase wall-clock accumulators (CUDA-synced so timings are real,
        # not just kernel-launch latency). Skip the first few steps (warm-up).
        t_model = t_step = t_markers = 0.0
        timed_from = 5

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        held_ego = None
        # Per contact-onset event (pre-speed-gate): impact speed, KE, group.
        event_speeds, event_kes, event_groups = [], [], []
        # Gait statistics (A/B diagnostics): per-step root height and
        # down/fallen fraction, split into early (pre-engagement) vs all.
        gait_heights, gait_down, gait_speed = [], [], []
        for step in range(steps):
            obs = self.agent.add_agent_info_to_obs(obs)

            _sync(); _t = time.perf_counter()
            # Honor action_hold so the probe measures the real exhibition path
            # (both fighters held between decode frames).
            if held_ego is None or self._frame % self.action_hold == 0:
                obs_td = self.agent.obs_dict_to_tensordict(obs)
                with self._autocast():
                    out = model(obs_td)
                held_ego = out["action"].float()
            _sync()
            if step >= timed_from:
                t_model += time.perf_counter() - _t

            _t = time.perf_counter()
            obs, _, dones, _, _ = self.env.step(held_ego)
            self._frame += 1
            _sync()
            if step >= timed_from:
                t_step += time.perf_counter() - _t

            done_ids = dones.nonzero(as_tuple=False).flatten()
            if len(done_ids) > 0:
                obs, _ = self.env.reset(done_ids)

            # Gait stats: root height, below-knockdown fraction, planar speed.
            rs = inner.simulator.get_root_state()
            gait_heights.append(rs.root_pos[:, 2].detach().cpu().clone())
            gait_down.append(
                (rs.root_pos[:, 2] < inner.battle_control.config.knockdown_height)
                .float().detach().cpu().clone()
            )
            gait_speed.append(
                rs.root_vel[:, :2].norm(dim=-1).detach().cpu().clone()
            )

            # Calibration: collect contact-onset events (pre-speed-gate).
            hs = inner.battle_control.hit_state
            mask = hs.last_event_speed > 0
            if mask.any():
                event_speeds.append(hs.last_event_speed[mask].detach().cpu())
                event_kes.append(hs.last_event_ke[mask].detach().cpu())
                if hs.strike_body_groups is not None:
                    event_groups.append(
                        hs.strike_body_groups[hs.last_event_striker[mask]]
                        .detach()
                        .cpu()
                    )

            if step == steps - 1:
                n = max(1, steps - timed_from)
                log.info(
                    "PROBE TIMING (avg ms/step over %d steps): "
                    "model=%.1f  env.step[sim+render]=%.1f  total=%.1f  (=%.1f fps)",
                    n,
                    1000 * t_model / n,
                    1000 * t_step / n,
                    1000 * (t_model + t_step) / n,
                    n / max(t_model + t_step, 1e-6),
                )
            if step % 30 == 0:
                ctx = inner.context
                b = ctx.battle
                root = inner.simulator.get_root_state().root_pos
                dist = float(torch.norm(root[0, :2] - b.opp_root_pos[0, :2]))
                task = obs.get("task_obs")
                log.info("probe step %d:", step)
                log.info(
                    "  dist=%.2fm | task_obs[0][:6]=%s | task_obs std=%.4f",
                    dist,
                    [round(float(v), 3) for v in task[0][:6]] if task is not None else "MISSING",
                    float(task.std()) if task is not None else -1.0,
                )
                log.info(
                    "  health=%s hit_dealt=%s downed=%s action_std=%.4f",
                    [round(float(v), 3) for v in b.health[:2]],
                    [round(float(v), 4) for v in b.hit_energy_dealt[:2]],
                    [round(float(v), 3) for v in b.downed[:2]],
                    float(held_ego.std()),
                )

        # Calibration report: per contact-onset event, the impact speed and
        # kinetic energy (0.5 m v^2) of the attributed striker, pre-speed-gate.
        if event_speeds:
            spd = torch.cat(event_speeds)
            kes = torch.cat(event_kes)
            qs = torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99])
            sp = torch.quantile(spd, qs)
            kp = torch.quantile(kes, qs)
            log.info(
                "HIT EVENTS: %d contact onsets | impact speed m/s "
                "p50=%.2f p75=%.2f p90=%.2f p95=%.2f p99=%.2f max=%.2f",
                spd.numel(),
                *[float(v) for v in sp],
                float(spd.max()),
            )
            log.info(
                "HIT EVENTS: kinetic energy J "
                "p50=%.1f p75=%.1f p90=%.1f p95=%.1f p99=%.1f max=%.1f",
                *[float(v) for v in kp],
                float(kes.max()),
            )
            if event_groups:
                grp = torch.cat(event_groups)
                labels = inner.battle_control.strike_group_labels
                for g, name in enumerate(labels):
                    sel = grp == g
                    if sel.any():
                        log.info(
                            "HIT EVENTS[%s]: n=%d speed p50=%.2f p95=%.2f | "
                            "KE p50=%.1f p95=%.1f",
                            name,
                            int(sel.sum()),
                            float(spd[sel].median()),
                            float(torch.quantile(spd[sel], 0.95)),
                            float(kes[sel].median()),
                            float(torch.quantile(kes[sel], 0.95)),
                        )
        else:
            log.info("HIT EVENTS: no contact onsets sampled")

        # Gait report (A/B diagnostics for rule-flag comparisons).
        if gait_heights:
            H = torch.stack(gait_heights)  # [T, 2N]
            D = torch.stack(gait_down)
            V = torch.stack(gait_speed)
            early = max(1, min(150, H.shape[0]))
            log.info(
                "GAIT: root height mean %.3f m (early %.3f) | down fraction "
                "%.1f%% (early %.1f%%) | planar speed mean %.2f m/s (early %.2f)",
                float(H.mean()), float(H[:early].mean()),
                100 * float(D.mean()), 100 * float(D[:early].mean()),
                float(V.mean()), float(V[:early].mean()),
            )

    @torch.no_grad()
    def record_pairing(
        self,
        adapter_a: str,
        adapter_b: str,
        out_path: str,
        max_frames: int = 1500,
        match_index: int = 0,
        title: Optional[str] = None,
        tail_frames: int = 20,
        bouts: int = 1,
    ) -> str:
        """Play A vs B and record arena ``match_index`` to an mp4 using the
        real IsaacSim viewport (the repo's RecordingMixin, record.py).

        Records ``bouts`` WHOLE bouts strung into one clip: each runs from its
        reset until that arena's match ends (knockout / ring-out / timeout),
        plus a short ``tail_frames`` so the finish stays on screen, then a
        brief black separator before the next bout. ``max_frames`` caps each
        individual bout (safety bound for a bout that never resolves). The
        camera follows the ego fighter's arena (the opponent shares it), the
        RGB annotator grabs each rendered frame, and frames are encoded to mp4.
        Requires a rendering sim — the CLI launches it headless with offscreen
        rendering (``enable_cameras``) and flips the sim to non-headless, so
        this works with no display. Returns a one-line outcome summary.
        """
        import numpy as np
        import os
        import signal
        import subprocess

        self._load_ego(adapter_a)
        self._load_opponent(adapter_b)

        inner = self.env.inner_env if hasattr(self.env, "inner_env") else self.env
        n = self.env.num_matches
        if not 0 <= match_index < n:
            raise ValueError(f"match_index {match_index} out of range [0,{n})")
        sim = inner.simulator
        if getattr(sim, "headless", True):
            raise RuntimeError(
                "record_pairing needs a rendering simulator; launch the CLI "
                "with --record (offscreen rendering) rather than --headless."
            )
        if not hasattr(sim, "grab_rgb_frame"):
            raise RuntimeError(
                f"{type(sim).__name__} has no grab_rgb_frame(); real-render "
                "recording is implemented for the IsaacLab simulator."
            )

        # Follow the ego fighter's arena; the opponent shares that arena.
        sim._camera_target = {"env": match_index, "element": 0}

        obs, _ = self.env.reset()
        model = self.agent.model
        model.eval()

        fps = int(max(10, min(30, round(1.0 / float(inner.dt)))))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        # Pipe raw RGB straight to an ffmpeg subprocess we control, rather than
        # via imageio. Under a running Kit app (this box's IsaacSim build) the
        # imageio-spawned ffmpeg dies mid-stream with a broken pipe — Kit's
        # signal/process-group handling kills the child. Starting ffmpeg in its
        # own session (setpgrp) with default SIGPIPE detaches it from Kit and
        # fixes that. Lazily opened on the first frame (needs the frame size).
        try:
            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"

        _writer = {"proc": None}

        def _child_preexec():
            os.setpgrp()  # detach from Kit's process group
            try:
                signal.signal(signal.SIGPIPE, signal.SIG_DFL)
            except Exception:
                pass

        def _append(fr):
            proc = _writer["proc"]
            if proc is None:
                h, w = fr.shape[:2]
                cmd = [
                    ffmpeg_exe, "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-vcodec", "rawvideo",
                    "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
                    "-i", "-", "-an",
                    "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out_path,
                ]
                proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, preexec_fn=_child_preexec,
                )
                _writer["proc"] = proc
            proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())

        def _close_writer():
            proc = _writer["proc"]
            if proc is None:
                return
            proc.stdin.close()
            err = proc.stderr.read().decode("utf-8", "ignore")
            proc.wait()
            if proc.returncode not in (0, None):
                raise RuntimeError(
                    f"ffmpeg exited {proc.returncode}: {err.strip()[:500]}"
                )

        name_a, name_b = Path(adapter_a).stem, Path(adapter_b).stem
        bc = inner.battle_control

        def _outcome(idx: int) -> str:
            """Winner + cause for the recorded arena, read at match end."""
            w = float(bc.win_signal[idx])
            if w > 0.5:
                who = f"{name_a} wins"
            elif w < -0.5:
                who = f"{name_b} wins"
            else:
                who = "draw"
            if bool(bc.end_cause_ko[idx]):
                cause = "KO"
            elif bool(bc.end_cause_oob[idx]):
                cause = "ring-out (points)"
            elif bool(bc.end_cause_points[idx]):
                cause = "points"
            else:
                cause = "decision"
            return f"{who} by {cause}"

        # The raw stream fed to ffmpeg has a FIXED frame size (set by the first
        # frame). Isaac can occasionally hand back a differently-sized frame
        # (e.g. a resolution ramp during a reset), and one off-size frame
        # desyncs the pipe and kills ffmpeg (broken pipe). Force every frame to
        # the first frame's H×W.
        target_hw = None

        def _fit(fr):
            nonlocal target_hw
            if target_hw is None:
                target_hw = fr.shape[:2]
                return fr
            if fr.shape[:2] == target_hw:
                return fr
            th, tw = target_hw
            h, w = fr.shape[:2]
            log.info("record: resizing off-size frame %sx%s -> %sx%s", h, w, th, tw)
            try:  # good resample if PIL is present
                from PIL import Image

                return np.asarray(
                    Image.fromarray(fr).resize((tw, th), Image.BILINEAR)
                )
            except Exception:  # dependency-free crop/pad fallback
                out = np.zeros((th, tw, fr.shape[2]), dtype=fr.dtype)
                out[: min(th, h), : min(tw, w)] = fr[: min(th, h), : min(tw, w)]
                return out

        # --- scoreboard overlay (champion on the left, opponent on the right,
        # each with an HP bar). Drawn onto every frame before it's encoded. ---
        ego_id, opp_id = match_index, match_index + n
        init_hp = max(float(getattr(bc.config, "initial_health", 1.0)), 1e-6)
        try:
            from PIL import Image, ImageDraw, ImageFont

            try:
                import matplotlib.font_manager as _fm

                _bold = _fm.findfont(_fm.FontProperties(family="DejaVu Sans", weight="bold"))
                _reg = _fm.findfont(_fm.FontProperties(family="DejaVu Sans"))
                _font = ImageFont.truetype(_bold, 18)
                _font_sm = ImageFont.truetype(_reg, 13)
            except Exception:
                _font = ImageFont.load_default()
                _font_sm = _font
            _pil_ok = True
        except Exception:
            _pil_ok = False

        def _hp_color(frac):
            if frac > 0.5:
                return (80, 200, 90)      # green
            if frac > 0.25:
                return (230, 200, 60)     # yellow
            return (220, 70, 60)          # red

        def _scoreboard(fr, hp_a, hp_b, bout_i):
            if not _pil_ok:
                return fr
            try:
                img = Image.fromarray(fr)
                d = ImageDraw.Draw(img)
                W = img.width
                strip_h = 46
                d.rectangle([0, 0, W, strip_h], fill=(15, 18, 24))
                barw, barh, pad, top = int(W * 0.30), 12, 12, 26
                fa = max(0.0, min(1.0, hp_a / init_hp))
                fb = max(0.0, min(1.0, hp_b / init_hp))
                # Left = champion (A)
                d.text((pad, 4), f"CHAMPION  {name_a}", font=_font_sm,
                       fill=(150, 190, 235))
                d.rectangle([pad, top, pad + barw, top + barh], outline=(90, 90, 90))
                d.rectangle([pad, top, pad + int(barw * fa), top + barh],
                            fill=_hp_color(fa))
                d.text((pad + barw + 8, top - 2), f"{fa*100:3.0f}%", font=_font_sm,
                       fill=(230, 230, 230))
                # Right = opponent (B)
                rx = W - pad - barw
                d.text((rx, 4), f"{name_b}  OPPONENT", font=_font_sm,
                       fill=(235, 175, 130))
                d.rectangle([rx, top, rx + barw, top + barh], outline=(90, 90, 90))
                d.rectangle([rx + barw - int(barw * fb), top, rx + barw, top + barh],
                            fill=_hp_color(fb))
                d.text((rx - 42, top - 2), f"{fb*100:3.0f}%", font=_font_sm,
                       fill=(230, 230, 230))
                # Center = bout counter
                label = f"BOUT {bout_i + 1}/{bouts}"
                tw = d.textlength(label, font=_font) if hasattr(d, "textlength") else 70
                d.text(((W - tw) / 2, 6), label, font=_font, fill=(235, 235, 235))
                return np.asarray(img)
            except Exception:
                return fr

        held_ego = None
        written = 0
        results = []
        black = None  # black separator frame, sized from the first real frame
        try:
            for bout in range(bouts):
                obs, _ = self.env.reset()
                held_ego = None
                ended_at = None
                for step in range(max_frames):
                    obs = self.agent.add_agent_info_to_obs(obs)
                    if held_ego is None or self._frame % self.action_hold == 0:
                        obs_td = self.agent.obs_dict_to_tensordict(obs)
                        with self._autocast():
                            out = model(obs_td)
                        key = (
                            "mean_action"
                            if self.deterministic and "mean_action" in out
                            else "action"
                        )
                        held_ego = out[key].float()
                    obs, _, dones, _, _ = self.env.step(held_ego)
                    self._frame += 1

                    frame = sim.grab_rgb_frame()
                    if frame is not None:
                        frame = _fit(frame)
                        if black is None:
                            black = np.zeros_like(frame)
                        frame = _scoreboard(
                            frame,
                            float(bc.health[ego_id]),
                            float(bc.health[opp_id]),
                            bout,
                        )
                        _append(frame)
                        written += 1

                    # Record the outcome the moment this arena's bout ends, then
                    # keep filming a short tail so the finish stays on screen.
                    if ended_at is None and bool(dones[match_index]):
                        ended_at = step
                        results.append(_outcome(match_index))
                    if ended_at is not None and step - ended_at >= tail_frames:
                        break

                if ended_at is None:
                    results.append(f"no decision (hit {max_frames}-frame cap)")
                # Short black separator between bouts (not after the last).
                if black is not None and bout < bouts - 1:
                    for _ in range(max(1, fps // 3)):
                        _append(black)
                        written += 1
        finally:
            _close_writer()

        if written == 0:
            raise RuntimeError(
                "recorded 0 frames — the RGB annotator returned no data "
                "(offscreen rendering not active?)"
            )
        summary = "; ".join(f"bout {i + 1}: {r}" for i, r in enumerate(results))
        log.info(
            "Recorded %s vs %s -> %s (%d bouts, %d frames @ %d fps) | %s",
            name_a,
            name_b,
            out_path,
            len(results),
            written,
            fps,
            summary,
        )
        return {"path": out_path, "results": results, "summary": summary}

    def run_round_robin(
        self,
        adapters: List[str],
        matches_per_pairing: int,
        elo_k: float = 32.0,
        initial_rating: float = 1000.0,
    ) -> TournamentReport:
        """Full round-robin over the adapter list, both colors per pairing."""
        names = [Path(a).stem for a in adapters]
        report = TournamentReport(adapters=list(adapters))
        report.ratings = {n: initial_rating for n in names}
        report.head_to_head = {n: {} for n in names}

        for (i, a), (j, b) in itertools.combinations(enumerate(adapters), 2):
            half = max(1, matches_per_pairing // 2)
            res_ab = self.run_pairing(a, b, half)
            res_ba = self.run_pairing(b, a, matches_per_pairing - half)
            combined = PairingResult(
                wins_a=res_ab.wins_a + res_ba.wins_b,
                wins_b=res_ab.wins_b + res_ba.wins_a,
                draws=res_ab.draws + res_ba.draws,
            )
            na, nb = names[i], names[j]
            report.head_to_head[na][nb] = {
                "wins": combined.wins_a,
                "losses": combined.wins_b,
                "draws": combined.draws,
            }
            report.head_to_head[nb][na] = {
                "wins": combined.wins_b,
                "losses": combined.wins_a,
                "draws": combined.draws,
            }
            # Sequential Elo updates, one per match
            for outcome in (
                [1] * combined.wins_a + [-1] * combined.wins_b + [0] * combined.draws
            ):
                score = 1.0 if outcome == 1 else (0.0 if outcome == -1 else 0.5)
                report.ratings[na], report.ratings[nb] = elo_update(
                    report.ratings[na], report.ratings[nb], score, k=elo_k
                )

        for name, rating in report.ladder():
            log.info("  %-40s Elo %.0f", name, rating)
        return report

    def regression_gate(
        self,
        candidate: str,
        previous: str,
        matches: int = 64,
        threshold: float = 0.55,
    ) -> bool:
        """Admission check: candidate must score >= threshold vs previous.

        Uses clean evaluation matches, decoupling the gate from noisy
        training-time counters (a correctness fix over IsaacLabASE).
        """
        half = max(1, matches // 2)
        res_fwd = self.run_pairing(candidate, previous, half)
        res_rev = self.run_pairing(previous, candidate, matches - half)
        combined = PairingResult(
            wins_a=res_fwd.wins_a + res_rev.wins_b,
            wins_b=res_fwd.wins_b + res_rev.wins_a,
            draws=res_fwd.draws + res_rev.draws,
        )
        score = combined.score_a()
        passed = score >= threshold
        log.info(
            "Regression gate: %s vs %s score %.3f (threshold %.2f) -> %s",
            Path(candidate).stem,
            Path(previous).stem,
            score,
            threshold,
            "PASS" if passed else "FAIL",
        )
        return passed


__all__ = ["BattleTournament", "PairingResult", "TournamentReport"]
