# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Warm battle-recording web app.

Launches the battle stack ONCE (frozen prior + league adapters + a 2-env
offscreen-rendering IsaacSim) and serves a tiny browser page with a bout-count
field and a Record button. Each click records N whole bouts of the current
champion vs the earliest snapshot as a real IsaacSim render (see
``BattleTournament.record_pairing``) and plays the clip inline — no chat
round-trip, no per-click Isaac startup.

Kit must run on the main thread, so the HTTP server runs on a background
thread and hands record jobs to the main thread through a queue.

Run inside the training container (host networking, so the port is reachable
at http://localhost:PORT on the box)::

    python protomotions/battle_record_server.py \\
        --resolved-configs results/soma_battle_league_v3/resolved_configs_inference.pt \\
        --port 8080

Then open http://localhost:8080 in a browser on the box.
"""

import argparse
import glob
import json
import logging
import queue
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_parser():
    p = argparse.ArgumentParser(description="Battle recording web app")
    p.add_argument(
        "--resolved-configs",
        required=True,
        help="Path to the league run's resolved_configs_inference.pt",
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="results/<run> dir to scan for league snapshots "
        "(default: the resolved-configs' parent).",
    )
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--simulator", default="isaaclab")
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Greedy actions (default on for clean, repeatable exhibitions).",
    )
    p.add_argument(
        "--out-dir",
        default="output/fight_videos",
        help="Where recorded mp4s are written and served from.",
    )
    return p


args = create_parser().parse_args()

# isaacgym/isaaclab must be imported before torch
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402


RUN_DIR = Path(args.run_dir) if args.run_dir else Path(args.resolved_configs).parent
LEAGUE_GLOB = str(RUN_DIR / "lightning_logs" / "*" / "league" / "policy_*.ckpt")
OUT_DIR = Path(args.out_dir)


def league_snapshots():
    """All league snapshots, oldest first (mtime order)."""
    paths = sorted(glob.glob(LEAGUE_GLOB), key=lambda p: Path(p).stat().st_mtime)
    return paths


# ---------------------------------------------------------------------------
# HTTP server (background thread) -> job queue -> main thread does the Isaac work
# ---------------------------------------------------------------------------
_jobs: "queue.Queue" = queue.Queue()


class _Job:
    def __init__(self, bouts):
        self.bouts = bouts
        self.done = threading.Event()
        self.result = None  # dict on success, or {"error": ...}


def _page_html(port):
    snaps = league_snapshots()
    n = len(snaps)
    latest = Path(snaps[-1]).stem if snaps else "?"
    earliest = Path(snaps[0]).stem if snaps else "?"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SOMA Fight Club — recorder</title>
<style>
  :root{{--ground:#0e1116;--panel:#161b22;--hair:#28303c;--ink:#dbe2ea;--muted:#8b95a3;--ring:#e0402f}}
  *{{box-sizing:border-box}}
  body{{margin:0;min-height:100vh;background:radial-gradient(120% 90% at 50% -10%,#1a222d 0%,var(--ground) 60%);
       color:var(--ink);font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
       display:flex;flex-direction:column;align-items:center;gap:1.2rem;padding:2.4rem 1.25rem 3rem}}
  .eyebrow{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.72rem;letter-spacing:.22em;
           text-transform:uppercase;color:var(--ring);margin:0}}
  h1{{margin:.1rem 0 0;font-size:clamp(1.3rem,3.4vw,1.9rem);font-weight:750;letter-spacing:-.01em}}
  .panel{{display:flex;gap:.75rem;align-items:center;flex-wrap:wrap;justify-content:center;
         background:var(--panel);border:1px solid var(--hair);border-radius:10px;padding:.9rem 1.1rem}}
  label{{font-size:.85rem;color:var(--muted)}}
  input[type=number]{{width:4.5rem;background:#0b0e13;color:var(--ink);border:1px solid var(--hair);
       border-radius:7px;padding:.45rem .5rem;font-size:1rem;font-variant-numeric:tabular-nums}}
  button{{background:var(--ring);color:#fff;border:0;border-radius:7px;padding:.55rem 1.1rem;
         font-size:.95rem;font-weight:650;cursor:pointer}}
  button:disabled{{opacity:.5;cursor:progress}}
  a.dl{{display:none;text-decoration:none;background:transparent;color:var(--ink);
        border:1px solid var(--hair);border-radius:7px;padding:.55rem 1.1rem;
        font-size:.95rem;font-weight:600}}
  a.dl.ready{{display:inline-block}}
  .frame{{width:min(100%,900px);border:1px solid var(--hair);border-radius:10px;overflow:hidden;
         background:#000;box-shadow:0 24px 60px -28px rgba(0,0,0,.8)}}
  video{{display:block;width:100%;height:auto}}
  .status{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.8rem;color:var(--muted);
          min-height:1.2em;text-align:center;max-width:900px}}
  .status b{{color:var(--ink)}}
</style></head><body>
  <div style="text-align:center">
    <p class="eyebrow">SOMA Fight Club · recorder</p>
    <h1>{latest} &nbsp;vs&nbsp; {earliest}</h1>
  </div>
  <div class="panel">
    <label for="bouts">Bouts</label>
    <input id="bouts" type="number" min="1" max="10" value="3">
    <button id="rec" onclick="record()">Record</button>
    <a id="dl" class="dl" download>Save video</a>
    <span class="status" id="pool">{n} snapshots in pool</span>
  </div>
  <div class="status" id="status">Pick a bout count and hit Record. Champion (latest) vs rookie (earliest).</div>
  <div class="frame"><video id="vid" controls autoplay loop muted playsinline></video></div>
<script>
async function record() {{
  const btn = document.getElementById('rec'), st = document.getElementById('status');
  const bouts = Math.max(1, Math.min(10, parseInt(document.getElementById('bouts').value)||3));
  btn.disabled = true;
  st.textContent = 'Recording ' + bouts + ' bout(s)… (a few seconds per bout)';
  const t0 = Date.now();
  try {{
    const r = await fetch('/record', {{method:'POST', headers:{{'Content-Type':'application/json'}},
                                       body: JSON.stringify({{bouts}})}});
    const j = await r.json();
    if (j.error) {{ st.textContent = 'Error: ' + j.error; }}
    else {{
      const secs = ((Date.now()-t0)/1000).toFixed(0);
      st.innerHTML = '<b>' + j.matchup + '</b> — ' + j.summary + '  ·  rendered in ' + secs + 's';
      document.getElementById('vid').src = '/video/' + j.video + '?t=' + Date.now();
      const dl = document.getElementById('dl');
      dl.href = '/video/' + j.video + '?dl=1';
      dl.download = j.video;
      dl.classList.add('ready');
    }}
  }} catch (e) {{ st.textContent = 'Error: ' + e; }}
  btn.disabled = false;
}}
</script>
</body></html>"""


def _make_handler(port):
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet default access logging
            pass

        def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                self._send(200, _page_html(port).encode("utf-8"))
                return
            if self.path.startswith("/video/"):
                rest = self.path[len("/video/"):]
                name = rest.split("?", 1)[0]
                query = rest.split("?", 1)[1] if "?" in rest else ""
                fpath = OUT_DIR / Path(name).name
                if not fpath.exists():
                    self._send(404, b"not found", "text/plain")
                    return
                data = fpath.read_bytes()
                extra = {"Accept-Ranges": "none"}
                if "dl=1" in query:  # force a Save dialog with the filename
                    extra["Content-Disposition"] = (
                        f'attachment; filename="{Path(name).name}"'
                    )
                self._send(200, data, "video/mp4", extra)
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/record":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                bouts = int(payload.get("bouts", 3))
            except Exception:
                bouts = 3
            bouts = max(1, min(10, bouts))
            job = _Job(bouts)
            _jobs.put(job)
            job.done.wait()  # main thread fills job.result
            self._send(
                200,
                json.dumps(job.result).encode("utf-8"),
                "application/json",
            )

    return Handler


def main():
    resolved_path = Path(args.resolved_configs)
    assert resolved_path.exists(), f"Missing resolved configs: {resolved_path}"
    resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)

    robot_config = resolved["robot"]
    simulator_config = resolved["simulator"]
    terrain_config = resolved.get("terrain")
    scene_lib_config = resolved["scene_lib"]
    motion_lib_config = resolved["motion_lib"]
    env_config = resolved["env"]
    agent_config = resolved["agent"]

    if args.num_envs % 2 != 0:
        raise ValueError("--num-envs must be even (2 envs per match)")
    simulator_config.num_envs = args.num_envs

    # Offscreen rendering: launch Kit headless (no window) with enable_cameras,
    # but run the protomotions sim non-headless so render()/frame-grab fire.
    simulator_config.headless = False

    if hasattr(agent_config, "league"):
        agent_config.league.staleness_epochs = 10**9
        agent_config.league.gate_min_games = 10**9

    fabric = Fabric(
        **FabricConfig(
            accelerator="cpu" if args.simulator == "mujoco" else "gpu",
            devices=1,
            num_nodes=1,
            loggers=[],
            callbacks=[],
        ).as_kwargs()
    )
    fabric.launch()

    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher = AppLauncher(
            {"headless": True, "enable_cameras": True, "device": str(fabric.device)}
        )
        simulator_extra_params["simulation_app"] = app_launcher.app

    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.utils.component_builder import build_all_components

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=getattr(env_config, "save_dir", None),
        **simulator_extra_params,
    )

    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(
        config=agent_config, env=env, fabric=fabric, root_dir=resolved_path.parent
    )
    agent.setup()

    from protomotions.agents.league.tournament import BattleTournament

    tournament = BattleTournament(
        agent,
        deterministic=args.deterministic,
        sampling_mode="nucleus",  # fast decode; viewing only
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Start the HTTP server on a background thread.
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), _make_handler(args.port))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    log.info("Recorder ready — open http://localhost:%d on this box", args.port)

    # Main thread: service record jobs (all Isaac/Kit work stays here).
    while True:
        try:
            job: _Job = _jobs.get(timeout=1.0)
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break
        try:
            snaps = league_snapshots()
            if not snaps:
                raise RuntimeError(f"no league snapshots under {LEAGUE_GLOB}")
            latest, earliest = snaps[-1], snaps[0]
            ts = int(time.time())
            name = f"rec_{Path(latest).stem}_vs_{Path(earliest).stem}_{ts}.mp4"
            out_path = str(OUT_DIR / name)
            log.info("Recording %d bout(s): %s vs %s",
                     job.bouts, Path(latest).stem, Path(earliest).stem)
            rec = tournament.record_pairing(
                latest, earliest, out_path, bouts=job.bouts
            )
            job.result = {
                "video": name,
                "matchup": f"{Path(latest).stem} vs {Path(earliest).stem}",
                "summary": rec.get("summary", f"{job.bouts} bout(s) recorded"),
            }
        except Exception as exc:  # keep the server alive on a bad record
            log.exception("record failed")
            job.result = {"error": str(exc)}
        finally:
            job.done.set()

    # Reached only on Ctrl-C: hard-exit so Kit's teardown can't hang the
    # process as a GPU-memory-holding zombie (sim.close() itself hangs, so we
    # skip it — the OS reclaims GPU memory on process death).
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
