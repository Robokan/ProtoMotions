# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Measurement hook for probe_impulse_vs_ke.py (imported, not run).

Wraps BattleControl so that on every simulator step we accumulate the
opponent-attributable contact impulse per damage body, and whenever the
existing hit FSM fires an event we log that event's kinetic energy (the
current damage quantity) beside the impulse measured over the following
window. Nothing in the damage path is modified.
"""
from __future__ import annotations

import os

import torch

_STATE = {}


def install() -> None:
    """Monkeypatch BattleControl.step to measure alongside it."""
    import protomotions.envs.battle.control as control_mod

    Control = control_mod.BattleControl
    if getattr(Control, "_impulse_probe_installed", False):
        return
    orig = Control.step
    window_ms = float(os.environ.get("PROBE_WINDOW_MS", "80"))
    out_path = os.environ.get("PROBE_OUT", "data/impulse_vs_ke_probe.txt")

    def _opponent_forces(self):
        """net contact force minus the ground/object-attributed part.

        Every contact sensor is configured with filter_prim_paths_expr
        (ground + objects), so force_matrix_w holds exactly what those
        contribute; the residual is the opponent.
        """
        sim = self.env.simulator
        smap = getattr(sim, "_contact_sensor_map", None)
        if not smap:
            return None
        names = list(sim._robot.data.body_names)
        out = torch.zeros(sim.num_envs, len(names), 3, device=sim.device)
        for bi, bn in enumerate(names):
            sensor = smap.get(bn)
            if sensor is None:
                continue
            net = sensor.data.net_forces_w[:, 0, :]
            fm = getattr(sensor.data, "force_matrix_w", None)
            if fm is not None:
                ground = fm[:, 0].sum(dim=1)
                out[:, bi, :] = net - ground
            else:
                out[:, bi, :] = net
            if not _STATE.get("shape_logged"):
                _STATE["shape_logged"] = True
                print(f"[probe] sensor {bn}: net{tuple(net.shape)} "
                      f"fm={'None' if fm is None else tuple(fm.shape)} "
                      f"|net|={float(net.norm(dim=-1).max()):.2f}", flush=True)
        return out

    def patched(self, *a, **kw):
        ret = orig(self, *a, **kw)
        try:
            st = _STATE.setdefault("s", {
                "accum": None, "left": None, "rows": [], "steps": 0})
            hs = self.hit_state
            dt = float(getattr(hs, "dt", 1.0 / 30.0))
            f_opp = _opponent_forces(self)
            if f_opp is None:
                return ret
            # per-damage-body opponent force magnitude
            d_ids = hs.damage_body_ids                      # [2N, D]
            mag = f_opp.norm(dim=-1).gather(1, d_ids)       # [2N, D]
            if st["accum"] is None:
                st["accum"] = torch.zeros_like(mag)
                st["left"] = torch.zeros_like(mag)
            # open a window wherever the FSM just fired an event
            ke = getattr(hs, "last_event_ke", None)
            spd = getattr(hs, "last_event_speed", None)
            if ke is None:
                return ret
            fired = ke > 0
            steps = max(1, int(round(window_ms / 1000.0 / dt)))
            st["left"] = torch.where(
                fired, torch.full_like(st["left"], float(steps)), st["left"])
            st["accum"] = torch.where(
                fired, torch.zeros_like(st["accum"]), st["accum"])
            open_w = st["left"] > 0
            st["accum"] = st["accum"] + mag * dt * open_w
            # record when a window closes
            closing = open_w & (st["left"] <= 1)
            if bool(closing.any()):
                idx = closing.nonzero(as_tuple=False)
                for e, dq in idx.tolist():
                    st["rows"].append((
                        float(st["accum"][e, dq]),            # N.s
                        float(_STATE.get("ke_at", {}).get((e, dq), 0.0)),
                        float(_STATE.get("sp_at", {}).get((e, dq), 0.0)),
                    ))
            # remember the KE/speed at onset for pairing
            if bool(fired.any()) and _STATE.get("ev_dbg", 0) < 3:
                _STATE["ev_dbg"] = _STATE.get("ev_dbg", 0) + 1
                e, dq = fired.nonzero(as_tuple=False).tolist()[0]
                bi = int(d_ids[e, dq])
                sim = self.env.simulator
                bn = list(sim._robot.data.body_names)[bi]
                sen = sim._contact_sensor_map.get(bn)
                nf = (sen.data.net_forces_w[e, 0].norm().item()
                      if sen is not None else float("nan"))
                fmv = getattr(sen.data, "force_matrix_w", None) if sen else None
                gf = (fmv[e, 0].sum(0).norm().item()
                      if fmv is not None else float("nan"))
                print(f"[probe] event body={bn} sensor={'yes' if sen else 'NO'} "
                      f"|net|={nf:.2f} |ground|={gf:.2f} "
                      f"|resid|={float(mag[e, dq]):.2f} ke={float(ke[e, dq]):.3f}",
                      flush=True)
            if bool(fired.any()):
                kd = _STATE.setdefault("ke_at", {})
                sd = _STATE.setdefault("sp_at", {})
                for e, dq in fired.nonzero(as_tuple=False).tolist():
                    kd[(e, dq)] = float(ke[e, dq])
                    sd[(e, dq)] = float(spd[e, dq]) if spd is not None else 0.0
            st["left"] = (st["left"] - 1).clamp_min(0)
            st["steps"] += 1
            if st["steps"] % 100 == 0:
                print(f"[probe] step {st['steps']} events={len(st['rows'])}", flush=True)
                _dump(st, out_path, window_ms)
        except Exception as exc:
            if not _STATE.get("warned"):
                _STATE["warned"] = True
                import traceback; traceback.print_exc()
        return ret

    def _dump(st, path, window_ms):
        rows = st["rows"]
        if not rows:
            return
        import statistics as S
        imp = [r[0] for r in rows if r[0] > 0]
        ke = [r[1] for r in rows if r[0] > 0]
        spd = [r[2] for r in rows if r[0] > 0]
        if not imp:
            return
        lines = [
            f"hit events measured: {len(imp)}   window {window_ms:.0f} ms",
            "",
            f"{'quantity':<22}{'median':>12}{'mean':>12}{'p90':>12}{'max':>12}",
        ]

        def row(name, v):
            v2 = sorted(v)
            p90 = v2[int(0.9 * (len(v2) - 1))]
            return (f"{name:<22}{S.median(v):12.3f}{S.fmean(v):12.3f}"
                    f"{p90:12.3f}{max(v):12.3f}")
        lines.append(row("impulse  [N.s]", imp))
        lines.append(row("current KE  [J]", ke))
        lines.append(row("closing speed [m/s]", spd))
        ratio = [i / k for i, k in zip(imp, ke) if k > 1e-6]
        if ratio:
            lines.append("")
            lines.append(row("impulse / KE", ratio))
            lines.append("")
            lines.append("A damage_to_health for impulse that keeps today's "
                         "median hit unchanged:")
            lines.append(f"  0.05 HP/J x median KE {S.median(ke):.2f} J = "
                         f"{0.05*S.median(ke):.3f} HP")
            lines.append(f"  -> impulse HP/(N.s) = {0.05*S.median(ke)/S.median(imp):.4f}")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    Control.step = patched
    Control._impulse_probe_installed = True
    print("[probe] impulse measurement installed", flush=True)
