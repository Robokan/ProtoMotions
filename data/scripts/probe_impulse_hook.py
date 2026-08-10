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


def _install_pelvis_sensor() -> None:
    """Extend the replayed config's contact sensors to the pelvis damage row.

    The probe replays a RESOLVED config from an old training run, and those
    runs sensored hands/feet/head/torso only — every robot's pelvis damage
    body (Hips / Waist / LINK_BASE / RigPelvis: always the root) reads a
    constant 0 N, so pelvis hits can neither score in the FSM nor be
    measured here. Wrapping SceneCfg.__init__ appends the root body (plus
    Waist/Spine1/Spine2 where they exist) to contact_bodies before the
    sensors are built. The env sizes contact_body_ids from the same mutated
    config object, so both sides stay consistent. Policy obs are unaffected
    (these experiments run observe_contacts=False).

    TIMING: install() runs before Isaac Sim boots, and the scene module
    transitively imports pxr, which only exists once the SimulationApp is
    up — importing it here crashes with ModuleNotFoundError. isaaclab.app
    is the one isaaclab module that IS importable pre-boot (it is the
    bootstrap), so the scene patch is deferred to just after
    AppLauncher.__init__ finishes.
    """
    from isaaclab.app import AppLauncher

    if getattr(AppLauncher, "_probe_pelvis_deferred", False):
        return
    orig_launch = AppLauncher.__init__

    def launch_then_patch(self, *args, **kwargs):
        orig_launch(self, *args, **kwargs)
        _apply_scene_patch()

    AppLauncher.__init__ = launch_then_patch
    AppLauncher._probe_pelvis_deferred = True


def _apply_scene_patch() -> None:
    """Wrap SceneCfg.__init__ — only callable once the app is running.

    The idempotence marker lives on the MODULE, never on SceneCfg:
    InteractiveScene._add_entities_from_cfg iterates every attribute of the
    scene config as an asset definition, and a stray `_probe_pelvis_installed
    = True` on the class dies with "Unknown asset config type".
    """
    from protomotions.simulator.isaaclab.utils import scene as scene_mod

    if getattr(scene_mod, "_probe_pelvis_installed", False):
        return
    orig = scene_mod.SceneCfg.__init__

    def patched(self, config, robot_config, *args, **kwargs):
        cb = getattr(robot_config, "contact_bodies", None)
        if cb is not None:
            body_names = list(robot_config.kinematic_info.body_names)
            # Pelvis damage row per known battle robot: usually the root
            # body, EXCEPT atlas, whose root is Hip while its damage row is
            # Waist (so "root" alone missed it). Production experiments now
            # derive this from battle_table_kwargs; the probe replays OLD
            # resolved configs where the robot name is not at hand, so it
            # enumerates the known pelvis/mid-torso rows instead.
            extra = [body_names[0]]
            extra += [b for b in ("Waist", "Spine1", "Spine2")
                      if b in body_names]
            added = [b for b in extra if b not in cb]
            if added:
                cb.extend(added)
                print(f"[probe] added contact sensors for: {added}", flush=True)
        return orig(self, config, robot_config, *args, **kwargs)

    scene_mod.SceneCfg.__init__ = patched
    scene_mod._probe_pelvis_installed = True


def install() -> None:
    """Monkeypatch BattleControl.step to measure alongside it."""
    import protomotions.envs.battle.control as control_mod

    _install_pelvis_sensor()

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

        ORDER MATTERS: hit_state's damage_body_ids index the COMMON body
        order (robot_config.kinematic_info.body_names), not the simulator's.
        The first version of this hook built the tensor in simulator order
        and gathered it with common-order ids — wrong columns — and its
        event printout mapped common ids through simulator names, which is
        how a real Head/Chest hit got reported as "LeftShin, sensor=NO" and
        a false "damage bodies have no sensors" finding was born. The
        sensor map is keyed by NAME, so building in common order is enough.
        """
        sim = self.env.simulator
        smap = getattr(sim, "_contact_sensor_map", None)
        if not smap:
            return None
        names = list(self.env.robot_config.kinematic_info.body_names)  # COMMON
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
                # damage_body_ids are COMMON-order — resolve the name in the
                # same order (see _opponent_forces docstring for the bug this
                # fixes).
                bn = list(self.env.robot_config.kinematic_info.body_names)[bi]
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
                # Fight telemetry: distinguishes "fighting but no qualifying
                # hits" from "fallen/flailing and never engaging" (the two
                # explanations for events=0). Rows [N:2N] are the partners
                # of [0:N] in the paired-env layout.
                try:
                    rs = self.env.simulator.get_robot_state()
                    root = rs.root_pos
                    n2 = root.shape[0] // 2
                    pair_d = (root[:n2, :2] - root[n2:, :2]).norm(dim=-1)
                    print(
                        f"[probe] step {st['steps']} events={len(st['rows'])} "
                        f"| root z mean {float(root[:, 2].mean()):.2f} "
                        f"min {float(root[:, 2].min()):.2f} "
                        f"| pair dist mean {float(pair_d.mean()):.2f} "
                        f"min {float(pair_d.min()):.2f} m",
                        flush=True,
                    )
                except Exception:
                    print(f"[probe] step {st['steps']} events={len(st['rows'])}",
                          flush=True)
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
