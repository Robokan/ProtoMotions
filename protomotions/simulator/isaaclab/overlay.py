# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Skinned character overlay driven by a robot articulation.

Renders fights with a rigged character's skinned mesh instead of capsule
visuals: each rendered frame, the robot's rigid-body world rotations are
converted into the character skeleton's joint-local transforms and written
to a ``UsdSkelAnimation`` prim. Physics is untouched — the character is a
pure visual skin (SKINNED_OVERLAY_PLAN.md).

Retarget math (world-rotation delta):
    char_world_rot(j) = body_world_rot(map(j)) * offset(j)
    offset(j)         = body_rest_rot(map(j))^-1 * char_bind_world_rot(j)
computed once from the two rest poses; joint locals then come from the
parent chain. Unmapped bones (twists, toes, fingers, face) keep bind pose.

Import only inside a running Kit app (needs pxr).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, wxyz, batched [..., 4]."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _mat_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion wxyz (single)."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s]
        )
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(m[i, i] - m[j, j] - m[k, k] + 1.0, 1e-12)) * 2
    q = np.empty(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class SkinnedOverlay:
    """Drives one UsdSkel character from one fighter's rigid-body state.

    Args:
        stage: the USD stage (live Kit stage).
        character_usd: path to the character asset (SkelRoot with skeleton).
        prim_root: where to reference the character in (e.g. "/World/overlay0").
        body_names: robot body names in the order of the state tensors.
        body_rest_rot_wxyz: robot rest-pose world rotations [B, 4] (T-pose).
        joint_map: robot body name -> character bone leaf name.
        scale: uniform character scale (match robot height).
    """

    def __init__(
        self,
        stage,
        character_usd: str,
        prim_root: str,
        body_names: Sequence[str],
        body_rest_rot_wxyz: np.ndarray,
        joint_map: Dict[str, str],
        scale: float = 1.0,
        root_only: bool = False,
        drive_bodies: Sequence[str] = (),
        rest_rel: Dict[str, Sequence[float]] = None,
        tpose_pos: Dict[str, Sequence[float]] = None,
        limb_match: bool = False,
        body_parents: Dict[str, str] = None,
        fists: bool = False,
        head_shift: float = 0.0,
    ):
        from pxr import Usd, UsdGeom, UsdSkel, Sdf, Gf  # noqa: F401

        self._stage = stage
        self._root = prim_root
        self._body_index = {n: i for i, n in enumerate(body_names)}
        self._root_only = root_only

        # Absolute layer path: a relative reference resolves the asset's own
        # relative texture paths (@./textures/...@) against the wrong anchor
        # -> untextured character.
        from pathlib import Path as _Path
        character_usd = str(_Path(character_usd).resolve())

        ref = stage.DefinePrim(prim_root, "Xform")
        ref.GetReferences().AddReference(character_usd)
        if root_only:
            # Root-only mode: the character keeps its authored pose (no
            # SkelAnimation at all — the exact configuration known to render
            # upright) and the whole prim rigidly follows the robot root.
            # The referenced asset authors its own ops on this prim (its
            # up-axis correction: translate, rotateXYZ, scale) — leave those
            # untouched and layer suffixed ops OUTSIDE them (first in
            # xformOpOrder = applied last to points).
            xf = UsdGeom.Xformable(ref)
            existing = xf.GetOrderedXformOps()
            self._t_op = xf.AddTranslateOp(opSuffix="overlayRoot")
            self._o_op = xf.AddOrientOp(opSuffix="overlayRoot")
            self._t_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
            self._o_op.Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            ops = [self._t_op, self._o_op] + existing
            if scale != 1.0:
                s_op = xf.AddScaleOp(opSuffix="overlayRoot")
                s_op.Set(Gf.Vec3f(scale, scale, scale))
                ops.insert(2, s_op)
            xf.SetXformOpOrder(ops)
        elif scale != 1.0:
            UsdGeom.Xformable(ref).AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

        skel_prim = next(
            (p for p in Usd.PrimRange(ref) if p.IsA(UsdSkel.Skeleton)), None
        )
        if skel_prim is None:
            raise ValueError(f"No UsdSkel.Skeleton under {character_usd}")
        self._skel = UsdSkel.Skeleton(skel_prim)
        self._joints: List[str] = list(self._skel.GetJointsAttr().Get())
        self._parents = self._parent_indices(self._joints)

        # Bind-pose world transforms per joint (row-major Gf -> numpy).
        bind = self._skel.GetBindTransformsAttr().Get()
        self._bind_world = np.array(
            [[list(m[i]) for i in range(4)] for m in bind], dtype=np.float64
        )  # [J, 4, 4]
        rest = self._skel.GetRestTransformsAttr().Get()
        self._rest_local = np.array(
            [[list(m[i]) for i in range(4)] for m in rest], dtype=np.float64
        )

        if root_only:
            from pxr import UsdGeom as _UsdGeom, Usd as _Usd
            self._Gf = Gf
            # Robot root body = the one mapped to the character hip.
            root_body = next(
                (b for b in ("Hips", "Hip", "Pelvis") if b in self._body_index),
                None,
            ) or next(iter(joint_map))
            self._root_bi = self._body_index[root_body]
            # Motion-lib rotations are XYZW (poselib legacy), not wxyz —
            # misreading them was the root of every "upside down" symptom:
            # a yaw quat read in the wrong layout becomes a rotation about
            # an axis that tilts with the clip's facing (hence upright on
            # some clips, on-its-back on others). Read as xyzw, standing
            # SOMA hips are a plain world-aligned yaw, so the rest baseline
            # is simply identity.
            self._q_rest_root = np.array([1.0, 0.0, 0.0, 0.0])
            # Character hip position in prim space (ref prim is at identity
            # when this runs): row 3 of bind(joint->skel) @ skel local-to-world.
            hip_bone = joint_map.get(root_body, "Hip")
            leaf = [j.split("/")[-1] for j in self._joints]
            hip_ji = leaf.index(hip_bone)
            l2w = _UsdGeom.Xformable(skel_prim).ComputeLocalToWorldTransform(
                _Usd.TimeCode.Default()
            )
            m = np.array([[l2w[i][j] for j in range(4)] for i in range(4)])
            self._hip0 = scale * (self._bind_world[hip_ji] @ m)[3, :3]
            # Auto scale+align: 1-D similarity fit of the character's bind
            # joint positions (hip-relative, prim space) onto the robot's
            # T-pose body positions (hip-relative). Both T-poses are the same
            # configuration in the same frames, so no rotation is needed.
            self._t_fit = -self._hip0  # fallback: raw hip glue
            self._s_fit = scale
            if tpose_pos:
                C, Rt = [], []
                for body, bone in joint_map.items():
                    if body not in tpose_pos or bone not in leaf:
                        continue
                    if rest_rel and body in rest_rel:
                        # body binds away from the robot's T-pose (A-pose
                        # arms): its bind position is not a T-pose sample
                        continue
                    cj = (self._bind_world[leaf.index(bone)] @ m)[3, :3]
                    C.append(cj)
                    Rt.append(np.asarray(tpose_pos[body], dtype=np.float64))
                if len(C) >= 4:
                    C = np.stack(C); Rt = np.stack(Rt)
                    cbar, rbar = C.mean(0), Rt.mean(0)
                    Cc, Rc = C - cbar, Rt - rbar
                    s = float((Cc * Rc).sum() / max((Cc * Cc).sum(), 1e-9))
                    t = rbar - s * cbar
                    rms = float(np.sqrt(((s * C + t - Rt) ** 2).sum(1).mean()))
                    self._s_fit = s
                    self._t_fit = t
                    log.info(
                        "overlay auto-fit over %d joints: scale %.4f, "
                        "offset %s, rms %.3f m",
                        len(C), s, np.round(t, 3), rms,
                    )
                    print(f"[overlay] auto-fit: scale {s:.4f} offset "
                          f"{np.round(t,3)} rms {rms*100:.1f} cm", flush=True)
            if self._s_fit != 1.0:
                s_op = xf.AddScaleOp(opSuffix="overlayFit")
                s_op.Set(Gf.Vec3f(self._s_fit, self._s_fit, self._s_fit))
                xf.SetXformOpOrder([self._t_op, self._o_op, s_op] + existing)

            # Optional driven joints on top of the root-follow: each drive
            # body's rotation RELATIVE TO THE HIPS (minus its standing-pose
            # constant c_b from rest_rel) is applied to the character bone's
            # bind orientation, in skeleton space. Undriven joints hold rest.
            self._drive = []  # (joint_i, body_i, conj(c_b))
            if drive_bodies:
                rest_rel = rest_rel or {}
                for b in drive_bodies:
                    bi = self._body_index.get(b)
                    ji2 = leaf.index(joint_map[b]) if joint_map.get(b) in leaf else None
                    if bi is None or ji2 is None:
                        log.warning("overlay: cannot drive '%s' (missing)", b)
                        continue
                    c = np.asarray(rest_rel.get(b, (1.0, 0, 0, 0)), dtype=np.float64)
                    self._drive.append((ji2, bi, _quat_conj(c)))
                self._hip_ji = hip_ji
                # Skeleton-space <-> world rotation (the asset may carry a
                # rotated root above the skel, e.g. Blender Y-up exports).
                # Needed to conjugate world-space drive deltas into skel axes.
                _l2w = _UsdGeom.Xformable(skel_prim).ComputeLocalToWorldTransform(
                    _Usd.TimeCode.Default())
                _m = np.array([[_l2w[i][j] for j in range(4)] for i in range(4)])
                _rot3 = _m[:3, :3]
                _rot3 = _rot3 / (np.cbrt(abs(np.linalg.det(_rot3))) or 1.0)
                self._world_to_skel_q = _quat_conj(_mat_to_quat_wxyz(_rot3.T))
                log.info("overlay root_only: world_to_skel_q=%s",
                         np.round(self._world_to_skel_q, 4))
                # Bind-pose world (=skel-space) quats per joint, and rest
                # locals for the undriven remainder.
                self._bind_q = np.stack(
                    [_mat_to_quat_wxyz(self._bind_world[j, :3, :3].T)
                     for j in range(len(self._joints))]
                )
                self._rest_q = np.stack(
                    [_mat_to_quat_wxyz(self._rest_local[j, :3, :3].T)
                     for j in range(len(self._joints))]
                )
                anim_path = skel_prim.GetPath().AppendChild("OverlayAnim")
                self._anim = UsdSkel.Animation.Define(stage, anim_path)
                self._anim.GetJointsAttr().Set(self._joints)
                for holder in (skel_prim, ):
                    UsdSkel.BindingAPI.Apply(holder).GetAnimationSourceRel(
                    ).SetTargets([anim_path])
                skelroot = skel_prim.GetParent()
                while skelroot and not skelroot.IsA(UsdSkel.Root):
                    skelroot = skelroot.GetParent()
                if skelroot and skelroot.IsA(UsdSkel.Root):
                    UsdSkel.BindingAPI.Apply(skelroot).GetAnimationSourceRel(
                    ).SetTargets([anim_path])
                # Constant channels: rest translations (optionally rescaled
                # per segment so joint-to-joint distances match the robot's),
                # unit scales.
                trans = [self._rest_local[j, 3, :3].copy()
                         for j in range(len(self._joints))]
                if limb_match and tpose_pos and body_parents:
                    # Full congruence: place each mapped joint at the robot's
                    # T-pose offset from its mapped parent (direction AND
                    # length), expressed in the parent joint's bind frame.
                    # This also fixes chain ANCHORS (e.g. the CC clavicle
                    # sits cm's away from SOMA's shoulder), not just lengths.
                    for body, pb in body_parents.items():
                        bone = joint_map.get(body)
                        pbone = joint_map.get(pb)
                        if not bone or not pbone or bone not in leaf \
                                or pbone not in leaf or body not in tpose_pos \
                                or pb not in tpose_pos:
                            continue
                        j = leaf.index(bone)
                        pj = leaf.index(pbone)
                        d_world = (
                            np.subtract(tpose_pos[body], tpose_pos[pb])
                            / self._s_fit
                        )
                        if body == "Head" and head_shift:
                            # cosmetic mesh alignment: character faces -Y in
                            # bind, so forward = -Y (positive = forward).
                            d_world = d_world + np.array(
                                [0.0, -head_shift / self._s_fit, 0.0])
                        dp = self._parents[j]
                        if dp != pj:
                            # Intermediate rig bones between the mapped pair
                            # (e.g. Hip -> Pelvis -> Thigh): subtract their
                            # bind offsets so the total offset from the
                            # mapped parent still matches the robot.
                            inter = (
                                self._bind_world[dp, 3, :3]
                                - self._bind_world[pj, 3, :3]
                            )
                            d_world = d_world - inter
                        # If the PARENT binds away from robot T-pose (A-pose
                        # arms), the runtime delta at robot-T is conj(c_pb):
                        # pre-rotate the target offset by c_pb so the child
                        # lands at the robot's T offset when the robot is in
                        # T-pose.
                        if rest_rel and pb in rest_rel:
                            c_pb = np.asarray(rest_rel[pb], dtype=np.float64)
                            d_world = _quat_to_mat(c_pb) @ d_world
                        # skel-space offset -> direct parent joint local frame
                        # (row-major bind rot: local = rows @ world)
                        d_local = self._bind_world[dp, :3, :3] @ d_world
                        moved = float(np.linalg.norm(d_local - trans[j]))
                        trans[j] = d_local
                        if moved > 0.01:
                            log.info("overlay limb-match: %s (%s) moved "
                                     "%.1f cm", body, bone, moved * 100)
                self._anim.GetTranslationsAttr().Set([
                    Gf.Vec3f(*[float(v) for v in t3]) for t3 in trans
                ])
                # Static fist pose for the (unmapped) finger joints: curl
                # each phalanx about finger_dir x palm_normal in its own
                # bind frame. Palm normal = hand local X (faces down in
                # bind), finger direction = joint local Y.
                self._static_L = {}
                if fists:
                    # Rig-agnostic fists, derived from bind GEOMETRY (no
                    # local-axis assumptions): finger directions come from
                    # joint positions, the palm normal from the finger-spread
                    # plane, and the palm side from where the thumb sits.
                    curl = {1: 80.0, 2: 90.0, 3: 55.0}
                    thumb = {1: 28.0, 2: 35.0, 3: 30.0}
                    DIGITS = ("thumb", "index", "middle", "mid", "ring",
                              "pinky")

                    def _digit_joints(side_cc, side_ue):
                        """{digit: {1: ji, 2: ji, 3: ji}} for either naming."""
                        import re as _re
                        out = {}
                        for j, lf in enumerate(leaf):
                            m = _re.match(
                                rf"{side_cc}_(Thumb|Index|Mid|Ring|Pinky)"
                                rf"(\d)$", lf)
                            if not m:
                                m = _re.match(
                                    r"(thumb|index|middle|ring|pinky)_0(\d)_"
                                    rf"{side_ue}$", lf)
                            if not m:
                                continue
                            d = m.group(1).lower()
                            d = "mid" if d in ("mid", "middle") else d
                            out.setdefault(d, {})[int(m.group(2))] = j
                        return out

                    pos = self._bind_world[:, 3, :3]
                    total = 0
                    for side_cc, side_ue in (("L", "l"), ("R", "r")):
                        digits = _digit_joints(side_cc, side_ue)
                        if not digits or "mid" not in digits:
                            continue
                        mid = digits["mid"]
                        fmid = pos[mid[2]] - pos[mid[1]]
                        fmid /= max(np.linalg.norm(fmid), 1e-9)
                        spread = None
                        if "index" in digits and "pinky" in digits:
                            spread = (pos[digits["index"][1]]
                                      - pos[digits["pinky"][1]])
                        if spread is None:
                            continue
                        palm = np.cross(fmid, spread)
                        palm /= max(np.linalg.norm(palm), 1e-9)
                        if "thumb" in digits:
                            to_thumb = pos[digits["thumb"][1]] - pos[mid[1]]
                            if np.dot(palm, to_thumb) < 0:
                                palm = -palm
                        # Canonical LOCAL flexion hinge from the index finger:
                        # digits share local frame conventions on both rig
                        # families, so this same local axis is the anatomical
                        # hinge for the thumb's outer joints (they flex "like
                        # fingers" but in the thumb's own plane). Deriving the
                        # thumb axis from the palm normal instead bends it
                        # backward/sideways.
                        hinge_local = None
                        if "index" in digits and 1 in digits["index"]:
                            ij = digits["index"][1]
                            fi = pos[digits["index"].get(2, ij)] - pos[ij]
                            fi /= max(np.linalg.norm(fi), 1e-9)
                            axw = np.cross(fi, palm)
                            axw /= max(np.linalg.norm(axw), 1e-9)
                            hinge_local = self._bind_world[ij, :3, :3] @ axw
                        for d, chain in digits.items():
                            tbl = thumb if d == "thumb" else curl
                            for k, j in chain.items():
                                nxt = chain.get(k + 1)
                                if nxt is not None:
                                    fdir = pos[nxt] - pos[j]
                                else:
                                    fdir = pos[j] - pos[chain[k - 1]] \
                                        if (k - 1) in chain else fmid
                                fdir = fdir / max(np.linalg.norm(fdir), 1e-9)
                                if d == "thumb" and k == 1:
                                    # CMC base: sweep across the palm toward
                                    # the fingers, plus a light hinge flex.
                                    axw = palm.copy()
                                    if np.dot(np.cross(axw, fdir), fmid) < 0:
                                        axw = -axw
                                    ax_l = self._bind_world[j, :3, :3] @ axw
                                    h1 = np.radians(tbl[1]) / 2.0
                                    q_sw = np.array(
                                        [np.cos(h1), *(np.sin(h1) * ax_l)])
                                    hx_l = hinge_local
                                    if hx_l is None:
                                        hx = np.cross(fdir, palm)
                                        hx /= max(np.linalg.norm(hx), 1e-9)
                                        hx_l = (
                                            self._bind_world[j, :3, :3] @ hx)
                                    h2 = np.radians(12.0) / 2.0
                                    q_fl = np.array(
                                        [np.cos(h2), *(np.sin(h2) * hx_l)])
                                    self._static_L[j] = _quat_mul(
                                        self._rest_q[j],
                                        _quat_mul(q_sw, q_fl))
                                    total += 1
                                    continue
                                if d == "thumb" and hinge_local is not None:
                                    # Outer thumb joints: pure hinge on the
                                    # shared local flexion axis.
                                    ax_l = hinge_local
                                else:
                                    axw = np.cross(fdir, palm)
                                    n = np.linalg.norm(axw)
                                    if n < 1e-6:
                                        continue
                                    ax_l = (
                                        self._bind_world[j, :3, :3]
                                        @ (axw / n))
                                half = np.radians(tbl.get(k, 45.0)) / 2.0
                                qrot = np.array(
                                    [np.cos(half), *(np.sin(half) * ax_l)])
                                self._static_L[j] = _quat_mul(
                                    self._rest_q[j], qrot)
                                total += 1
                    log.info("overlay fists: curled %d finger joints",
                             len(self._static_L))
                self._anim.GetScalesAttr().Set(
                    [Gf.Vec3h(1.0, 1.0, 1.0)] * len(self._joints)
                )
            log.info(
                "SkinnedOverlay(root_only): robot '%s' -> hip bone '%s', "
                "hip0=%s, driving %d joints",
                root_body, hip_bone, np.round(self._hip0, 3), len(self._drive),
            )
            return

        # Per-joint driver: index of the robot body driving it (-1 = hold).
        leaf = [j.split("/")[-1] for j in self._joints]
        bone_to_joint = {l: i for i, l in enumerate(leaf)}
        self._driven = np.full(len(self._joints), -1, dtype=np.int64)
        for body, bone in joint_map.items():
            if body not in self._body_index:
                log.warning("overlay: robot body %s not in state tensors", body)
                continue
            ji = bone_to_joint.get(bone)
            if ji is None:
                log.warning("overlay: character bone %s not found", bone)
                continue
            self._driven[ji] = self._body_index[body]

        # Offsets are built AFTER the world->skel transform is known (see
        # _build_offsets): rest poses live in WORLD space, bind poses in
        # SKELETON space — the rest pose must be rotated into skeleton space
        # first or every joint inherits the asset's up-axis correction as a
        # constant error (symptom: whole character inverted, angles skewed).
        self._body_rest_rot = np.asarray(body_rest_rot_wxyz, dtype=np.float64)

        # SkelAnimation prim bound to the skeleton.
        anim_path = skel_prim.GetPath().AppendChild("OverlayAnim")
        self._anim = UsdSkel.Animation.Define(stage, anim_path)
        self._anim.GetJointsAttr().Set(self._joints)
        binding = UsdSkel.BindingAPI.Apply(skel_prim)
        binding.GetAnimationSourceRel().SetTargets([anim_path])
        # Some runtimes resolve the animation source at the SkelRoot level —
        # bind there as well (harmless duplication per UsdSkel inheritance).
        skelroot = skel_prim.GetParent()
        while skelroot and not skelroot.IsA(UsdSkel.Root):
            skelroot = skelroot.GetParent()
        if skelroot and skelroot.IsA(UsdSkel.Root):
            rb = UsdSkel.BindingAPI.Apply(skelroot.GetPrim() if hasattr(skelroot,'GetPrim') else skelroot)
            rb.GetAnimationSourceRel().SetTargets([anim_path])
            log.info("overlay: bound animation at SkelRoot %s", skelroot.GetPath())

        n = len(self._joints)
        self._scales = [Gf.Vec3h(1.0, 1.0, 1.0)] * n
        self._Gf = Gf

        # Character assets carry their own up-axis/unit correction above the
        # skeleton (Reallusion rigs are Y-up under a rotated root). Joint
        # animation lives in SKELETON space, so all world-space targets must
        # be pre-transformed by the inverse of the skeleton's local-to-world.
        from pxr import UsdGeom, Usd
        l2w = UsdGeom.Xformable(skel_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        m = np.array([[l2w[i][j] for j in range(4)] for i in range(4)])
        rot3 = m[:3, :3]
        self._skel_scale = float(np.cbrt(abs(np.linalg.det(rot3))) or 1.0)
        rot3n = rot3 / self._skel_scale
        self._world_to_skel_q = _quat_conj(_mat_to_quat_wxyz(rot3n.T))
        self._skel_origin = m[3, :3]
        self._world_to_skel_R = rot3n.T  # row-major transpose = inverse rot
        self._calibrated = False
        self._ref_prim = ref
        self._build_offsets()
        # DIAGNOSTIC (first-frame): dump the transform facts so frame bugs
        # are identified by evidence, not theory.
        np.set_printoptions(precision=3, suppress=True)
        print("overlay DIAG skel prim: ", skel_prim.GetPath())
        print("overlay DIAG skel l2w rot:\n", rot3n)
        print("overlay DIAG skel l2w scale: %.4f origin: %s",
                 self._skel_scale, m[3, :3], flush=True)
        hipji = next(i for i,d in enumerate(self._driven) if d >= 0)
        print("overlay DIAG hip bind rot (rows):\n%s",
                 self._bind_world[hipji, :3, :3])
        print("overlay DIAG hip rest_local rot:\n%s",
                 self._rest_local[hipji, :3, :3])
        # the reference prim's own accumulated transform
        refl2w = UsdGeom.Xformable(ref).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        print("overlay DIAG ref prim l2w:\n%s",
                 np.array([[refl2w[i][j] for j in range(4)] for i in range(3)]))
        log.info(
            "SkinnedOverlay: %d joints, %d driven by robot bodies",
            n, int((self._driven >= 0).sum()),
        )

    def _build_offsets(self) -> None:
        """offset(j) = (q_w2s * rest_world(body))^-1 * bind_skel(j) — both
        factors in SKELETON space. At robot rest the character then shows
        exactly its bind pose."""
        self._offset = np.tile(np.array([1.0, 0, 0, 0]), (len(self._joints), 1))
        for ji, bi in enumerate(self._driven):
            if bi < 0:
                continue
            bind_q = _mat_to_quat_wxyz(self._bind_world[ji, :3, :3].T)
            rest_skel = _quat_mul(self._world_to_skel_q, self._body_rest_rot[bi])
            self._offset[ji] = _quat_mul(_quat_conj(rest_skel), bind_q)

    @staticmethod
    def _parent_indices(joints: List[str]) -> np.ndarray:
        index = {j: i for i, j in enumerate(joints)}
        parents = np.full(len(joints), -1, dtype=np.int64)
        for i, j in enumerate(joints):
            parent = j.rsplit("/", 1)[0] if "/" in j else None
            if parent and parent in index:
                parents[i] = index[parent]
        return parents

    def _calibrate_scale(self, hips_world_z: float) -> None:
        """One-time: uniformly scale the WHOLE character (mesh + skeleton
        together, via the root prim's scale op) so its hip bind height
        matches the robot's hip height — size first, then rotate joints.
        Refreshes the cached world<->skel transform to include the scale."""
        from pxr import Usd, UsdGeom, Gf

        hip_ji = [i for i, d in enumerate(self._driven) if d >= 0][0]
        for i, d in enumerate(self._driven):
            if d >= 0 and (self._parents[i] < 0 or self._driven[self._parents[i]] < 0):
                hip_ji = i
                break
        bind_t = self._bind_world[hip_ji, 3, :3]
        # character hip position in WORLD (through the asset correction)
        l2w = UsdGeom.Xformable(self._skel.GetPrim()).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        m = np.array([[l2w[i][j] for j in range(4)] for i in range(4)])
        hip_world = bind_t @ m[:3, :3] + m[3, :3]
        char_h = float(abs(hip_world[2]))
        if char_h < 1e-4:
            log.warning("overlay: cannot calibrate scale (hip at ground)")
            self._calibrated = True
            return
        s = float(hips_world_z) / char_h
        xf = UsdGeom.Xformable(self._ref_prim)
        ops = xf.GetOrderedXformOps()
        scale_op = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeScale), None)
        if scale_op is None:
            scale_op = xf.AddScaleOp()
        scale_op.Set(Gf.Vec3f(s, s, s))
        # recompute cached transform with the scale baked in
        l2w = UsdGeom.Xformable(self._skel.GetPrim()).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        m = np.array([[l2w[i][j] for j in range(4)] for i in range(4)])
        rot3 = m[:3, :3]
        self._skel_scale = float(np.cbrt(abs(np.linalg.det(rot3))) or 1.0)
        rot3n = rot3 / self._skel_scale
        self._world_to_skel_q = _quat_conj(_mat_to_quat_wxyz(rot3n.T))
        self._skel_origin = m[3, :3]
        self._world_to_skel_R = rot3n.T
        self._calibrated = True
        log.info("overlay: auto-scale %.3f (char hip %.3f m -> robot hip %.3f m)",
                 s, char_h, hips_world_z)

    def sync(self, body_pos: torch.Tensor, body_rot_wxyz: torch.Tensor) -> None:
        """Write one frame. body_pos [B,3], body_rot [B,4] world, wxyz."""
        if self._root_only:
            Gf = self._Gf
            p = np.asarray(
                body_pos[self._root_bi].detach().cpu().numpy(), dtype=np.float64
            )
            q = np.asarray(
                body_rot_wxyz[self._root_bi].detach().cpu().numpy(),
                dtype=np.float64,
            )[[3, 0, 1, 2]]  # motion-lib rotations are xyzw -> reorder to wxyz
            # Rotation relative to the robot's rest orientation: at rest the
            # character shows its authored (upright) pose exactly.
            q_rel = _quat_mul(q, _quat_conj(self._q_rest_root))
            R = _quat_to_mat(q_rel)
            # Auto-fit offset (falls back to -hip0 = raw hip glue): places
            # the whole scaled character least-squares onto the robot.
            t = p + R @ self._t_fit
            self._t_op.Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
            self._o_op.Set(
                Gf.Quatf(
                    float(q_rel[0]),
                    Gf.Vec3f(float(q_rel[1]), float(q_rel[2]), float(q_rel[3])),
                )
            )
            if self._drive:
                # Skeleton space == robot-hip-aligned world frame (the prim
                # op above carries the hip rotation), so per-body deltas
                # relative to the hips apply as skel-space pre-rotations of
                # the bind pose. Chain: driven joints get absolute targets,
                # undriven joints follow their parent at rest.
                J = len(self._joints)
                drive_at = {ji: (bi, cc) for ji, bi, cc in self._drive}
                W = np.zeros((J, 4))
                L = self._rest_q.copy()
                for j in range(J):
                    pi = self._parents[j]
                    Wp = W[pi] if pi >= 0 else np.array([1.0, 0, 0, 0])
                    if j in drive_at:
                        bi, cc = drive_at[j]
                        qb = np.asarray(
                            body_rot_wxyz[bi].detach().cpu().numpy(),
                            dtype=np.float64,
                        )[[3, 0, 1, 2]]  # xyzw -> wxyz
                        # World-space hip-relative delta, conjugated into
                        # skeleton axes (no-op when the asset's skel root is
                        # unrotated, e.g. the Blender creature exports), then
                        # pre-multiplied onto the skel-space bind rotation.
                        # Verified exact (0.0 cm joint error) by the offline
                        # variant sweep once the skeleton units were fixed —
                        # the historical "mangled mesh" was never this math,
                        # it was the overlay asset's skeleton translations
                        # having been wrongly rescaled to meters while the
                        # mesh points are authored in cm.
                        delta = _quat_mul(_quat_mul(_quat_conj(q), qb), cc)
                        delta = _quat_mul(
                            _quat_mul(self._world_to_skel_q, delta),
                            _quat_conj(self._world_to_skel_q))
                        W[j] = _quat_mul(delta, self._bind_q[j])
                        L[j] = _quat_mul(_quat_conj(Wp), W[j])
                    else:
                        Lj = getattr(self, "_static_L", {}).get(j)
                        if Lj is not None:
                            L[j] = Lj
                        W[j] = _quat_mul(Wp, self._rest_q[j])
                import os as _osf
                if _osf.environ.get("OVERLAY_FREEZE"):
                    # Write-path probe: slam a 90-deg roll onto every DRIVEN
                    # joint's local. If the render doesn't kink, per-frame
                    # rotation writes never reach the skeleton in this mode.
                    _k = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])
                    for jf in drive_at:
                        L[jf] = _quat_mul(self._rest_q[jf], _k)
                self._anim.GetRotationsAttr().Set([
                    Gf.Quatf(float(L[j][0]),
                             Gf.Vec3f(float(L[j][1]), float(L[j][2]),
                                      float(L[j][3])))
                    for j in range(J)
                ])
                import os as _os2
                if _os2.environ.get("OVERLAY_FREEZE") and not hasattr(
                        self, "_bind_dbg"):
                    self._bind_dbg = True
                    _j0 = next(iter(drive_at))
                    _rb = self._anim.GetRotationsAttr().Get()
                    print(f"[ov-bind] anim prim: {self._anim.GetPrim().GetPath()}",
                          flush=True)
                    print(f"[ov-bind] wrote L[{_j0}]={np.round(L[_j0],3)} "
                          f"read-back={_rb[_j0] if _rb else None}", flush=True)
                    from pxr import UsdSkel as _USk
                    _skp = self._anim.GetPrim().GetParent()
                    _b = _USk.BindingAPI(_skp)
                    print(f"[ov-bind] skel prim {_skp.GetPath()} animSource "
                          f"targets: {_b.GetAnimationSourceRel().GetTargets()}",
                          flush=True)
                    _cache = _USk.Cache()
                    _sq = _cache.GetSkelQuery(_USk.Skeleton(_skp))
                    _aq = _sq.GetAnimQuery()
                    print(f"[ov-bind] skelq anim query: "
                          f"{_aq.GetPrim().GetPath() if _aq else 'NONE'}",
                          flush=True)
                self._d2n = getattr(self, "_d2n", 0) + 1
                if _os2.environ.get("OVERLAY_DIAG2") and self._d2n % 120 == 1:
                    def _eul(qq):
                        R = _quat_to_mat(qq)
                        return np.degrees([
                            np.arctan2(R[2, 1], R[2, 2]),
                            np.arcsin(np.clip(-R[2, 0], -1, 1)),
                            np.arctan2(R[1, 0], R[0, 0])]).round(1)
                    sw = _quat_conj(self._world_to_skel_q)
                    for jj, bi, _cc in self._drive[:3]:
                        qb = np.asarray(
                            body_rot_wxyz[bi].detach().cpu().numpy(),
                            dtype=np.float64)[[3, 0, 1, 2]]
                        # char world rot of this joint = prim_q * S * W[j]
                        cw = _quat_mul(q, _quat_mul(sw, W[jj]))
                        # expected: qb * S * bind (world delta on world bind)
                        ew = _quat_mul(qb, _quat_mul(sw, self._bind_q[jj]))
                        err = _quat_mul(_quat_conj(ew), cw)
                        print(f"[ov-diag2] {self._joints[jj].split('/')[-1]}: "
                              f"hip q eul {_eul(q)} qb eul {_eul(qb)} "
                              f"char-vs-expected err eul {_eul(err)}",
                              flush=True)
                # Diagnostic (once per ~4s): FK the skeleton we just wrote
                # and report where the character's hand joints actually land
                # vs the robot's hand bodies.
                import os as _os
                self._diag_n = getattr(self, "_diag_n", 0) + 1
                if _os.environ.get("OVERLAY_DIAG") and self._diag_n % 120 == 1:
                    tloc = getattr(self, "_trans_cache", None)
                    if tloc is None:
                        tloc = [np.array(v) for v in
                                self._anim.GetTranslationsAttr().Get()]
                        self._trans_cache = tloc
                    Pj = np.zeros((J, 3))
                    for j in range(J):
                        pi = self._parents[j]
                        if pi < 0:
                            Pj[j] = tloc[j]
                            continue
                        Rw = _quat_to_mat(W[pi])
                        Pj[j] = Pj[pi] + Rw @ tloc[j]
                    q_prim = q  # root quat (wxyz) applied by the prim op
                    Rp = _quat_to_mat(q_prim)
                    for hb, bone in (("LeftHand", None), ("RightHand", None)):
                        bi = self._body_index.get(hb)
                        ji2 = next((jj for jj, b2, _ in self._drive
                                    if b2 == bi), None)
                        if bi is None or ji2 is None:
                            continue
                        char_w = (np.asarray(
                            body_pos[self._root_bi].detach().cpu().numpy())
                            + Rp @ (self._t_fit + self._s_fit * Pj[ji2]))
                        rob_w = np.asarray(
                            body_pos[bi].detach().cpu().numpy())
                        print(f"[overlay-diag] {hb}: char-robot delta "
                              f"{np.round((char_w - rob_w) * 100, 1)} cm",
                              flush=True)
            return
        import os
        mode = os.environ.get("OVERLAY_TEST", "")
        if mode:
            # Anchor tests: stage "static" writes the pure rest pose (must
            # render the untouched bind-pose character -> pipeline OK);
            # "spinz" yaws the first driven joint 30 deg about world Z via
            # the delta formula (must YAW, not roll -> conventions OK).
            Gf = self._Gf
            self._t = getattr(self, "_t", 0) + 1
            translations, rotations = [], []
            hipji = next(i for i, d in enumerate(self._driven) if d >= 0)
            for ji in range(len(self._joints)):
                rl = self._rest_local[ji]
                lq = _mat_to_quat_wxyz(rl[:3, :3].T)
                lt = rl[3, :3]
                if mode == "spinz" and ji == hipji:
                    ang = np.radians(90.0) * np.sin(self._t / 15.0)  # fast unmissable sway
                    dz = np.array([np.cos(ang / 2), 0, 0, np.sin(ang / 2)])
                    dz = _quat_mul(
                        _quat_mul(self._world_to_skel_q, dz),
                        _quat_conj(self._world_to_skel_q))  # world -> skel axes
                    bind_q = _mat_to_quat_wxyz(self._bind_world[ji, :3, :3].T)
                    world_q = _quat_mul(dz, bind_q)
                    pi = self._parents[ji]
                    # parent assumed at rest (RL_BoneRoot)
                    if pi >= 0:
                        pq = _mat_to_quat_wxyz(self._rest_local[pi][:3, :3].T)
                        lq = _quat_mul(_quat_conj(pq), world_q)
                    else:
                        lq = world_q
                rotations.append(Gf.Quatf(float(lq[0]), float(lq[1]), float(lq[2]), float(lq[3])))
                translations.append(Gf.Vec3f(float(lt[0]), float(lt[1]), float(lt[2])))
            self._anim.GetTranslationsAttr().Set(translations)
            self._anim.GetRotationsAttr().Set(rotations)
            self._anim.GetScalesAttr().Set(self._scales)
            if self._t % 120 == 1:
                print(f"[overlay-test] mode={mode} write #{self._t} ok", flush=True)
            return
        Gf = self._Gf
        pos = body_pos.detach().cpu().numpy().astype(np.float64)
        rot = body_rot_wxyz.detach().cpu().numpy().astype(np.float64)
        # Auto-scale calibration removed: the static anchor test proved the
        # asset renders at correct size untouched — the calibrator was
        # measuring bind translations in the skeleton's internal units
        # against meters and shrinking the character ~100x.
        self._calibrated = True

        # World rotation per joint: driven -> body*offset; else parent-follow
        # keeps bind-relative local (computed below via rest_local fallback).
        world_q = np.zeros((len(self._joints), 4))
        world_t = np.zeros((len(self._joints), 3))
        translations = []
        rotations = []
        # Robot world -> character skeleton space (undoes the asset's up-axis
        # correction and any scale above the skeleton).
        rot_s = _quat_mul(np.broadcast_to(self._world_to_skel_q, rot.shape), rot)
        pos_s = (pos - self._skel_origin) @ self._world_to_skel_R.T / max(
            self._skel_scale, 1e-9
        )

        for ji in range(len(self._joints)):
            pi = self._parents[ji]
            bi = self._driven[ji]
            if bi >= 0:
                world_q[ji] = _quat_mul(rot_s[bi], self._offset[ji])
                world_t[ji] = pos_s[bi]
            elif pi >= 0:
                # undriven: keep rest local under the (possibly moved) parent
                rl = self._rest_local[ji]
                rl_q = _mat_to_quat_wxyz(rl[:3, :3].T)
                world_q[ji] = _quat_mul(world_q[pi], rl_q)
                world_t[ji] = world_t[pi] + _quat_to_mat(world_q[pi]) @ rl[3, :3]
            else:
                # undriven ROOT joint: keep its REST rotation — this is where
                # the character's upright correction lives; writing identity
                # here flattened it and inverted the whole skeleton (the
                # static/spinz anchors passed precisely because they keep
                # rest locals on the root).
                rl = self._rest_local[ji]
                world_q[ji] = _mat_to_quat_wxyz(rl[:3, :3].T)
                world_t[ji] = rl[3, :3]

            # Joint-local rotation animates; translation stays at the BIND
            # pose (fixed bone lengths — writing chain-derived translations
            # stretched the character to the robot's proportions and mangled
            # the skinning). Only the root-most driven joint translates, so
            # the character follows the fighter through the arena.
            if pi >= 0:
                lq = _quat_mul(_quat_conj(world_q[pi]), world_q[ji])
            else:
                lq = world_q[ji]
            # ALL translations stay at rest for now — including the root.
            # (The old root-follow wrote Z-up world coordinates into the
            # Y-up-bind-frame hip joint: axes scrambled = the inversion.
            # Arena-follow returns via the overlay PRIM transform instead.)
            lt = self._rest_local[ji][3, :3]
            rotations.append(Gf.Quatf(float(lq[0]), float(lq[1]), float(lq[2]), float(lq[3])))
            translations.append(Gf.Vec3f(float(lt[0]), float(lt[1]), float(lt[2])))

        self._anim.GetTranslationsAttr().Set(translations)
        self._anim.GetRotationsAttr().Set(rotations)
        self._anim.GetScalesAttr().Set(self._scales)

    def set_capsules_visible(self, robot_prim_path: str, visible: bool) -> None:
        from pxr import UsdGeom
        prim = self._stage.GetPrimAtPath(robot_prim_path)
        if prim:
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
                "inherited" if visible else "invisible"
            )


class IdentitySkelOverlay:
    """Direct SkelAnimation driver for characters whose skeleton IS the
    robot (fbx2robot creatures). No retarget math: each frame writes the
    robot's per-body world rotations into the character's joint LOCAL
    rotations through the skeleton's own bind transforms.

    char_world(j) = robot_world_delta(body j) @ bind_world(j)
    char_local(j) = char_world(parent)^-1 @ char_world(j)

    Root follows the robot root (translate + orient on the reference prim).
    Joints without a matching robot body hold their rest pose.
    """

    def __init__(self, stage, character_usd: str, prim_root: str,
                 body_names, body_rest_rot_wxyz):
        from pathlib import Path as _Path

        import numpy as np
        from pxr import Gf, Usd, UsdGeom, UsdSkel

        self._np = np
        self._Gf = Gf
        character_usd = str(_Path(character_usd).resolve())
        ref = stage.DefinePrim(prim_root, "Xform")
        ref.GetReferences().AddReference(character_usd)
        xf = UsdGeom.Xformable(ref)
        existing = xf.GetOrderedXformOps()
        self._t_op = xf.AddTranslateOp(opSuffix="overlayRoot")
        self._o_op = xf.AddOrientOp(opSuffix="overlayRoot")
        self._t_op.Set(Gf.Vec3d(0, 0, 0))
        self._o_op.Set(Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)))
        xf.SetXformOpOrder([self._t_op, self._o_op] + existing)

        skel_prim = next(
            (p for p in Usd.PrimRange(ref) if p.IsA(UsdSkel.Skeleton)), None
        )
        if skel_prim is None:
            raise ValueError(f"No UsdSkel.Skeleton under {character_usd}")
        self._skel = UsdSkel.Skeleton(skel_prim)
        joints = list(self._skel.GetJointsAttr().Get())
        self._joints = joints
        leaf = [j.split("/")[-1] for j in joints]
        parents = SkinnedOverlay._parent_indices(joints)
        self._parents = parents

        bind = self._skel.GetBindTransformsAttr().Get()
        rest = self._skel.GetRestTransformsAttr().Get()
        self._bind_world = np.array(
            [np.array(m).reshape(4, 4) for m in bind]
        )  # row-major, row-vector convention: world = local @ parent
        self._rest_local = np.array(
            [np.array(m).reshape(4, 4) for m in rest]
        )

        # robot body index per joint (None = hold rest)
        b_idx = {n: i for i, n in enumerate(body_names)}
        self._joint_body = [b_idx.get(n) for n in leaf]
        self._root_ji = next(
            i for i, b in enumerate(self._joint_body) if b is not None
        )
        self._root_bi = self._joint_body[self._root_ji]
        # robot rest world rotations per body — deltas are measured against
        # these. They arrive in the sim's COMMON convention (xyzw):
        # reorder once here.
        rest_arr = np.asarray(body_rest_rot_wxyz, dtype=np.float64)
        self._body_rest = rest_arr[:, [3, 0, 1, 2]]

        # SkelAnimation bound to the skeleton
        anim_prim = stage.DefinePrim(f"{prim_root}_anim", "SkelAnimation")
        self._anim = UsdSkel.Animation(anim_prim)
        self._anim.GetJointsAttr().Set(joints)
        UsdSkel.BindingAPI.Apply(skel_prim.GetPrim()) \
            .CreateAnimationSourceRel().SetTargets([anim_prim.GetPath()])
        # initialize with rest decomposition
        t0, r0, s0 = [], [], []
        for j in range(len(joints)):
            M = self._rest_local[j]
            t0.append(Gf.Vec3f(*[float(v) for v in M[3, :3]]))
            q = Gf.Transform(Gf.Matrix4d(*M.flatten())).GetRotation().GetQuat()
            r0.append(Gf.Quatf(q.GetReal(), Gf.Vec3f(*[float(v) for v in q.GetImaginary()])))
            s0.append(Gf.Vec3h(1, 1, 1))
        self._rest_t, self._rest_r = t0, r0
        self._anim.GetTranslationsAttr().Set(t0)
        self._anim.GetRotationsAttr().Set(r0)
        self._anim.GetScalesAttr().Set(s0)

    @staticmethod
    def _q_to_mat(q):
        import numpy as np
        w, x, y, z = q
        n = w * w + x * x + y * y + z * z
        s = 2.0 / max(n, 1e-12)
        return np.array([
            [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
            [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
            [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
        ])

    def sync(self, body_pos, body_rot_wxyz) -> None:
        """body_pos [B,3], body_rot [B,4] world; rotations arrive xyzw from
        the motion lib and are reordered here (mirrors SkinnedOverlay)."""
        np = self._np
        Gf = self._Gf
        pos = body_pos.detach().cpu().numpy().astype(np.float64)
        rot = body_rot_wxyz.detach().cpu().numpy().astype(np.float64)
        rot = rot[:, [3, 0, 1, 2]]  # xyzw -> wxyz

        J = len(self._joints)
        # world rotation per joint (3x3), robot delta applied to bind
        Rw = [None] * J
        world = [None] * J
        for j in range(J):
            bi = self._joint_body[j]
            Bw = self._bind_world[j]
            if bi is None:
                p = self._parents[j]
                if p < 0:
                    world[j] = self._rest_local[j]
                else:
                    world[j] = self._rest_local[j] @ world[p]
                continue
            q = rot[bi]
            qr = self._body_rest[bi]
            # delta = q * conj(rest)
            w1, x1, y1, z1 = q
            w2, x2, y2, z2 = qr[0], -qr[1], -qr[2], -qr[3]
            dq = (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
            D = self._q_to_mat(dq)
            M = Bw.copy()
            M[:3, :3] = Bw[:3, :3] @ D.T  # row-vector convention
            M[3, :3] = pos[bi]
            world[j] = M

        # root prim op: glue reference frame to the robot root
        rj = self._root_ji
        Wr = world[rj]
        Bw = self._bind_world[rj]
        A = np.linalg.inv(Bw) @ Wr
        T = Gf.Transform(Gf.Matrix4d(*A.flatten()))
        q = T.GetRotation().GetQuat()
        t = T.GetTranslation()
        self._t_op.Set(Gf.Vec3d(t[0], t[1], t[2]))
        self._o_op.Set(Gf.Quatf(
            float(q.GetReal()), Gf.Vec3f(*[float(v) for v in q.GetImaginary()])
        ))

        # joint locals relative to parent within the (already glued) frame
        ts, rs = list(self._rest_t), list(self._rest_r)
        for j in range(J):
            p = self._parents[j]
            L = world[j] @ np.linalg.inv(world[p]) if p >= 0 else \
                world[j] @ np.linalg.inv(A)
            TL = Gf.Transform(Gf.Matrix4d(*L.flatten()))
            qq = TL.GetRotation().GetQuat()
            tt = TL.GetTranslation()
            ts[j] = Gf.Vec3f(float(tt[0]), float(tt[1]), float(tt[2]))
            rs[j] = Gf.Quatf(
                float(qq.GetReal()),
                Gf.Vec3f(*[float(v) for v in qq.GetImaginary()]),
            )
        self._anim.GetTranslationsAttr().Set(ts)
        self._anim.GetRotationsAttr().Set(rs)
