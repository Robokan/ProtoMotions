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
    ):
        from pxr import Usd, UsdGeom, UsdSkel, Sdf, Gf  # noqa: F401

        self._stage = stage
        self._root = prim_root
        self._body_index = {n: i for i, n in enumerate(body_names)}

        ref = stage.DefinePrim(prim_root, "Xform")
        ref.GetReferences().AddReference(character_usd)
        if scale != 1.0:
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
