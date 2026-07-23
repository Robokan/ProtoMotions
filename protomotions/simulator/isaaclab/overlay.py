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

        # offset(j) = rest_rot(body)^-1 * bind_world_rot(j)
        self._offset = np.tile(np.array([1.0, 0, 0, 0]), (len(self._joints), 1))
        for ji, bi in enumerate(self._driven):
            if bi < 0:
                continue
            bind_q = _mat_to_quat_wxyz(self._bind_world[ji, :3, :3].T)
            self._offset[ji] = _quat_mul(
                _quat_conj(body_rest_rot_wxyz[bi]), bind_q
            )

        # SkelAnimation prim bound to the skeleton.
        anim_path = skel_prim.GetPath().AppendChild("OverlayAnim")
        self._anim = UsdSkel.Animation.Define(stage, anim_path)
        self._anim.GetJointsAttr().Set(self._joints)
        binding = UsdSkel.BindingAPI.Apply(skel_prim)
        binding.GetAnimationSourceRel().SetTargets([anim_path])

        n = len(self._joints)
        self._scales = [Gf.Vec3h(1.0, 1.0, 1.0)] * n
        self._Gf = Gf
        log.info(
            "SkinnedOverlay: %d joints, %d driven by robot bodies",
            n, int((self._driven >= 0).sum()),
        )

    @staticmethod
    def _parent_indices(joints: List[str]) -> np.ndarray:
        index = {j: i for i, j in enumerate(joints)}
        parents = np.full(len(joints), -1, dtype=np.int64)
        for i, j in enumerate(joints):
            parent = j.rsplit("/", 1)[0] if "/" in j else None
            if parent and parent in index:
                parents[i] = index[parent]
        return parents

    def sync(self, body_pos: torch.Tensor, body_rot_wxyz: torch.Tensor) -> None:
        """Write one frame. body_pos [B,3], body_rot [B,4] world, wxyz."""
        Gf = self._Gf
        pos = body_pos.detach().cpu().numpy().astype(np.float64)
        rot = body_rot_wxyz.detach().cpu().numpy().astype(np.float64)

        # World rotation per joint: driven -> body*offset; else parent-follow
        # keeps bind-relative local (computed below via rest_local fallback).
        world_q = np.zeros((len(self._joints), 4))
        world_t = np.zeros((len(self._joints), 3))
        translations = []
        rotations = []
        for ji in range(len(self._joints)):
            pi = self._parents[ji]
            bi = self._driven[ji]
            if bi >= 0:
                world_q[ji] = _quat_mul(rot[bi], self._offset[ji])
                world_t[ji] = pos[bi]
            elif pi >= 0:
                # undriven: keep rest local under the (possibly moved) parent
                rl = self._rest_local[ji]
                rl_q = _mat_to_quat_wxyz(rl[:3, :3].T)
                world_q[ji] = _quat_mul(world_q[pi], rl_q)
                world_t[ji] = world_t[pi] + _quat_to_mat(world_q[pi]) @ rl[3, :3]
            else:
                world_q[ji] = np.array([1.0, 0, 0, 0])

            # joint-local for the animation
            if pi >= 0:
                lq = _quat_mul(_quat_conj(world_q[pi]), world_q[ji])
                lt = _quat_to_mat(world_q[pi]).T @ (world_t[ji] - world_t[pi])
            else:
                lq, lt = world_q[ji], world_t[ji]
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
