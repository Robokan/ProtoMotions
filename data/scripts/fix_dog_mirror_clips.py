# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the dog corpora's *_mirror clips as TRUE mirrors of their originals.

The retarget pipeline's --mirror produced clips whose positions are correct
mirrors but whose every body carries a residual roll relative to a true
mirror -- invisible on roll-symmetric capsules and on positions, but the
skull rendered upside down and (once naively 'corrected') the knees bent
backward. The saga distilled to one fact about mirroring motion on a FIXED
skeleton:

  The reflection must be applied PER BODY, about that body's own lateral
  plane -- and which local axis is lateral is dictated by geometry, not
  convention. Bodies whose children attach with lateral offsets are PINNED
  (the reflection must map each child offset onto its L/R sibling's offset:
  here trunk/Spine1/Neck reflect local z). Bodies whose children hang along
  the bone axis are FREE, and must reflect local y to keep the axial chain
  (head, chest) upright.

Construction, given original global rotations G and the L/R body permutation:
  M_j     = solved per body from child offsets; default y-reflection.
  root:     G'_root = M_world . G_perm(root) . M_root      (M_world = diag(1,-1,1))
  locals:   L'_j    = M_parent(j) . L_perm(j) . M_j
  chain G' from the root; positions (gts) keep the pipeline's (already true);
  dps re-derived per joint as intrinsic-XYZ euler of the new locals (the
  packaging convention, verified to 6e-6); dvs/gavs by finite difference.

Acceptance on both corpora (2026-08-21): FK(root,dps) matches gts to 3 mm
(the skeleton itself carries a 1.5 mm L/R asymmetry in RightArm) and grs to
machine precision; heads upright 51/51 (full) and 74/74 (flat); Eric's eye
confirms knees/chest/head.

Backups: <corpus>.pt.bak_mirrorfix hold the pre-repair originals.
"""

# The executable form of this repair was run in-session on 2026-08-21 against
# data/motions/dog_v2/dog_full.pt and dog_flat.pt. If mirrors are ever
# regenerated from BVH (data/scripts/retarget_bvh_to_dog.py --mirror), DO NOT
# trust its mirrored output orientations -- run this construction on the
# packaged corpus afterwards, or port the per-body reflection into the
# retargeter. The pipeline below is the reference implementation.

import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation as R

MJCF = "protomotions/data/assets/mjcf/dog_v2_bones.xml"
CORPORA = [
    "data/motions/dog_v2/dog_full.pt",
    "data/motions/dog_v2/dog_flat.pt",
]


def main():
    m = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(m)
    from protomotions.robot_configs.factory import robot_config

    rc = robot_config("dog_v2")
    names = rc.kinematic_info.body_names
    dof_names = rc.kinematic_info.dof_names
    B = len(names)
    trip = {
        bi: [i for i, dn in enumerate(dof_names) if dn.startswith(b + "_")]
        for bi, b in enumerate(names)
    }
    trip = {bi: ids for bi, ids in trip.items() if ids}
    parent = {bi: int(m.body(names[bi]).parentid) for bi in range(B)}
    mjid_to_bi = {int(m.body(names[bi]).id): bi for bi in range(B)}
    perm = list(range(B))
    idx = {n: i for i, n in enumerate(names)}
    for n in names:
        if n.startswith("Left") and ("Right" + n[4:]) in idx:
            perm[idx[n]], perm[idx["Right" + n[4:]]] = idx["Right" + n[4:]], idx[n]

    M_world = np.diag([1.0, -1.0, 1.0])
    My, Mz = np.diag([1.0, -1.0, 1.0]), np.diag([1.0, 1.0, -1.0])
    body_off = {bi: m.body(names[bi]).pos.copy() for bi in range(B)}
    children = {bi: [] for bi in range(B)}
    for bi in range(B):
        if bi:
            children[mjid_to_bi[parent[bi]]].append(bi)
    Msel = {}
    for bi in range(B):
        need = None
        for c in children[bi]:
            off, offp = body_off[c], body_off[perm[c]]
            if abs(off[1]) < 1e-6 and abs(off[2]) < 1e-6:
                continue
            for tag, cand in (("y", My), ("z", Mz)):
                if np.allclose(cand @ off, offp, atol=1e-4):
                    need = tag if need in (None, tag) else "CONFLICT"
        assert need != "CONFLICT", names[bi]
        Msel[bi] = need or "y"
    Mmat = {bi: (My if Msel[bi] == "y" else Mz) for bi in range(B)}
    topo = sorted(range(B), key=lambda bi: int(m.body(names[bi]).id))

    for path in CORPORA:
        d = torch.load(path, map_location="cpu", weights_only=False)
        files = [f.split("/")[-1] for f in d["motion_files"]]
        fixed = 0
        for i, f in enumerate(files):
            if "mirror" not in f or f.replace("_mirror", "") not in files:
                continue
            o = files.index(f.replace("_mirror", ""))
            so, sm = int(d["length_starts"][o]), int(d["length_starts"][i])
            F = int(d["motion_num_frames"][i])
            if int(d["motion_num_frames"][o]) != F:
                continue
            grs_o = d["grs"][so : so + F].numpy().astype(np.float64)
            new = np.zeros((F, B, 4))
            new[:, 0] = R.from_matrix(
                M_world @ R.from_quat(grs_o[:, perm[0]]).as_matrix() @ Mmat[0]
            ).as_quat()
            for bi in topo:
                if bi == 0:
                    continue
                pbi = mjid_to_bi[parent[bi]]
                po = mjid_to_bi[parent[perm[bi]]]
                Lo = (
                    R.from_quat(grs_o[:, po]).inv()
                    * R.from_quat(grs_o[:, perm[bi]])
                ).as_matrix()
                Ln = np.einsum("ij,njk,kl->nil", Mmat[pbi], Lo, Mmat[bi])
                new[:, bi] = (
                    R.from_quat(new[:, pbi]) * R.from_matrix(Ln)
                ).as_quat()
            d["grs"][sm : sm + F] = torch.from_numpy(new).float()
            dt = float(d["motion_dt"][i])
            nd = d["dps"][sm : sm + F].numpy().copy()
            for bi, ids in trip.items():
                pbi = mjid_to_bi[parent[bi]]
                loc = R.from_quat(new[:, pbi]).inv() * R.from_quat(new[:, bi])
                nd[:, ids] = np.unwrap(loc.as_euler("XYZ"), axis=0)
            d["dps"][sm : sm + F] = torch.from_numpy(nd).float()
            d["dvs"][sm : sm + F] = torch.from_numpy(
                np.gradient(nd, dt, axis=0)
            ).float()
            dq = R.from_quat(new[1:].reshape(-1, 4)) * R.from_quat(
                new[:-1].reshape(-1, 4)
            ).inv()
            w = dq.as_rotvec().reshape(F - 1, B, 3) / dt
            d["gavs"][sm : sm + F] = torch.from_numpy(
                np.concatenate([w, w[-1:]], 0)
            ).float()
            fixed += 1
        torch.save(d, path)
        print(f"{path}: {fixed} mirror clips rebuilt")


if __name__ == "__main__":
    main()
