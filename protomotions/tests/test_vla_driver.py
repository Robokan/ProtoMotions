# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""The VLA driver sends back exactly the conditioning the recorder labelled."""

from types import SimpleNamespace

import torch

from protomotions.envs.control.lerobot_recorder import masked_target_names, masked_targets
from protomotions.envs.control.vla_driver import targets_to_world
from protomotions.utils import rotations


class _Terrain:
    def get_ground_heights(self, xy):
        return 0.1 * xy[..., 0:1]  # a slope, so "height above ground" matters


def _quat(yaw, pitch):
    z = torch.tensor([[0.0, 0.0, 1.0]]).expand(len(yaw), 3)
    y = torch.tensor([[0.0, 1.0, 0.0]]).expand(len(yaw), 3)
    return rotations.quat_mul(
        rotations.quat_from_angle_axis(yaw, z, True),
        rotations.quat_from_angle_axis(pitch, y, True), True,
    )


def test_masked_targets_round_trip():
    torch.manual_seed(0)
    n, steps, bodies = 3, 5, 6
    cond = torch.tensor([0, 2, 3, 5])
    goal = SimpleNamespace(
        env=SimpleNamespace(robot_config=SimpleNamespace(kinematic_info=SimpleNamespace(
            body_names=[f"b{i}" if i else "base_link" for i in range(bodies)]))),
        conditionable_body_ids=cond,
    )
    root = SimpleNamespace(
        root_pos=torch.tensor([[1.0, 2.0, 0.4], [-3.0, 0.5, 0.35], [0.0, 0.0, 0.3]]),
        root_rot=_quat(torch.tensor([0.3, -2.0, 3.0]), torch.tensor([0.2, -0.1, 0.0])),
    )
    ref_pos = torch.randn(n, steps, bodies, 3)
    ref_rot = _quat(torch.rand(n * steps * bodies) * 6 - 3, torch.rand(n * steps * bodies) - 0.5)
    ref_rot = ref_rot.view(n, steps, bodies, 4)
    masks = torch.rand(n, steps, len(cond), 2) > 0.3
    mm = SimpleNamespace(
        ref_pos=ref_pos, ref_rot=ref_rot, target_bodies_masks=masks.view(n, -1),
        time_offsets=torch.rand(n, steps),
    )
    terrain = _Terrain()
    act = masked_targets(mm, goal, root, terrain)
    assert act.shape[1] == len(masked_target_names(goal))
    pos, rot, bits, seconds = targets_to_world(act, root.root_pos, root.root_rot, terrain)
    final = masks[:, -1]
    assert torch.equal(bits, final)
    assert torch.allclose(seconds, mm.time_offsets[:, -1])
    want_pos = ref_pos[:, -1][:, cond]
    on = final[..., 0]
    assert torch.allclose(pos[on], want_pos[on], atol=1e-5)
    want_rot = ref_rot[:, -1][:, cond]
    rot_on = final[..., 1]
    # q and -q are the same rotation.
    dot = (rot[rot_on] * want_rot[rot_on]).sum(-1).abs()
    assert torch.allclose(dot, torch.ones_like(dot), atol=1e-5)
    # Orientations it does not ask for come back as the identity.
    assert torch.allclose(rot[~rot_on][:, 3], torch.ones(int((~rot_on).sum())))
