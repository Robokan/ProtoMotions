# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""The camera eye's geometry: projection, window test and crop."""

import math

import torch

from protomotions.envs.control.camera_eye import (
    crop_frames,
    project_to_camera,
    window_contains,
)

ROOT = torch.tensor([[0.0, 0.0, 0.29]])
LENS = 0.33


def _quat(axis, deg):
    """Axis-angle -> xyzw."""
    half = math.radians(deg) / 2
    s = math.sin(half)
    return torch.tensor([[axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)]])


LEVEL = _quat((0, 0, 1), 0)


def _image(point, rot=LEVEL):
    return project_to_camera(torch.tensor([point]), ROOT, rot, LENS, 120.0, 16 / 9)


def _whole_frame(u, v, depth, tan_h, tan_v):
    zero = torch.zeros(1)
    return bool(
        window_contains(u, v, depth, 0.12, tan_h, tan_v, zero, zero, torch.ones(1))
    )


def test_straight_ahead_is_centred_and_below_the_horizon():
    u, v, depth, _, _ = _image([LENS + 3.0, 0.0, 0.12])
    assert abs(float(u)) < 1e-6
    assert float(v) > 0  # the ball rests below the lens
    assert abs(float(depth) - 3.0) < 1e-5


def test_half_fov_from_the_lens_is_the_frame_edge():
    a = math.radians(60)
    u, *_ = _image([LENS + 3 * math.cos(a), 3 * math.sin(a), 0.29])
    assert abs(float(u) + 1.0) < 1e-5  # left edge


def test_behind_the_lens_is_never_seen():
    assert not _whole_frame(*_image([-2.0, 0.0, 0.12]))


def test_body_roll_moves_the_ball_in_the_image():
    point = [LENS + 2.0, 1.0, 0.12]
    level = _image(point)
    rolled = _image(point, _quat((1, 0, 0), 25))
    assert abs(float(rolled[1]) - float(level[1])) > 0.1


def test_partly_visible_ball_at_the_edge_counts_as_seen():
    # Centre just past the left edge, but well within half a radius of it.
    a = math.radians(61)
    near = _image([LENS + 1.0 * math.cos(a), 1.0 * math.sin(a), 0.29])
    assert float(near[0]) < -1.0
    assert _whole_frame(*near)


def test_zoomed_window_excludes_what_the_frame_includes():
    # 34 deg left: well inside the frame, well outside a centred 4x window.
    u, v, depth, tan_h, tan_v = _image([LENS + 3.0, 2.0, 0.12])
    assert _whole_frame(u, v, depth, tan_h, tan_v)
    zoomed = window_contains(
        u, v, depth, 0.12, tan_h, tan_v,
        torch.zeros(1), torch.zeros(1), torch.full((1,), 4.0),
    )
    assert not bool(zoomed)


def test_crop_centres_the_gaze_and_zoom_one_is_the_whole_frame():
    frames = torch.zeros(1, 720, 1280, 3, dtype=torch.uint8)
    frames[0, 355:365, 955:965] = 255  # a dot at u = 0.5, v = 0
    crop = crop_frames(
        frames, torch.tensor([0.5]), torch.zeros(1), torch.full((1,), 4.0), 224
    )
    assert crop.shape == (1, 224, 224, 3)
    ys, xs = torch.nonzero(crop[0, ..., 0] > 50, as_tuple=True)
    assert abs(float(xs.float().mean()) - 111.5) < 2
    assert abs(float(ys.float().mean()) - 111.5) < 2

    whole = crop_frames(frames, torch.zeros(1), torch.zeros(1), torch.ones(1), 224)
    ys, xs = torch.nonzero(whole[0, ..., 0] > 5, as_tuple=True)
    assert abs(float(xs.float().mean()) - 0.75 * 224) < 2


def test_viewer_outline_lands_on_the_window_edges():
    from types import SimpleNamespace

    from protomotions.envs.control.camera_eye import CameraEye, CameraEyeConfig

    rot = _quat((1, 0, 0), 15)  # a rolled dog: the outline must ride the pose
    goal = SimpleNamespace(
        config=SimpleNamespace(fov_deg=120.0, sight_aspect=16 / 9, sight_forward_m=LENS)
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        dt=0.02,
        control_manager=SimpleNamespace(components={"masked_mimic": goal}),
        simulator=SimpleNamespace(
            show_markers=True,
            get_root_state=lambda: SimpleNamespace(root_pos=ROOT, root_rot=rot),
        ),
    )
    eye = CameraEye(CameraEyeConfig(), env)
    eye.center_u[:] = 0.3
    eye.center_v[:] = -0.2
    eye.zoom[:] = 2.5
    dots = eye.get_markers_state()["eye_window"].translation[0]
    u, v, depth, _, _ = project_to_camera(
        dots, ROOT.expand(len(dots), 3), rot.expand(len(dots), 4), LENS, 120.0, 16 / 9
    )
    half = 1 / 2.5
    edge = torch.minimum(
        ((u - 0.3).abs() - half).abs(), ((v + 0.2).abs() - half).abs()
    )
    assert float(edge.max()) < 1e-4
    assert float((u - 0.3).abs().max()) <= half + 1e-4
    assert float((v + 0.2).abs().max()) <= half + 1e-4
    assert torch.allclose(depth, torch.full_like(depth, 0.5), atol=1e-4)


def test_lost_ball_zooms_out_one_level_at_a_time_around_the_same_spot():
    from types import SimpleNamespace

    from protomotions.envs.control.camera_eye import CameraEye, CameraEyeConfig

    # A ball well to the left: outside a window zoomed in on the right.
    ball = _image([LENS + 3.0, 2.0, 0.12])
    goal = SimpleNamespace(
        config=SimpleNamespace(ball_radius_m=0.12), _ball_image=lambda: ball
    )
    env = SimpleNamespace(
        num_envs=1, device="cpu", dt=0.02,
        control_manager=SimpleNamespace(components={"masked_mimic": goal}),
    )
    eye = CameraEye(CameraEyeConfig(hz=50.0), env)  # an eye update every step
    eye.center_u[:] = 0.5
    eye.zoom[:] = 4.0
    zooms, centres = [], []
    for _ in range(3):
        eye.step()
        zooms.append(round(float(eye.zoom), 3))
        centres.append(round(float(eye.center_u), 3))
    # Out a level, out to the whole frame -- where the ball IS in view, so
    # the eye reacquires it and starts zooming back in on it.
    assert zooms == [2.0, 1.0, 1.5]
    assert centres[0] == 0.5  # widened in place, not recentred
    assert centres[1] == 0.0  # zoom 1 is the whole frame
    assert centres[2] < 0.0  # re-centred on the ball, off to the left


def test_camera_marker_sits_on_the_lens_and_points_along_the_body():
    from types import SimpleNamespace

    from protomotions.envs.control.camera_marker import CameraMarker, CameraMarkerConfig
    from protomotions.robot_configs.go2 import go2_front_camera

    cam = go2_front_camera()
    for rot, forward in ((LEVEL, (1.0, 0.0, 0.0)), (_quat((0, 0, 1), 90), (0.0, 1.0, 0.0))):
        env = SimpleNamespace(
            num_envs=1, device="cpu",
            simulator=SimpleNamespace(
                config=SimpleNamespace(onboard_cameras={"front_camera": cam}),
                show_markers=True,
                get_root_state=lambda rot=rot: SimpleNamespace(root_pos=ROOT, root_rot=rot),
            ),
        )
        state = CameraMarker(CameraMarkerConfig(), env).get_markers_state()
        lens = state["camera_lens"].translation[0, 0]
        axis = state["camera_axis"].translation[0]
        # Forward offset along the body's heading, height straight up.
        expected = ROOT[0] + torch.tensor(forward) * cam.pos[0] + torch.tensor([0.0, 0.0, cam.pos[2]])
        assert torch.allclose(lens, expected, atol=1e-5)
        step = axis[1] - axis[0]
        assert torch.allclose(step / step.norm(), torch.tensor(forward), atol=1e-5)
