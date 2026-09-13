# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free Linux gamepad reader for steering teleop.

Reads the kernel joystick interface (/dev/input/js0) directly -- 8-byte
events, no pygame/evdev required -- on a daemon thread, and exposes the
current stick/button state. Mapping follows the IsaacLabASE game-controller
convention on a standard Xbox-layout pad:

    left stick Y  (axis 1, inverted)  -> forward velocity command
    right stick X (axis 3)            -> yaw-rate command (turn)
    left stick X  (axis 0)            -> lateral velocity command
    face buttons 0..N-1               -> skill buttons (held = 1.0)

Axis values are normalized to [-1, 1] and scaled by the steering component's
per-channel command bounds at the point of use.
"""

import fcntl
import glob
import logging
import os
import select
import struct
import threading
import time

log = logging.getLogger(__name__)

_EVENT_FORMAT = "IhBB"  # u32 time_ms, s16 value, u8 type, u8 number
_EVENT_SIZE = struct.calcsize(_EVENT_FORMAT)
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS = 0x02
_JS_EVENT_INIT = 0x80

AXIS_FORWARD = 1  # left stick Y (up = negative raw value)
AXIS_TURN = 3     # right stick X
AXIS_SIDE = 0     # left stick X
DEADZONE = 0.08


# ioctls from linux/joystick.h
_JSIOCGAXES = 0x80016A11     # u8: number of axes
_JSIOCGBUTTONS = 0x80016A12  # u8: number of buttons
_JSIOCGNAME_LEN = 128
_JSIOCGNAME = 0x80006A13 | (_JSIOCGNAME_LEN << 16)

# A stick axis that REST at (near) full deflection is not a stick. Real pads
# rest their sticks near centre; only triggers rest at an extreme, and none of
# the axes we map is a trigger.
_REST_LIMIT = 0.5


def probe_device(device: str, settle_s: float = 0.4):
    """Identify a /dev/input/js* node.

    Returns (name, num_axes, resting, usable, reason). ``resting`` maps axis
    number -> normalized resting value, taken from the JS_EVENT_INIT burst the
    kernel queues on open.

    Exists because ANY HID device can claim a js* node: this machine's
    /dev/input/js0 is an "ASRock LED Controller" (motherboard RGB, 11 axes,
    8 buttons) whose axes sit pinned at full scale. The old "a js* node exists
    => teleop ON" check enabled it and pinned the viewed robot at max forward
    + max left yaw + max lateral forever (Eric, 2026-09-13: "its always
    turning to the left").
    """
    try:
        f = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
        return None, 0, {}, False, f"open failed: {e}"
    try:
        try:
            axes = struct.unpack("B", fcntl.ioctl(f, _JSIOCGAXES, b"\0"))[0]
            buf = bytearray(_JSIOCGNAME_LEN)
            fcntl.ioctl(f, _JSIOCGNAME, buf)
            name = bytes(buf).split(b"\0")[0].decode("utf-8", "replace")
        except OSError as e:
            return None, 0, {}, False, f"ioctl failed: {e}"

        resting = {}
        deadline = time.time() + settle_s
        while time.time() < deadline:
            r, _, _ = select.select([f], [], [], max(0.0, deadline - time.time()))
            if not r:
                break
            try:
                data = os.read(f, _EVENT_SIZE)
            except BlockingIOError:
                break
            if len(data) < _EVENT_SIZE:
                break
            _, value, ev_type, number = struct.unpack(_EVENT_FORMAT, data)
            if ev_type & ~_JS_EVENT_INIT == _JS_EVENT_AXIS:
                resting[number] = value / 32767.0

        pinned = [
            n for n in (AXIS_FORWARD, AXIS_TURN, AXIS_SIDE)
            if abs(resting.get(n, 0.0)) > _REST_LIMIT
        ]
        if axes < 4:
            return name, axes, resting, False, f"only {axes} axes"
        if pinned:
            vals = ", ".join(f"axis{n}={resting[n]:+.2f}" for n in pinned)
            return name, axes, resting, False, f"axes pinned at rest ({vals})"
        return name, axes, resting, True, "ok"
    finally:
        os.close(f)


def find_gamepad():
    """First /dev/input/js* node that actually behaves like a gamepad.

    Returns (device, name) or (None, None). Rejected nodes are logged at INFO
    so a phantom device is visible rather than silently driving the robot.
    """
    for dev in sorted(glob.glob("/dev/input/js*")):
        name, axes, _resting, usable, reason = probe_device(dev)
        if usable:
            log.info("gamepad: using %s (%s, %d axes)", dev, name, axes)
            return dev, name
        log.info("gamepad: ignoring %s (%s): %s", dev, name or "unknown", reason)
    return None, None


class GamepadReader:
    """Background reader for one /dev/input/js* device."""

    def __init__(self, device: str = None, num_buttons: int = 4):
        # None = auto-select a device that passes probe_device(); an explicit
        # path is still validated, and a failing one disables teleop rather
        # than feeding the robot a phantom stick.
        if device is None:
            device, _name = find_gamepad()
        elif device:
            _name, _axes, _rest, usable, reason = probe_device(device)
            if not usable:
                log.warning(
                    "gamepad: %s rejected (%s) -- teleop disabled", device, reason
                )
                device = None
        self.device = device
        self.usable = device is not None
        self.connected = False
        # Normalized command channels [forward, turn, side] in [-1, 1].
        self.channels = [0.0, 0.0, 0.0]
        self.buttons = [0.0] * max(num_buttons, 1)
        self._last_active = 0.0  # wall time of the last nonzero input
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if not self.device:
            return
        while True:
            if not os.path.exists(self.device):
                self.connected = False
                threading.Event().wait(2.0)
                continue
            try:
                with open(self.device, "rb") as f:
                    self.connected = True
                    log.info("gamepad connected: %s", self.device)
                    while True:
                        data = f.read(_EVENT_SIZE)
                        if len(data) < _EVENT_SIZE:
                            break
                        _, value, ev_type, number = struct.unpack(
                            _EVENT_FORMAT, data
                        )
                        ev_type &= ~_JS_EVENT_INIT
                        if ev_type == _JS_EVENT_AXIS:
                            v = value / 32767.0
                            if abs(v) < DEADZONE:
                                v = 0.0
                            with self._lock:
                                if v != 0.0:
                                    self._last_active = time.time()
                                if number == AXIS_FORWARD:
                                    self.channels[0] = -v  # stick up = forward
                                elif number == AXIS_TURN:
                                    self.channels[1] = -v  # stick left = +yaw
                                elif number == AXIS_SIDE:
                                    self.channels[2] = -v  # stick left = +lateral
                        elif ev_type == _JS_EVENT_BUTTON:
                            with self._lock:
                                if value:
                                    self._last_active = time.time()
                                if number < len(self.buttons):
                                    self.buttons[number] = float(value)
            except OSError as e:
                self.connected = False
                log.warning("gamepad read error (%s); retrying", e)
                threading.Event().wait(2.0)

    def state(self):
        """Thread-safe snapshot: ([fwd, turn, side], [buttons], active).

        active is True while the pad is actually being USED: any nonzero
        input within the last second, or a stick/button currently deflected.
        An idle pad releases the selected robot back to the random command
        generator so it behaves like every other env."""
        with self._lock:
            deflected = any(abs(c) > 0.0 for c in self.channels) or any(
                b > 0.0 for b in self.buttons
            )
            active = deflected or (time.time() - self._last_active) < 1.0
            return list(self.channels), list(self.buttons), active
