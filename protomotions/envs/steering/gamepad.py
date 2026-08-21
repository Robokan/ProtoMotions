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

import logging
import os
import struct
import threading

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


class GamepadReader:
    """Background reader for one /dev/input/js* device."""

    def __init__(self, device: str = "/dev/input/js0", num_buttons: int = 4):
        self.device = device
        self.connected = False
        # Normalized command channels [forward, turn, side] in [-1, 1].
        self.channels = [0.0, 0.0, 0.0]
        self.buttons = [0.0] * max(num_buttons, 1)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
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
                                if number == AXIS_FORWARD:
                                    self.channels[0] = -v  # stick up = forward
                                elif number == AXIS_TURN:
                                    self.channels[1] = -v  # stick left = +yaw
                                elif number == AXIS_SIDE:
                                    self.channels[2] = -v  # stick left = +lateral
                        elif ev_type == _JS_EVENT_BUTTON:
                            with self._lock:
                                if number < len(self.buttons):
                                    self.buttons[number] = float(value)
            except OSError as e:
                self.connected = False
                log.warning("gamepad read error (%s); retrying", e)
                threading.Event().wait(2.0)

    def state(self):
        """Thread-safe snapshot: ([fwd, turn, side] in [-1,1], [buttons])."""
        with self._lock:
            return list(self.channels), list(self.buttons)
