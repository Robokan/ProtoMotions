# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Keyboard-controlled viewport camera tracker for IsaacLab simulator.
# Allows orbiting around robots, tracking their movement, and switching between robots.
#
# Controls:
#   TAB / = : Next robot
#   -       : Previous robot
#   T       : Toggle tracking (camera looks at robot)
#   F       : Toggle following (camera follows robot position)
#   LEFT/RIGHT : Rotate view around robot
#   PAGE_UP/DOWN : Change camera distance
#   HOME/END : Change target height
#   UP/DOWN : Change camera height

import omni
import carb
import math
import numpy as np
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from protomotions.simulator.isaaclab.simulator import IsaacLabSimulator


class KeyboardCameraTracker:
    """
    Keyboard-controlled camera tracker for orbiting around and following robots.
    
    This provides a more interactive camera control than the default follow camera,
    allowing rotation around the target, distance adjustment, and switching between
    multiple robots.
    """
    
    def __init__(
        self,
        simulator: "IsaacLabSimulator",
        num_envs: int,
        get_root_pos_func: Optional[Callable] = None,
    ):
        """
        Initialize the camera tracker.
        
        Args:
            simulator: The IsaacLab simulator instance
            num_envs: Number of environments/robots
            get_root_pos_func: Optional function to get root positions. 
                               If None, uses simulator._robot.data.root_pos_w
        """
        self._simulator = simulator
        self._num_envs = num_envs
        self._get_root_pos_func = get_root_pos_func
        
        # Camera state
        self._robot_index = 0
        self._tracking = True  # Camera looks at robot
        self._following = True  # Camera follows robot position
        
        # View parameters
        self._eye = None
        self._target = None
        self._rotation = 0.0
        self._desired_rotation = 0.0
        self._distance = 4.0
        self._desired_distance = 4.0
        self._camera_height = 1.5
        self._desired_camera_height = 1.5
        self._target_height = 0.5
        self._desired_target_height = 0.5
        
        # Computed view direction
        self._x_view_amount = 1.0
        self._y_view_amount = 0.0
        
        # Set up keyboard input
        self._setup_keyboard()
        
        print("[KeyboardCameraTracker] Initialized with controls:")
        print("  TAB/= : Next robot  |  - : Previous robot")
        print("  T : Toggle tracking  |  F : Toggle following")
        print("  LEFT/RIGHT : Rotate  |  PAGE_UP/DOWN : Distance")
        print("  HOME/END : Target height  |  UP/DOWN : Camera height")
    
    def _setup_keyboard(self) -> None:
        """Set up keyboard event subscription."""
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )
        
        # Key mappings: key_name -> (function, kwargs)
        self._key_mapping = {
            'TAB': (self.next_robot, {}),
            'EQUAL': (self.next_robot, {}),
            'MINUS': (self.prev_robot, {}),
            'T': (self.toggle_tracking, {}),
            'F': (self.toggle_following, {}),
            'LEFT': (self.rotate_view_by, {'amount': -0.2}),
            'RIGHT': (self.rotate_view_by, {'amount': 0.2}),
            'PAGE_UP': (self.change_distance, {'amount': -0.5}),
            'PAGE_DOWN': (self.change_distance, {'amount': 0.5}),
            'HOME': (self.change_target_height, {'amount': 0.1}),
            'END': (self.change_target_height, {'amount': -0.1}),
            'UP': (self.change_camera_height, {'amount': 0.2}),
            'DOWN': (self.change_camera_height, {'amount': -0.2}),
        }
    
    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            if key in self._key_mapping:
                func, params = self._key_mapping[key]
                func(**params)
    
    def get_root_pos(self) -> np.ndarray:
        """Get root positions of all robots."""
        if self._get_root_pos_func is not None:
            return self._get_root_pos_func()
        else:
            return self._simulator._robot.data.root_pos_w.clone().detach().cpu().numpy()
    
    @property
    def robot_index(self) -> int:
        """Current tracked robot index."""
        return self._robot_index
    
    @robot_index.setter
    def robot_index(self, value: int) -> None:
        """Set current tracked robot index."""
        self._robot_index = value % self._num_envs
    
    # =====================================================
    # Robot Selection
    # =====================================================
    
    def next_robot(self) -> None:
        """Switch to next robot."""
        self._robot_index = (self._robot_index + 1) % self._num_envs
        print(f"[Camera] Tracking robot {self._robot_index}")
        self._move_camera_to_robot()
    
    def prev_robot(self) -> None:
        """Switch to previous robot."""
        self._robot_index = (self._robot_index - 1) % self._num_envs
        print(f"[Camera] Tracking robot {self._robot_index}")
        self._move_camera_to_robot()
    
    # =====================================================
    # Toggle States
    # =====================================================
    
    def toggle_tracking(self) -> None:
        """Toggle whether camera looks at robot."""
        self._tracking = not self._tracking
        print(f"[Camera] Tracking: {self._tracking}")
    
    def toggle_following(self) -> None:
        """Toggle whether camera follows robot position."""
        self._following = not self._following
        print(f"[Camera] Following: {self._following}")
    
    # =====================================================
    # View Parameter Adjustment
    # =====================================================
    
    def rotate_view_by(self, amount: float) -> None:
        """Rotate camera view around robot."""
        if not self._following:
            return
        self._desired_rotation += amount
    
    def change_distance(self, amount: float) -> None:
        """Change camera distance from robot."""
        if not self._following:
            return
        self._desired_distance += amount
        self._desired_distance = self._clip(self._desired_distance, 1.0, 50.0)
    
    def change_camera_height(self, amount: float) -> None:
        """Change camera height."""
        if not self._following:
            return
        self._desired_camera_height += amount
        self._desired_camera_height = self._clip(self._desired_camera_height, 0.2, 20.0)
    
    def change_target_height(self, amount: float) -> None:
        """Change target look-at height."""
        if not self._tracking:
            return
        self._desired_target_height += amount
        self._desired_target_height = self._clip(self._desired_target_height, 0.1, 5.0)
    
    # =====================================================
    # Camera Update
    # =====================================================
    
    def step(self) -> None:
        """
        Update camera position and orientation.
        Call this each frame/step to smoothly update the camera.
        """
        self._update_smooth_parameters()
        
        if self._following:
            self._update_eye()
        if self._tracking:
            self._update_target()
        
        if self._following or self._tracking:
            self._move_camera_to_robot()
    
    def _update_smooth_parameters(self) -> None:
        """Smoothly interpolate view parameters towards desired values."""
        smooth_factor = 0.1
        
        # Smooth rotation
        rotation_delta = (self._desired_rotation - self._rotation) * smooth_factor
        self._rotation += rotation_delta
        self._x_view_amount = math.cos(self._rotation)
        self._y_view_amount = math.sin(self._rotation)
        
        # Smooth distance
        distance_delta = (self._desired_distance - self._distance) * smooth_factor
        self._distance += distance_delta
        
        # Smooth camera height
        height_delta = (self._desired_camera_height - self._camera_height) * smooth_factor
        self._camera_height += height_delta
        
        # Smooth target height
        target_height_delta = (self._desired_target_height - self._target_height) * smooth_factor
        self._target_height += target_height_delta
    
    def _update_eye(self) -> None:
        """Update camera eye position based on robot position."""
        root_pos = self.get_root_pos()
        robot_pos = root_pos[self._robot_index]
        
        self._eye = np.array([
            robot_pos[0] + self._x_view_amount * self._distance,
            robot_pos[1] + self._y_view_amount * self._distance,
            robot_pos[2] + self._camera_height,
        ])
    
    def _update_target(self) -> None:
        """Update camera target (look-at) position based on robot position."""
        root_pos = self.get_root_pos()
        robot_pos = root_pos[self._robot_index]
        
        self._target = np.array([
            robot_pos[0],
            robot_pos[1],
            robot_pos[2] + self._target_height,
        ])
    
    def _move_camera_to_robot(self) -> None:
        """Apply camera position and target to the simulator."""
        if self._eye is None:
            self._update_eye()
        if self._target is None:
            self._update_target()
        
        if self._eye is not None and self._target is not None:
            self._simulator._sim.set_camera_view(
                eye=self._eye.tolist(),
                target=self._target.tolist()
            )
    
    # =====================================================
    # Utilities
    # =====================================================
    
    @staticmethod
    def _clip(value: float, min_val: float, max_val: float) -> float:
        """Clip value to range."""
        return max(min_val, min(max_val, value))
    
    def cleanup(self) -> None:
        """Clean up keyboard subscription."""
        if hasattr(self, '_sub_keyboard') and self._sub_keyboard is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None
