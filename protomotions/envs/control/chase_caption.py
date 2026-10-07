# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Caption the viewer: what the followed dog was asked and what it says.

Viewer only (nothing is drawn headless, and it is a UI overlay, not stage
content, so it never gets into camera frames or recordings). The text is the
recorder's (lerobot_recorder.language_texts), so what is on screen is what a
recording would label that frame. When a VLA is driving, its own last answer
is shown with the demonstrator's alongside for comparison.

Follows whichever dog the camera follows (the viewer's next/previous env keys).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class ChaseCaptionConfig(ControlComponentConfig):
    """Configuration for the viewer caption."""

    _target_: str = "protomotions.envs.control.chase_caption.ChaseCaption"

    goal_component: str = "masked_mimic"
    ball_component: str = "ball"
    # The recorder's tables when empty (LeRobotRecorderConfig).
    prompts: Dict[str, List[str]] = field(default_factory=dict)
    responses: Dict[str, List[str]] = field(default_factory=dict)
    font_size: float = 26.0
    # Gap under the caption, pixels from the bottom of the viewport.
    bottom_margin_px: float = 40.0


class ChaseCaption(ControlComponent):
    """Show the followed dog's prompt and response at the bottom of the viewport."""

    config: ChaseCaptionConfig

    def __init__(self, config: ChaseCaptionConfig, env: "BaseEnv"):
        super().__init__(config, env)
        from protomotions.envs.control.lerobot_recorder import LeRobotRecorderConfig

        defaults = LeRobotRecorderConfig()
        self._prompts = config.prompts or defaults.prompts
        self._responses = config.responses or defaults.responses
        self._frame = None
        self._prompt_label = None
        self._response_label = None
        self._shown = None
        self._failed = False

    def reset(self, env_ids: Tensor) -> None:
        pass

    def populate_context(self, ctx: EnvContext) -> None:
        """Visualization only."""

    def _build(self) -> None:
        """The overlay: a dark box centred at the bottom of the viewport."""
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport_window

        window = get_active_viewport_window()
        if window is None:
            raise RuntimeError("no active viewport window")
        # omni.ui colours are 0xAABBGGRR.
        text = {"font_size": self.config.font_size, "color": 0xFFFFFFFF}
        said = {"font_size": self.config.font_size, "color": 0xFF7FE6FF}
        self._frame = window.get_frame("chase_caption")
        with self._frame:
            with ui.VStack():
                ui.Spacer()
                with ui.HStack(height=0):
                    ui.Spacer()
                    with ui.ZStack(width=0, height=0):
                        ui.Rectangle(style={"background_color": 0xB0000000, "border_radius": 8})
                        with ui.HStack(width=0, height=0):
                            ui.Spacer(width=18)
                            with ui.VStack(width=0, height=0, spacing=4):
                                ui.Spacer(height=10)
                                self._prompt_label = ui.Label("", style=text, width=0)
                                self._response_label = ui.Label("", style=said, width=0)
                                ui.Spacer(height=10)
                            ui.Spacer(width=18)
                    ui.Spacer()
                ui.Spacer(height=self.config.bottom_margin_px)

    def _texts(self, e: int):
        from protomotions.envs.control.lerobot_recorder import language_texts

        goal = self.env.control_manager.components[self.config.goal_component]
        ball = self.env.control_manager.components.get(self.config.ball_component)
        source = getattr(ball, "command_source", None)
        colors = list(getattr(getattr(source, "config", None), "colors", None) or ["red"])
        prompt, truth = language_texts(
            goal, source, self._prompts, self._responses, colors,
            self.env.num_envs, self.env.device,
        )[e]
        vla_said: Optional[List[str]] = getattr(goal, "_vla_prev", None)
        if vla_said is not None:
            # The VLA was asked its own prompt (VlaGoalControl._prompts).
            prompt = goal._prompts()[e]
            return f'dog {e}:  "{prompt}"', f"VLA: {vla_said[e]}     (demonstrator: {truth})"
        return f'dog {e}:  "{prompt}"', truth

    def step(self) -> None:
        if self.env.simulator.headless or self._failed:
            return
        try:
            if self._frame is None:
                self._build()
            e = int(getattr(self.env.simulator, "_camera_target", {}).get("env", 0))
            shown = self._texts(e)
        except Exception as err:  # noqa: BLE001 -- say so once, keep the sim running
            print(f"[chase-caption] cannot draw the caption ({err!r}) -- no caption.", flush=True)
            self._failed = True
            return
        if shown != self._shown:
            self._prompt_label.text, self._response_label.text = shown
            self._shown = shown
