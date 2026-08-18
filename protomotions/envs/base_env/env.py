# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base environment implementation for reinforcement learning.

This module provides the foundational environment class for all RL tasks. It integrates
the simulator, handles robot state management, computes observations and rewards, manages
episode resets, and coordinates with terrain and scene systems.

Key Classes:
    - BaseEnv: Core environment class that all tasks inherit from

Key Features:
    - Multi-simulator support (IsaacGym, IsaacLab, Genesis)
    - Terrain integration for complex ground surfaces
    - Scene management for object interaction
    - Motion library integration for reference motions
    - Modular observation components

## BaseEnv

| Member | Type | Why Kept |
|--------|------|----------|
| `config` | `EnvConfig` | Core config, used everywhere |
| `robot_config` | `RobotConfig` | Core config, used everywhere |
| `device` | `torch.device` | Required for tensor creation |
| `terrain` | `Terrain` | Core dependency for terrain queries |
| `scene_lib` | `SceneLib` | Core dependency for scene/object handling |
| `motion_lib` | `MotionLib` | Core dependency for reference motions |
| `simulator` | `Simulator` | Core dependency for physics |
| `num_envs` | `int` | Frequently accessed, avoiding repeated `simulator.num_envs` |
| `max_episode_length` | `int` | Mutable - modified by agent for curriculum learning |
| `dt` | `float` | Frequently accessed, avoiding repeated `simulator.dt` |
| `rew_buf` | `Tensor` | Mutable buffer - accumulates rewards each step |
| `reset_buf` | `Tensor` | Mutable buffer - tracks which envs need reset |
| `progress_buf` | `Tensor` | Mutable buffer - tracks episode progress |
| `terminate_buf` | `Tensor` | Mutable buffer - tracks terminations |
| `extras` | `dict` | Mutable - collects per-step logging data |
| `respawn_root_offset` | `Tensor` | Mutable state - tracks spawn position offsets |
| `skip_height_correction` | `bool` | Performance optimization flag (read-only after init) |
| `motion_manager` | `MotionManager` | Core component for motion sampling |
| `motion_manager_disable_resample` | `bool` | Mutable flag - controlled by evaluator |
| `terrain_obs_cb` | `TerrainObs` | Observation component |
| `scene_obs_cb` | `SceneObs` | Observation component |

"""

from functools import cached_property
from typing import Any, Dict, Optional, TYPE_CHECKING, Tuple

import os
import torch
from torch import Tensor
from protomotions.utils.hydra_replacement import get_class

from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
    MarkerState,
)
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    ObjectState,
    ResetState,
)
from protomotions.envs.terminations import check_max_length_term
from protomotions.envs.context_views import (
    EnvContext,
    CurrentStateView,
    HistoricalView,
    TerrainContext,
    SceneSurfaceContext,
)
from protomotions.envs.obs.observation_noise import (
    NoisyObservations,
    apply_observation_noise,
    apply_reset_noise,
)
from protomotions.components.terrains.terrain import Terrain
from protomotions.envs.obs.scene_obs import SceneObs
from protomotions.envs.obs.terrain_obs import TerrainObs
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.control.manager import ControlManager

# Component infrastructure for MdpComponent-based configs
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.base_env.utils import (
    combine_rewards,
    combine_terminations,
)
from protomotions.components.pose_lib import build_body_ids_tensor

from protomotions.robot_configs.base import RobotConfig

if TYPE_CHECKING:
    from protomotions.components.scene_lib import SceneLib
    from protomotions.components.motion_lib import MotionLib


class BaseEnv:
    """Base class for all reinforcement learning environments.

    Provides core functionality for robot simulation including:
    - Simulator integration (IsaacGym, IsaacLab, Genesis)
    - Terrain management
    - Scene and object handling
    - Motion library integration
    - Observation and reward computation
    - Episode management and resets

    Subclasses should implement task-specific reward functions and
    observation spaces by overriding compute_reward() and compute_observations().

    Attributes:
        simulator: The physics simulator instance.
        num_envs: Number of parallel environments.
        device: PyTorch device for computations.
        terrain: Terrain instance for complex ground surfaces.
        scene_lib: Library of object scenes for interaction tasks.
        motion_lib: Library of reference motions for imitation tasks.

    Example:
        >>> config = SteeringEnvConfig()
        >>> robot_config = G1Config()
        >>> env = Steering(config, robot_config, simulator_config, device)
        >>> obs, _ = env.reset()
        >>> next_obs, rewards, dones, info = env.step(action_dict)
    """

    def __init__(
        self,
        config: EnvConfig,
        robot_config: RobotConfig,
        device: torch.device,
        terrain: "Terrain",
        simulator: Simulator,
        scene_lib: "SceneLib",
        motion_lib: "MotionLib",
        *args,
        **kwargs,
    ):
        """Initialize BaseEnv.

        Args:
            config: Environment configuration
            robot_config: Robot configuration
            device: Device for computation
            terrain: Pre-created Terrain object (always provided, can be None for visualizers)
            simulator: Pre-created Simulator shell (not yet initialized, will be initialized by env)
            scene_lib: Pre-created SceneLib (always provided, empty if no scenes)
            motion_lib: Pre-created MotionLib (always provided, empty if no motions)
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        self.config = config
        self.robot_config = robot_config
        self.device = device
        self.terrain = terrain
        self.scene_lib = scene_lib
        self.motion_lib = motion_lib
        self.simulator = simulator
        self.num_envs = simulator.num_envs

        self.max_episode_length = self.config.max_episode_length

        # Buffers
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.terminate_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        # Per-env flag: True while an env is in a random-orientation "get-up"
        # episode. These envs get an extended grace window (see
        # check_resets_and_terminations) so they have time to stand up AND
        # rejoin the reference before tracking-error termination can fire.
        self.is_getup_env = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        # Get-up training (AmpGetupEnv port). recovery_counter counts down the
        # steps of termination immunity an episode still has; the fall-state
        # bank is built lazily on first reset because it needs a live sim.
        self.recovery_counter = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._fall_states = None
        self._fall_reset_mask = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        # Manual (R key) resets must override get-up termination immunity.
        self._user_reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self.respawn_root_offset = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

        # Per-episode odometer corruption parameters.
        # Sampled once at episode reset; held constant within the episode.
        # Identity values (scale=1, yaw_bias=0) until first reset.
        self.odom_scale = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.odom_yaw_cos_sin = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )

        # Viewer/demo fast path: set True by inference_agent.py to skip
        # training-only per-step work (reward components, raw-state logging).
        # Leave False for training and evaluation.
        self.inference_mode = False
        self.odom_yaw_cos_sin[:, 0] = 1.0  # cos(0) = 1

        # Contact force tracking for impact penalty rewards
        # Initialized properly after simulator init when we know num_bodies
        self.prev_contact_force_magnitudes = None

        # Action buffers (current step only; previous actions come from
        # state_history). A multi-robot simulator takes PADDED action tensors
        # (max across its robots) — buffers must match its width.
        num_actions = getattr(
            self.simulator, "state_action_dim", robot_config.number_of_actions
        )
        self._current_raw_action = torch.zeros(
            self.num_envs, num_actions, dtype=torch.float, device=self.device
        )
        self._current_processed_action = torch.zeros(
            self.num_envs, num_actions, dtype=torch.float, device=self.device
        )

        # Inference/demo mode: skip reward computation in the step (rewards are
        # only needed for training). Set True by inference_agent.py to speed up
        # the viewer; leave False for training/evaluation.
        self.inference_mode = False

        # Global context cache - built once per step in post_physics_step
        # and reused by observations, rewards, and terminations
        self._current_context: Dict[str, Any] = None

        # Noisy observation cache - computed once in post_physics_step,
        # reused by both state_history and _build_global_context
        self._current_noisy_obs = None

        self.skip_height_correction = (
            self.config.skip_correct_terrain_height_on_flat and self.terrain.is_flat()
        )

        self.initialize_simulator()

    def initialize_simulator(self):
        """Initialize simulator with task-specific visualization markers.

        Called at the end of __init__ to finalize simulator setup after visualization
        markers have been created (potentially by child env class override).
        """
        if (
            hasattr(self.robot_config, "kinematic_info")
            and self.robot_config.kinematic_info is not None
        ):
            self.robot_config.kinematic_info.to(self.device)

        # Initialize contact force buffer now that we know num_bodies.
        # A multi-robot simulator pads state tensors to max(num_bodies/
        # num_dofs/actions) across its robots — state-shaped buffers must
        # match ITS widths (single-robot sims don't declare them).
        num_bodies = getattr(
            self.simulator, "state_num_bodies",
            self.robot_config.kinematic_info.num_bodies,
        )
        self.prev_contact_force_magnitudes = torch.zeros(
            self.num_envs, num_bodies, dtype=torch.float, device=self.device
        )

        if self.config.num_state_history_steps > 0:
            # Check if observation noise is configured - if so, allocate noisy buffers
            store_noisy = (
                self.simulator.config.domain_randomization is not None
                and self.simulator.config.domain_randomization.observation_noise
                is not None
                and self.simulator.config.domain_randomization.observation_noise.has_noise()
            )
            self.state_history = StateHistoryBuffer(
                num_envs=self.num_envs,
                num_history_steps=self.config.num_state_history_steps,
                num_bodies=num_bodies,
                num_dofs=getattr(
                    self.simulator, "state_num_dofs",
                    self.robot_config.kinematic_info.num_dofs,
                ),
                action_dim=getattr(
                    self.simulator, "state_action_dim",
                    self.robot_config.number_of_actions,
                ),
                num_contact_bodies=len(self.contact_body_ids),
                anchor_body_index=self.robot_config.anchor_body_index,
                device=self.device,
                store_noisy=store_noisy,
            )
        else:
            self.state_history = None

        if (
            self.motion_lib.num_motions() > 0
            and self.config.ref_contact_smooth_window > 0
        ):
            self.motion_lib.smooth_contacts(self.config.ref_contact_smooth_window)

        self.dt = self.simulator.dt

        if self.motion_lib.num_motions() > 0:
            self._validate_motion_lib_compatibility()
            self.create_motion_manager()
        else:
            self.motion_manager = None

        self.terrain_obs_cb = TerrainObs(self.terrain.config, self)
        self.scene_obs_cb = SceneObs(self.config.scene_obs, self)

        self._key_bindings = self.simulator.user_interface.scope("env")
        self._key_bindings.register("R", "reset", "Reset all environments")
        self.control_manager = ControlManager(self.config.control_components, self)

        visualization_markers = self.create_visualization_markers(
            self.simulator.headless
        )
        self.simulator._initialize_with_markers(visualization_markers)

        # Component infrastructure for MdpComponent
        self._component_manager = ComponentManager(self.device)
        self._observation_buffer: Dict[str, Tensor] = {}

        # Initialize observations
        self._initialize_observations()

    def _validate_motion_lib_compatibility(self):
        """Validate that the motion file is compatible with the robot config."""
        ki = self.robot_config.kinematic_info
        expected_dofs = ki.num_dofs
        expected_bodies = ki.num_bodies

        sample_state = self.motion_lib.get_motion_state(
            torch.zeros(1, dtype=torch.long, device=self.device),
            torch.zeros(1, device=self.device),
        )
        motion_dofs = sample_state.dof_pos.shape[1]
        motion_bodies = sample_state.rigid_body_pos.shape[1]

        if motion_dofs != expected_dofs or motion_bodies != expected_bodies:
            raise ValueError(
                f"\n{'=' * 70}\n"
                f"MOTION FILE / ROBOT MISMATCH\n"
                f"{'=' * 70}\n"
                f"Motion file has {motion_dofs} DOFs and {motion_bodies} bodies,\n"
                f"but robot '{type(self.robot_config).__name__}' expects "
                f"{expected_dofs} DOFs and {expected_bodies} bodies.\n\n"
                f"The motion file was likely generated for a different robot.\n"
                f"Make sure --motion-file matches the robot in your "
                f"checkpoint/config.\n"
                f"{'=' * 70}"
            )

    ###############################################################
    # Getters
    ###############################################################
    def is_simulation_running(self):
        """Check if the physics simulation is running.

        Returns:
            Boolean indicating simulation state
        """
        return self.simulator.is_simulation_running()

    def get_obs(self):
        """Gather observations from all components.

        Returns:
            Dictionary of observation tensors from humanoid, terrain, scene,
            and dynamic observation components
        """
        obs = {}
        terrain_obs = self.terrain_obs_cb.get_obs()
        obs.update(terrain_obs)
        if self.scene_lib.num_scenes() > 0 and self.config.scene_obs.enabled:
            scene_obs = self.scene_obs_cb.get_obs()
            obs.update(scene_obs)

        # Get dynamic observations
        dynamic_obs = {
            name: tensor.clone() for name, tensor in self._observation_buffer.items()
        }
        obs.update(dynamic_obs)

        if os.environ.get("PROTOMOTIONS_DEBUG_NONFINITE"):
            # Opt-in probe for new simulator backends: a NaN/Inf in the state
            # reaches the policy as a NaN action distribution many steps later,
            # where the traceback says nothing about its origin. Report the
            # first offending observation with the bodies/dofs involved.
            for name, tensor in obs.items():
                if not torch.is_floating_point(tensor):
                    continue
                bad = ~torch.isfinite(tensor)
                if bool(bad.any()):
                    rows = bad.any(dim=tuple(range(1, tensor.dim()))).nonzero().flatten()
                    cols = bad.any(dim=0).nonzero().flatten()
                    raise RuntimeError(
                        f"non-finite observation '{name}' at step "
                        f"{int(self.progress_buf.max())}: {int(bad.sum())} values "
                        f"in {rows.numel()} envs (first envs {rows[:5].tolist()}, "
                        f"first feature indices {cols[:10].tolist()}); "
                        f"finite range [{tensor[~bad].min():.3g}, "
                        f"{tensor[~bad].max():.3g}]"
                    )

        return obs

    def get_action_size(self):
        """Get the dimensionality of the action space.

        Returns:
            Number of action dimensions
        """
        return self.simulator.num_act

    def consume_reset_request(self) -> bool:
        """Return and consume a user-interface reset request."""
        return self._key_bindings.reset.consume()

    ###############################################################
    # Component Processing
    ###############################################################
    def _initialize_observations(self):
        """Initialize observation buffers."""
        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._process_observations(self.context, all_env_ids)

    def _process_observations(self, context: EnvContext, env_ids: Tensor):
        """Process observations using MdpComponent."""
        raw_obs = self._component_manager.execute_all(
            components=self.config.observation_components,
            ctx=context,
        )

        # Update observation buffer with results
        for name, obs_value in raw_obs.items():
            if name not in self._observation_buffer:
                self._observation_buffer[name] = torch.zeros(
                    self.num_envs,
                    obs_value.shape[-1],
                    dtype=obs_value.dtype,
                    device=self.device,
                )
            # MdpComponent always computes for all envs, update specified subset
            self._observation_buffer[name][env_ids] = obs_value[env_ids]

    def _process_rewards(
        self, context: EnvContext, grace_mask: Optional[Tensor] = None
    ):
        """Process rewards using MdpComponent."""
        raw_rewards = self._component_manager.execute_all(
            components=self.config.reward_components,
            ctx=context,
        )

        return combine_rewards(
            raw_rewards=raw_rewards,
            configs=self.config.reward_components,
            grace_mask=grace_mask,
            num_envs=self.num_envs,
            device=self.device,
        )

    def _process_terminations(self, context: EnvContext):
        """Process terminations using MdpComponent."""
        raw_terms = self._component_manager.execute_all(
            components=self.config.termination_components,
            ctx=context,
        )

        return combine_terminations(
            raw_terms=raw_terms,
            configs=self.config.termination_components,
            num_envs=self.num_envs,
            device=self.device,
        )

    _action_config_device_ready: bool = False

    def _process_action(self, action: Tensor, context: EnvContext) -> Dict[str, Tensor]:
        """Process action using single action config dict.

        action_config is a single dict with "fn" key and parameters.
        """
        if self.config.action_config is None:
            return {"processed_action": action}

        # Lazy device migration on first call
        if not self._action_config_device_ready:
            for key, val in self.config.action_config.items():
                if isinstance(val, torch.Tensor):
                    self.config.action_config[key] = val.to(action.device)
            self._action_config_device_ready = True

        fn = self.config.action_config["fn"]
        # Extract all params except "fn"
        params = {k: v for k, v in self.config.action_config.items() if k != "fn"}
        params["action"] = action
        return fn(**params)

    ###############################################################
    # Cached Properties
    ###############################################################
    @cached_property
    def contact_body_ids(self) -> torch.Tensor:
        """Body indices for contact sensing."""
        return build_body_ids_tensor(
            self.robot_config.kinematic_info.body_names,
            self.robot_config.contact_bodies,
            self.device,
        )

    @cached_property
    def non_termination_contact_body_ids(self) -> torch.Tensor:
        """Body indices that don't trigger termination on contact."""
        body_names = self.robot_config.kinematic_info.body_names
        if self.robot_config.non_termination_contact_bodies == "all":
            return build_body_ids_tensor(body_names, body_names, self.device)
        else:
            return build_body_ids_tensor(
                body_names,
                self.robot_config.non_termination_contact_bodies,
                self.device,
            )

    @cached_property
    def default_reset_state(self) -> ResetState:
        """Default robot reset state from simulator."""
        return self.simulator.get_default_robot_reset_state()

    @cached_property
    def default_object_state(self) -> ObjectState:
        """Default object state (empty if no scenes)."""
        return self.scene_lib.get_default_object_state(self.device)

    @cached_property
    def _motion_support_anchors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-motion support cell anchors from the terrain.

        Returns:
            anchor_xy: World (x, y) anchor per motion id [num_motions, 2]
            has_support: Whether the motion has a support cell [num_motions]
        """
        support_origins = getattr(self.terrain, "motion_support_origins", {})
        num_motions = self.motion_lib.num_motions()
        anchor_xy = torch.zeros((num_motions, 2), device=self.device)
        has_support = torch.zeros(
            num_motions, dtype=torch.bool, device=self.device
        )
        for motion_id, (anchor_x, anchor_y) in support_origins.items():
            if motion_id < num_motions:
                anchor_xy[motion_id, 0] = anchor_x
                anchor_xy[motion_id, 1] = anchor_y
                has_support[motion_id] = True
        return anchor_xy, has_support

    def update_respawn_root_offset_by_env_ids(
        self,
        env_ids,
        ref_state: Optional[RobotState] = None,
        sample_flat: bool = False,
    ) -> torch.Tensor:
        """
        Samples a new starting position for the environment.
        And obtains the root translation offset relative to the reference state.

        This method considers both scene and terrain requirements.

        When a scene is required for obj interaction,
        the character is spawned relative to the scene's position.

        For environments without a scene, a random valid coordinate is sampled,
        and non-negative vertical offset is added based on terrain height.

        During co-training, scene groups use flat terrain, but during
        inference the resolved terrain may be complex (with negative heights
        that get normalised).  Height correction is applied to both scene
        and non-scene envs unless the terrain is entirely flat.

        """

        respawn_offset = torch.zeros((len(env_ids), 3), device=self.device)

        # Get boolean masks for scene vs non-scene envs
        scene_mask, non_scene_mask = self.get_scene_non_scene_mask(env_ids)

        if scene_mask.any():
            scene_pos = self.scene_lib.get_scene_positions(self.terrain, self.device)
            respawn_offset[scene_mask, :2] = scene_pos[env_ids[scene_mask], :2]

            # Scene envs also need terrain height correction — the object
            # playground is flat at height-field 0, but terrain normalisation
            # (shifting min height to z=0) can raise the playground above
            # world z=0.  Without correction the agent spawns underground.
            if not self.skip_height_correction:
                if ref_state is not None:
                    rigid_body_pos = ref_state.rigid_body_pos[scene_mask].clone()
                    rigid_body_pos_spawned = rigid_body_pos + respawn_offset[
                        scene_mask
                    ].unsqueeze(1)
                else:
                    rigid_body_pos_spawned = respawn_offset[scene_mask].unsqueeze(1)

                terrain_heights = self.terrain.find_terrain_height_for_max_below_body(
                    rigid_body_pos_spawned
                )
                respawn_offset[scene_mask, 2] = terrain_heights

        # Motion-support envs: anchor the motion at its dedicated support cell
        # so the reference motion aligns exactly with the stamped support
        # geometry. The offset is the cell anchor itself (NOT offset by
        # ref_root) because the support boxes were stamped in motion-local
        # coordinates anchored there: world_ref = ref + anchor.
        # Only applies to reference resets (ref_state set), which run after
        # motion_manager.sample_motions() assigned fresh motion ids.
        support_mask = torch.zeros_like(non_scene_mask)
        support_origins = getattr(self.terrain, "motion_support_origins", {})
        if support_origins and ref_state is not None and self.motion_manager is not None:
            anchor_xy, has_support = self._motion_support_anchors
            motion_ids = self.motion_manager.motion_ids[env_ids]
            support_mask = non_scene_mask & has_support[motion_ids]
            if support_mask.any():
                respawn_offset[support_mask, :2] = anchor_xy[motion_ids[support_mask]]
                # Exclude support envs from random location sampling below
                non_scene_mask = non_scene_mask & ~support_mask

        if non_scene_mask.any():
            num_non_scene = non_scene_mask.sum().item()
            respawn_position_xy = self.terrain.sample_valid_locations(
                num_envs=num_non_scene,
                sample_flat=sample_flat,
                max_distance=getattr(self.config, "env_spacing", None),
            )

            if ref_state is None:
                ref_root = torch.zeros((num_non_scene, 2), device=self.device)
            else:
                ref_root = ref_state.root_pos[non_scene_mask, :2]
            respawn_offset[non_scene_mask, :2] = respawn_position_xy - ref_root

        if non_scene_mask.any() and not self.skip_height_correction:
            if ref_state is not None:
                rigid_body_pos = ref_state.rigid_body_pos[non_scene_mask].clone()
                rigid_body_pos_spawned = rigid_body_pos + respawn_offset[
                    non_scene_mask
                ].unsqueeze(1)
            else:
                rigid_body_pos_spawned = respawn_offset[non_scene_mask].unsqueeze(1)

            terrain_heights = self.terrain.find_terrain_height_for_max_below_body(
                rigid_body_pos_spawned
            )
            respawn_offset[non_scene_mask, 2] = terrain_heights

        if support_mask.any():
            # Support cells share the motion's ground frame: the cell floor is
            # flat and the boxes reproduce the clip's elevations, so the only
            # z offset needed is the cell floor's world height. The
            # max-below-body heuristic must NOT be used here — with matching
            # geometry every foot has near-zero clearance and a box-top
            # tie-break would lift the robot by a full box height.
            respawn_offset[support_mask, 2] = getattr(
                self.terrain, "motion_support_floor_z", 0.0
            )

        respawn_offset[:, 2] += self.config.ref_respawn_offset

        self.respawn_root_offset[env_ids] = respawn_offset

    def align_motion_with_humanoid(self, env_ids, root_pos):
        """Compute XY offset between humanoid spawn position and reference motion data.

        Args:
            env_ids: Environment indices to align
            root_pos: Desired root positions [len(env_ids), 3]
        """
        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids[env_ids],
            self.motion_manager.motion_times[env_ids],
        )

        self.respawn_root_offset[env_ids, :2] = (
            root_pos[:, :2] - ref_state.rigid_body_pos[:, 0, :2]
        )

    def get_spawn_to_ref_pose_offset_with_terrain_height_correction(
        self, target_pos: Tensor, env_ids: Optional[Tensor] = None
    ) -> Tensor:
        """Compute spawn offset with terrain height correction for reference poses.

        Used by motion tracking tasks to correctly position reference poses in the environment,
        accounting for both XY spawn offset and terrain height.

        Args:
            target_pos: Reference body positions [num_envs, num_bodies, 3]
                       without spawning offset applied.
            env_ids: Environment indices [num_envs]. If None, uses all envs.

        Returns:
            Offset to add to target_pos [num_envs, num_bodies, 3].

        Note:
            - For XY offset: all bodies share the same respawn_root_offset
            - For Z offset: all bodies share the same offset computed from
              the body furthest below terrain
            - This preserves the rigid body structure during spawning
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        new_offset = torch.zeros_like(target_pos)
        new_offset[:, :, :2] = self.respawn_root_offset[env_ids, :2][:, None, :]

        if not self.skip_height_correction:
            target_pos_spawned = target_pos.clone() + new_offset
            z_offset = self.terrain.find_terrain_height_for_max_below_body(
                target_pos_spawned
            )

            # Motions anchored in support cells already match the terrain
            # geometry — their z offset is the constant cell-floor height.
            # The max-below-body heuristic flips discontinuously when feet
            # cross box edges, which would bounce the whole reference pose.
            support_origins = getattr(self.terrain, "motion_support_origins", {})
            if support_origins and self.motion_manager is not None:
                _, has_support = self._motion_support_anchors
                support_mask = has_support[self.motion_manager.motion_ids[env_ids]]
                if support_mask.any():
                    z_offset = torch.where(
                        support_mask,
                        torch.full_like(
                            z_offset,
                            getattr(self.terrain, "motion_support_floor_z", 0.0),
                        ),
                        z_offset,
                    )

            new_offset[:, :, 2] = z_offset.unsqueeze(1)

        return new_offset

    def get_scene_non_scene_mask(self, env_ids):
        """
        Returns boolean masks indicating which envs require a scene and which don't.

        Args:
            env_ids: Environment IDs to check

        Returns:
            scene_mask: Boolean tensor (len(env_ids),) - True for scene envs
            non_scene_mask: Boolean tensor (len(env_ids),) - True for non-scene envs

        Note: For now assumes either all or none require a scene
        """
        num_envs = len(env_ids)
        if self.scene_lib.num_scenes() > 0:
            scene_mask = torch.ones(num_envs, device=self.device, dtype=torch.bool)
            non_scene_mask = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        else:
            scene_mask = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
            non_scene_mask = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        return scene_mask, non_scene_mask

    def get_markers_state(self):
        """Compute visualization marker positions for rendering.

        Returns:
            Dictionary mapping marker names to MarkerState objects
        """
        if self.simulator.headless:
            return {}

        markers_state = {}

        # Update terrain markers
        if self.config.show_terrain_markers:
            height_maps = self.terrain.get_height_maps(
                self.simulator.get_root_state(), None, return_all_dims=True
            ).view(self.num_envs, -1, 3)
            markers_state["terrain_markers"] = MarkerState(
                translation=height_maps,
                orientation=torch.zeros(
                    self.num_envs, height_maps.shape[1], 4, device=self.device
                ),
            )

        # Merge markers from control components
        control_markers_state = self.control_manager.get_markers_state()
        markers_state.update(control_markers_state)

        return markers_state

    ###############################################################
    # Environment step logic
    ###############################################################
    def step(self, action: Tensor):
        """Step the environment forward one timestep.

        Args:
            action: Raw action tensor from the policy [num_envs, num_actions]

        Returns:
            obs, rewards, dones, terminated, extras
        """
        self.extras = {}

        # Invalidate cached context - will be rebuilt after physics in post_physics_step
        self._current_context = None
        self._current_noisy_obs = None

        # Store current actions
        self._current_raw_action[:] = action

        # Process action
        action_dict = self._process_action(action, self.context)
        processed_action = action_dict["processed_action"]
        self._current_processed_action[:] = processed_action

        self.simulator.step(processed_action, markers_callback=self.get_markers_state)

        # Forensic tripwire, off unless PM_NAN_DEBUG=1 in the environment.
        # The tiger dies rarely (epoch 4984, ~7 h in) with NaN surfacing at
        # the policy's action distribution -- far downstream of the cause.
        # This catches the FIRST non-finite physics state at its source and
        # dumps who/when/why before the run dies.
        if getattr(self, "_nan_dbg_on", None) is None:
            import os as _os
            self._nan_dbg_on = bool(_os.environ.get("PM_NAN_DEBUG"))
        if self._nan_dbg_on:
            self._nan_debug_probe()

        self.post_physics_step()

        if self.consume_reset_request():
            self.user_reset()

        obs = self.get_obs()
        return obs, self.rew_buf, self.reset_buf, self.terminate_buf, self.extras

    def on_epoch_end(self, current_epoch: int):
        """Hook called at end of each training epoch. Override in subclasses if needed.

        Args:
            current_epoch: Current epoch number
        """
        pass

    def post_physics_step(self):
        """Update environment state after physics simulation step.

        Increments progress counter, updates motion manager, computes observations and rewards,
        checks for resets, and stores raw robot state in extras for logging.
        """
        self.progress_buf += 1
        self._update_recovery_count()

        if self.state_history is not None:
            current_state = self.simulator.get_robot_state()
            ground_heights = self.terrain.get_ground_heights(
                current_state.rigid_body_pos[:, 0]
            ).squeeze(-1)
            body_contacts = current_state.rigid_body_contacts[
                :, self.contact_body_ids
            ].bool()

            # Compute noisy versions if observation noise is configured and history stores noisy data
            noisy_kwargs = {}
            if self.state_history.store_noisy:
                obs_noise_cfg = (
                    self.simulator.config.domain_randomization.observation_noise
                )

                # Single source of truth: uniform noise via apply_observation_noise
                noisy = apply_observation_noise(
                    obs_noise_cfg=obs_noise_cfg,
                    robot_state=current_state,
                    anchor_idx=self.robot_config.anchor_body_index,
                    ground_heights=ground_heights,
                )
                self._current_noisy_obs = noisy

                # Extract noisy tensors for history buffer
                noisy_kwargs["noisy_rigid_body_pos"] = noisy.rigid_body_pos
                noisy_kwargs["noisy_rigid_body_rot"] = noisy.rigid_body_rot
                noisy_kwargs["noisy_rigid_body_vel"] = noisy.rigid_body_vel
                noisy_kwargs["noisy_rigid_body_ang_vel"] = noisy.rigid_body_ang_vel
                noisy_kwargs["noisy_dof_pos"] = noisy.dof_pos
                noisy_kwargs["noisy_dof_vel"] = noisy.dof_vel
                noisy_kwargs["noisy_ground_heights"] = noisy.ground_heights

            self.state_history.rotate_and_update(
                rigid_body_pos=current_state.rigid_body_pos,
                rigid_body_rot=current_state.rigid_body_rot,
                rigid_body_vel=current_state.rigid_body_vel,
                rigid_body_ang_vel=current_state.rigid_body_ang_vel,
                dof_pos=current_state.dof_pos,
                dof_vel=current_state.dof_vel,
                actions=self._current_raw_action,
                ground_heights=ground_heights,
                body_contacts=body_contacts,
                processed_actions=self._current_processed_action,
                **noisy_kwargs,
            )

        if self.motion_manager is not None and hasattr(
            self.motion_manager, "post_physics_step"
        ):
            self.motion_manager.post_physics_step()

        self.control_manager.step()

        if (
            self.motion_manager is not None
            and self.motion_manager.config.realign_motion_with_humanoid_on_each_step
        ):
            # When realign_motion_with_humanoid_on_each_step is True, we re-align before computing observations and rewards.
            # This ensures the robot only matches the local-pose with global orientation.
            self.align_motion_with_humanoid(
                torch.arange(self.num_envs, device=self.device, dtype=torch.long),
                self.simulator.get_root_state().root_pos,
            )

        # Build context once and reuse for observations, rewards, and terminations
        self._current_context = self._build_global_context()

        self.compute_observations(context=self._current_context)
        # Rewards are only needed for training; skip them in the viewer/demo to
        # avoid the per-step reward compute that tanks the inference frame rate.
        if not self.inference_mode:
            self.compute_reward(context=self._current_context)
        self.reset_buf[:], self.terminate_buf[:] = self.check_resets_and_terminations(
            context=self._current_context
        )

        self.extras["terminate"] = self.terminate_buf

        rbs: RobotState = self.simulator.get_robot_state()
        if not self.inference_mode:
            # Raw-state extras exist for training-time logging; flattening
            # every body tensor per step is wasted work in the viewer.
            for k, _ in rbs.get_shape_mapping(flattened=True).items():
                self.extras[f"raw/{k}"] = rbs.flatten_bodies(k)

        # Update previous contact forces for next step's impact penalty
        self.prev_contact_force_magnitudes[:] = torch.norm(
            rbs.rigid_body_contact_forces, dim=-1
        )

    def user_reset(self):
        """Force environments to reset on next check (triggered by user input)."""
        self.progress_buf[:] = 100000000000
        # Recorded explicitly: a get-up episode suppresses the max-episode
        # length reset for the duration of its recovery window, which would
        # otherwise swallow this too and leave the R key doing nothing.
        self._user_reset_buf[:] = True

    def compute_observations(self, env_ids=None, context: EnvContext = None):
        """Compute observations for specified environments.

        Args:
            env_ids: Environment indices to update (None = all environments)
            context: Pre-built EnvContext from self.context property.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if context is None:
            raise ValueError("context is required - use self.context to build it")

        # Process dynamic observations
        self._process_observations(context, env_ids)

        self.terrain_obs_cb.compute_observations(env_ids)
        if self.scene_lib.num_scenes() > 0:
            self.scene_obs_cb.compute_observations(env_ids)

    def check_resets_and_terminations(self, context: EnvContext):
        """Check reset and termination conditions.

        Only handles max episode length directly. All other terminations
        (including height/fall termination) should be configured via:
        - termination_components (dynamic termination system)
        - control_components (task-specific terminations)

        Args:
            context: Pre-built context from self.context property.

        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors
        """
        max_length_reached = check_max_length_term(
            self.progress_buf, self.max_episode_length
        )
        reset_buf = max_length_reached.clone()
        terminated = torch.zeros_like(self.reset_buf, dtype=torch.bool)

        # Get-up envs are a "recover to the reference pose" skill judged locally
        # (joint + gravity-relative up-axis match; see compute_reward /
        # _compute_getup_reward) on quantities a fallen robot can actually sense
        # (IMU + encoders) and reach. They are therefore NEVER terminated on the
        # absolute tracking error (which they can neither observe nor recover) --
        # they run to max-episode-length. Normal tracking envs are unaffected.
        keep = ~self.is_getup_env

        # An episode inside its recovery window is immune: it was deliberately
        # started on the floor (or continued from a crash), so every fall-based
        # termination would fire immediately and it would never get the chance
        # to stand up. Timeout is suppressed too -- see below.
        if self._getup_enabled():
            keep = keep & (self.recovery_counter <= 0)
            reset_buf = reset_buf & (self.recovery_counter <= 0)
            # ...but never swallow an explicit user reset.
            reset_buf = reset_buf | self._user_reset_buf

        comp_reset, comp_terminate = (
            self.control_manager.check_resets_and_terminations()
        )
        reset_buf = reset_buf | (comp_reset & keep)
        terminated = terminated | (comp_terminate & keep)

        # Process terminations
        comp_reset, comp_terminate, term_logging = self._process_terminations(context)
        reset_buf = reset_buf | (comp_reset & keep)
        terminated = terminated | (comp_terminate & keep)
        self.extras.update(term_logging)

        return reset_buf, terminated

    ###############################################################
    # Dynamic Reward System
    ###############################################################
    @property
    def context(self) -> EnvContext:
        """Get global context for observation/reward/termination evaluation.

        Returns cached context from _current_context if set (after post_physics_step),
        otherwise builds a fresh context.

        Returns:
            Typed EnvContext for observation/reward/termination functions.
        """
        if self._current_context is None:
            self._current_context = self._build_global_context()
        return self._current_context

    def _build_global_context(self) -> EnvContext:
        """Build a fresh global context for observations, rewards, and terminations.

        Creates typed EnvContext with view wrappers around existing data structures.
        Controllers populate their task-specific views via populate_context().

        When observation noise is configured:
        - noisy views have noise applied
        - current views contain clean data

        When no observation noise is configured:
        - Both point to the same tensors (memory efficient)

        Returns:
            Typed EnvContext for observation/reward/termination functions.
        """
        current_state = self.simulator.get_robot_state()
        anchor_idx = self.robot_config.anchor_body_index

        ground_heights = self.terrain.get_ground_heights(
            current_state.rigid_body_pos[:, 0]
        ).squeeze(-1)

        body_contacts = current_state.rigid_body_contacts[
            :, self.contact_body_ids
        ].bool()

        # Contact force magnitudes for impact penalty rewards
        current_contact_force_magnitudes = torch.norm(
            current_state.rigid_body_contact_forces, dim=-1
        )

        # Use cached noisy obs from post_physics_step when available.
        # During init/reset the cache is None — use clean (no-noise) fallback.
        if self._current_noisy_obs is not None:
            noisy = self._current_noisy_obs
        else:
            noisy = apply_observation_noise(
                obs_noise_cfg=None,
                robot_state=current_state,
                anchor_idx=anchor_idx,
                ground_heights=ground_heights,
            )

        scene_surface_context = self._build_scene_surface_context()

        # Observation kernels must see the EGO robot's widths; a multi-robot
        # simulator pads state to max(robot dims), so slice the views back.
        ego_nb = self.robot_config.kinematic_info.num_bodies
        ego_nd = self.robot_config.kinematic_info.num_dofs
        mb = ego_nb if getattr(
            self.simulator, "state_num_bodies", ego_nb) > ego_nb else None
        md = ego_nd if getattr(
            self.simulator, "state_num_dofs", ego_nd) > ego_nd else None

        # Build context with view wrappers
        ctx = EnvContext(
            # Core state views (wrap RobotState without copying)
            current=CurrentStateView(
                current_state, anchor_idx, max_bodies=mb, max_dofs=md
            ),
            noisy=CurrentStateView(noisy, anchor_idx, max_bodies=mb, max_dofs=md),
            # Historical views (wrap StateHistoryBuffer without copying)
            historical=HistoricalView(
                self.state_history, use_noisy=False, max_bodies=mb, max_dofs=md
            )
            if self.state_history
            else None,
            noisy_historical=HistoricalView(
                self.state_history, use_noisy=True, max_bodies=mb, max_dofs=md
            )
            if self.state_history
            else None,
            # Actions (historical)
            current_processed_action=self._current_processed_action,
            previous_action=self.state_history.actions[:, 1]
            if (self.state_history and self.state_history.num_history_steps >= 1)
            else None,
            previous_processed_action=self.state_history.processed_actions[:, 1]
            if (self.state_history and self.state_history.num_history_steps >= 1)
            else None,
            # Environment state
            ground_heights=ground_heights,
            noisy_ground_heights=noisy.ground_heights,
            terrain=TerrainContext(
                self.terrain.height_points,
                self.terrain.height_samples,
            ),
            scene=scene_surface_context,
            body_contacts=body_contacts,
            current_contact_force_magnitudes=current_contact_force_magnitudes,
            prev_contact_force_magnitudes=self.prev_contact_force_magnitudes,
            dt=self.dt,
            progress_buf=self.progress_buf,
            # Contact tracking
            contact_body_ids=self.contact_body_ids,
            non_termination_contact_body_ids=self.non_termination_contact_body_ids,
            # Per-episode odometer corruption parameters
            odom_scale=self.odom_scale,
            odom_yaw_cos_sin=self.odom_yaw_cos_sin,
        )

        # Controllers populate their task-specific views
        self.control_manager.populate_context(ctx)

        return ctx

    def _build_scene_surface_context(self) -> SceneSurfaceContext:
        """Build scene-object surface tensors for component observations.

        Nearest-surface observations bind these fields unconditionally. Envs
        without object pointclouds receive empty tensors, which lets the compute
        kernel naturally fall back to terrain-only behavior.
        """
        has_object_pointclouds = (
            getattr(self.scene_lib, "_object_pointclouds", None) is not None
        )
        if self.scene_lib.num_objects_per_scene <= 0 or not has_object_pointclouds:
            object_pos = torch.zeros(self.num_envs, 0, 3, device=self.device)
            object_rot = torch.zeros(self.num_envs, 0, 4, device=self.device)
            neutral_pointclouds = torch.zeros(
                self.num_envs, 0, 0, 3, device=self.device
            )
            object_valid_mask = torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )
            return SceneSurfaceContext(
                object_pos=object_pos,
                object_rot=object_rot,
                neutral_pointclouds=neutral_pointclouds,
                object_valid_mask=object_valid_mask,
            )

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        object_state = self.simulator.get_object_root_state()
        return SceneSurfaceContext(
            object_pos=object_state.root_pos,
            object_rot=object_state.root_rot,
            neutral_pointclouds=self.scene_lib.get_scene_neutral_pointcloud(env_ids),
            object_valid_mask=self.scene_lib.get_per_object_valid_mask(env_ids),
        )

    def get_has_reset_grace(self):
        """Check if environments are in the grace period after reset.

        Grace period is useful for zeroing rewards that are unreliable immediately
        after reset (e.g., power consumption, contact changes).

        Returns:
            Boolean tensor indicating which environments are within reset_grace_period steps of last reset.
            Returns None if reset_grace_period is 0 or negative.
        """
        if self.config.reset_grace_period <= 0:
            return None
        return self.progress_buf <= self.config.reset_grace_period

    def compute_reward(self, context: EnvContext):
        """Compute base rewards using the dynamic reward component system.

        Args:
            context: Pre-built EnvContext from self.context property.

        Subclasses should override this to add task-specific rewards, calling super().compute_reward() first.
        """
        grace_mask = self.get_has_reset_grace()

        # Process rewards
        combined_reward, reward_logging = self._process_rewards(context, grace_mask)

        # Get-up envs: override the (absolute) tracking reward with a local
        # recover-to-pose reward judged only on what a fallen robot can sense
        # (IMU + joint encoders) and reach -- joint-position match + the
        # reference's gravity-relative up-axis. Global position and yaw are
        # excluded (unobservable post-fall and unrecoverable in place).
        if bool(self.is_getup_env.any()):
            getup_r = self._compute_getup_reward(context)
            combined_reward = torch.where(self.is_getup_env, getup_r, combined_reward)

        self.rew_buf[:] = combined_reward
        self.extras.update(reward_logging)
        self.extras["total_env_reward"] = combined_reward

    def _compute_getup_reward(self, context: EnvContext):
        """Local get-up reward: match the reference joint configuration and the
        reference's gravity-relative orientation (up-axis), both IMU/encoder
        observable. Position- and yaw-free, so a fallen robot can actually earn
        it by standing up into the reference pose in place."""
        from protomotions.utils.rotations import quat_rotate_inverse

        cur_dof = context.current.dof_pos
        ref_dof = context.mimic.ref_state.dof_pos
        cur_rot = context.current.root_rot  # xyzw
        ai = self.robot_config.anchor_body_index
        ref_rot = context.mimic.ref_state.rigid_body_rot[:, ai]

        # joint-position match (encoders)
        dof_err = (cur_dof - ref_dof).abs().mean(dim=-1)

        # gravity-direction (IMU "projected gravity") match -- yaw-invariant
        gdir = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(
            cur_rot.shape[0], 3
        )
        cur_g = quat_rotate_inverse(cur_rot, gdir, True)
        ref_g = quat_rotate_inverse(ref_rot, gdir, True)
        orient_err = 1.0 - (cur_g * ref_g).sum(dim=-1).clamp(-1.0, 1.0)

        return 0.6 * torch.exp(-5.0 * dof_err) + 0.4 * torch.exp(-3.0 * orient_err)

    ###############################################################
    # Handle Resets
    ###############################################################
    def move_reset_robot_obj_states_to_respawn_position(
        self,
        env_ids,
        new_states: ResetState,
        new_object_states: ObjectState,
    ) -> Tuple[ResetState, ObjectState]:
        new_states.root_pos += self.respawn_root_offset[env_ids]
        if self.scene_lib.num_scenes() > 0:
            new_object_states.root_pos += self.respawn_root_offset[env_ids].unsqueeze(1)

        return new_states, new_object_states

    def compute_default_reset_state(
        self, env_ids, sample_flat: bool = False
    ) -> Tuple[ResetState, ObjectState]:
        """Reset environments to default state."""

        new_states = self.default_reset_state[env_ids].clone()
        new_object_states = self.default_object_state[env_ids].clone()

        self.update_respawn_root_offset_by_env_ids(
            env_ids,
            ref_state=None,
            sample_flat=sample_flat,
        )

        return self.move_reset_robot_obj_states_to_respawn_position(
            env_ids, new_states, new_object_states
        )

    def compute_ref_reset_state(
        self,
        env_ids,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        sample_flat: bool = False,
    ) -> Tuple[ResetState, ObjectState]:
        """Compute reset state from reference motion data.

        Args:
            env_ids: Environment indices to reset
            motion_ids: Motion IDs to use [len(env_ids)]
            motion_times: Start times for each motion [len(env_ids)]
            sample_flat: If True, spawn on flat terrain

        Returns:
            Tuple of (reset_state, object_reset_state)
        """

        ref_state = self.motion_lib.get_motion_state(motion_ids, motion_times)
        new_states = ResetState.from_robot_state(ref_state)

        # Clamp RSI-written dof velocities to each joint's ControlInfo.velocity_limit
        # (30 rad/s on the tiger). Jump clips can carry ~45 rad/s — above the
        # robot's own ceiling — and writing those into PhysX on a heavy
        # under-armatured animal is a known NaN precursor.
        if new_states.dof_vel is not None:
            ci = self.robot_config.control.control_info
            lims = []
            for name in self.robot_config.kinematic_info.dof_names:
                info = ci.get(name)
                vlim = None if info is None else info.velocity_limit
                lims.append(30.0 if vlim is None else float(vlim))
            lim = torch.as_tensor(
                lims, device=new_states.dof_vel.device, dtype=new_states.dof_vel.dtype
            )
            if lim.numel() == new_states.dof_vel.shape[-1]:
                new_states.dof_vel = new_states.dof_vel.clamp(min=-lim, max=lim)

        new_object_states = self.scene_lib.get_scene_pose(
            env_ids, motion_times, respawn_offset=self.config.ref_object_respawn_offset
        )
        new_object_states.root_vel = torch.zeros_like(new_object_states.root_pos)
        new_object_states.root_ang_vel = torch.zeros_like(new_object_states.root_pos)

        self.update_respawn_root_offset_by_env_ids(
            env_ids,
            ref_state=ref_state,
            sample_flat=sample_flat,
        )

        return self.move_reset_robot_obj_states_to_respawn_position(
            env_ids, new_states, new_object_states
        )

    def reset(
        self,
        env_ids=None,
        sample_flat=False,
        force_default_mask=None,
        disable_motion_resample=False,
    ):
        """Reset environments and return observations.

        - auto if no motion_lib: reset from default state
        - auto if motion_lib exists: reset from reference motion
        - force_default_mask: optional boolean mask [len(env_ids)] to force specific envs
            ref_prob = 0.5
            mask = torch.bernoulli(torch.full((len(env_ids),), 1-ref_prob)).bool()
            env.reset(env_ids, force_default_mask=mask)

        Args:
            env_ids: Environment IDs to reset, or None to reset all
            sample_flat: If True, spawn on flat terrain (useful for evaluation)
            force_default_mask: Optional boolean mask [len(env_ids)] to force specific envs
                               to use default reset instead of reference motion reset.
                               Only used if motion_lib exists.
            disable_motion_resample: If True, skip resampling motions (use existing motion_ids/times).
                               Useful for evaluation when you want to replay specific motions.

        Returns:
            obs: Dictionary of observation tensors
            info: Dictionary containing reset metadata (currently empty)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if len(env_ids) == 0:
            return self.get_obs(), {}

        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(self.device)

        # Start with default reset for all envs
        new_states, new_object_states = self.compute_default_reset_state(
            env_ids, sample_flat
        )

        # STEP 1: Reset motion manager and determine which envs need reference motion reset
        # This calls motion_manager.sample_motions() internally
        ref_env_ids, motion_ids, motion_times = self._get_ref_reset_envs(
            env_ids, force_default_mask, disable_motion_resample
        )

        # Overwrite ref envs with reference motion reset
        if len(ref_env_ids) > 0:
            ref_states, ref_object_states = self.compute_ref_reset_state(
                ref_env_ids, motion_ids, motion_times, sample_flat
            )

            ref_indices = torch.isin(env_ids, ref_env_ids).nonzero(as_tuple=True)[0]

            new_states[ref_indices] = ref_states
            new_object_states[ref_indices] = ref_object_states

        if self.robot_config.reset_noise is not None:
            apply_reset_noise(
                reset_state=new_states,
                config=self.robot_config.reset_noise,
                dof_limits_lower=self.robot_config.kinematic_info.dof_limits_lower,
                dof_limits_upper=self.robot_config.kinematic_info.dof_limits_upper,
            )

        # Random get-up resets: spawn some envs in random orientation + random joints
        if self.config.random_getup_prob > 0.0:
            self._apply_random_getup_reset(new_states, env_ids)

        # Get-up resets: start from a settled fallen pose, or continue from the
        # pose the robot terminated in. Both get termination immunity.
        getup_mask = None
        if self._getup_enabled():
            if self._fall_states is None and self.config.fall_init_prob > 0.0:
                self._generate_fall_states()
            getup_mask = self._apply_getup_resets(new_states, env_ids)

        self.simulator.reset_envs(new_states, new_object_states, env_ids)

        default_mask = ~torch.isin(env_ids, ref_env_ids)
        if getup_mask is not None and getup_mask.any():
            # A fallen robot has no reference frames behind it, so its history
            # is seeded by repeating the current pose (what the default path
            # does) rather than by querying the motion lib at t-dt, t-2dt...
            default_mask = default_mask | getup_mask
        if self.state_history is not None:
            self._reset_state_history(
                env_ids, default_mask, ref_env_ids, motion_ids, motion_times
            )

        # Reset control components after motion_manager has been reset
        self.control_manager.reset(env_ids)

        self.progress_buf[env_ids] = 0
        self._user_reset_buf[env_ids] = False
        self.reset_buf[env_ids] = False
        self.terminate_buf[env_ids] = False
        self.prev_contact_force_magnitudes[env_ids] = 0.0
        self._current_raw_action[env_ids] = 0.0
        self._current_processed_action[env_ids] = 0.0

        # Resample per-episode odometer corruption parameters.
        # These remain constant within an episode and are used by
        # corrupted_xy_offset_factory when present in observation components.
        n = len(env_ids)
        self.odom_scale[env_ids] = torch.empty(n, device=self.device).uniform_(
            self.config.odom_scale_range[0], self.config.odom_scale_range[1]
        )
        yaw_bias = torch.empty(n, device=self.device).uniform_(
            -self.config.odom_yaw_range_deg, self.config.odom_yaw_range_deg
        ) * (3.14159265358979 / 180.0)
        self.odom_yaw_cos_sin[env_ids, 0] = torch.cos(yaw_bias)
        self.odom_yaw_cos_sin[env_ids, 1] = torch.sin(yaw_bias)

        # Update cached noisy obs for the reset envs with fresh noise
        if self._current_noisy_obs is not None:
            current_state = self.simulator.get_robot_state()
            ground_heights = self.terrain.get_ground_heights(
                current_state.rigid_body_pos[env_ids, 0]
            ).squeeze(-1)
            obs_noise_cfg = self.simulator.config.domain_randomization.observation_noise
            noisy_subset = apply_observation_noise(
                obs_noise_cfg=obs_noise_cfg,
                robot_state=current_state,
                env_ids=env_ids,
                anchor_idx=self.robot_config.anchor_body_index,
                ground_heights=ground_heights,
            )
            self._current_noisy_obs.update_subset(env_ids, noisy_subset)

        # Recompute observations after reset to reflect new control component state
        # Invalidate and rebuild context since state changed
        self._current_context = None
        self.compute_observations(env_ids, context=self.context)

        return self.get_obs(), {}

    def _getup_enabled(self) -> bool:
        return (self.config.fall_init_prob > 0.0
                or self.config.recovery_episode_prob > 0.0)

    @torch.no_grad()
    def _generate_fall_states(self):
        """Build a bank of genuinely fallen poses by dropping the robot.

        This is the part that makes get-up training work. Sampling a random
        quaternion and random joint angles (see _apply_random_getup_reset)
        produces poses that are not physically reachable -- limbs pass through
        each other and nothing rests against the floor -- so the policy learns
        to stand up from configurations it will never actually be in.

        Instead every env is dropped from twice its standing height with a
        uniformly random orientation, holding a constant random PD target, and
        physics is stepped until it settles. Whatever it lands in IS a
        reachable fallen pose by construction. The bank holds one entry per
        env; resets sample from it.

        Velocities are stored as zero: a replayed fall starts at rest, not
        mid-tumble.
        """
        cfg = self.config
        num = self.num_envs
        all_ids = torch.arange(num, device=self.device)
        drop_h = float(getattr(self.robot_config, "default_root_height", 0.5))

        new_states, new_object_states = self.compute_default_reset_state(
            all_ids, sample_flat=True
        )
        # uniform on SO(3): normalized Gaussian 4-vector
        q = torch.randn(num, 4, device=self.device)
        new_states.root_rot = q / q.norm(dim=-1, keepdim=True)
        # ADD the drop height rather than scaling the reset height: scaling
        # would double the terrain offset too, dropping the robot from the
        # wrong altitude on any non-flat terrain.
        new_states.root_pos[:, 2] = new_states.root_pos[:, 2] + drop_h
        for attr in ("root_vel", "root_ang_vel", "dof_vel"):
            if getattr(new_states, attr, None) is not None:
                setattr(new_states, attr, torch.zeros_like(getattr(new_states, attr)))
        self.simulator.reset_envs(new_states, new_object_states, all_ids)

        # A constant random PD target during the drop, so limbs land in
        # varied configurations rather than all in the default pose.
        rand_action = (
            torch.rand(num, self.robot_config.number_of_actions, device=self.device)
            - 0.5
        )
        self._current_context = None
        processed = self._process_action(rand_action, self.context)["processed_action"]
        for _ in range(cfg.fall_state_settle_steps):
            self.simulator.step(processed)

        state = self.simulator.get_robot_state()
        low = state.rigid_body_pos[:, :, 2].min(dim=1).values

        # Store the root height RELATIVE TO THE ROBOT'S OWN LOWEST BODY, not
        # as an absolute z. Replay then places the root at
        # (local ground + this), so the lowest body always lands just above
        # the floor whatever the terrain. Storing absolute z instead let
        # poses that had sunk during the drop be replayed embedded in the
        # ground, which spawns the episode inside an explosive contact --
        # measured to -31.6 cm on the first run.
        #
        # Height is stored RELATIVE TO THE GROUND THE POSE SETTLED ON, not
        # relative to the robot's own lowest body.
        #
        # The lowest-body version was wrong and it is worth being precise
        # about why, because the same mistake bit fix_motion_ground.py.
        # rigid_body_pos holds joint CENTRES, and a collider reaches below
        # its origin by as much as its own extent -- 48.6 cm for the tiger's
        # RigRFLeg1, whose capsule hangs off one end. Placing the lowest
        # ORIGIN just above the floor therefore buries the colliders, and the
        # depenetration impulse produced
        #     RuntimeError: normal expects all elements of std >= 0.0
        # when the resulting NaN reached the policy's action distribution.
        # A fixed clearance margin cannot fix it either: large enough to
        # cover the worst collider means spawning half a metre in the air.
        #
        # Isolated by probe: fall_init_prob=0 ran past epoch 150, get-up
        # enabled died at 104.
        #
        # The settled pose is already resting correctly on the ground, so its
        # height above THAT ground is exactly right and needs no knowledge of
        # collider geometry at all. Replay adds the local ground height back.
        bank_ground = self.terrain.get_ground_heights(
            state.root_pos[:, :2]
        ).squeeze(-1)
        root_z_rel = state.root_pos[:, 2] - bank_ground

        # EVERY settled pose is banked -- no filtering. The settle time does
        # the work: measured on the raptor, 3 steps (0.2 s) leaves all 512
        # poses still in motion with roots up to 88 cm, while 50 steps
        # (3.3 s = 0.32 s of fall + ~3 s of settling) leaves 1 of 512 moving
        # and every root within 4-30 cm of its own lowest body.
        #
        # The handful that are still tumbling are deliberately KEPT. A robot
        # knocked down in a fight is mid-tumble, so recovering from one is a
        # skill worth having, and replay zeroes velocities anyway -- a
        # tumbling capture becomes a stationary robot in a tumbled pose,
        # which then falls naturally. Filtering them also meant needing a
        # "keep everything" fallback when nothing passed, which silently
        # banked mid-air poses on a misconfigured settle time.
        speed = state.root_vel.norm(dim=-1) if state.root_vel is not None \
            else torch.zeros(num, device=self.device)
        self._fall_states = {
            "root_z_rel": root_z_rel.clone(),
            "root_rot": state.root_rot.clone(),
            "dof_pos": state.dof_pos.clone(),
        }
        print(f"[getup] fall-state bank: {num} poses after "
              f"{cfg.fall_state_settle_steps} settle steps "
              f"({cfg.fall_state_settle_steps * 4 / 60:.1f} s); root sits "
              f"{root_z_rel.min()*100:.0f}-{root_z_rel.max()*100:.0f} cm above "
              f"the GROUND it settled on (standing is {drop_h*100:.0f} cm); "
              f"lowest body origin {(state.root_pos[:, 2] - low).min()*100:.0f}"
              f"-{(state.root_pos[:, 2] - low).max()*100:.0f} cm below the root; "
              f"{int((speed >= 0.5).sum())} still moving, "
              f"{int((root_z_rel >= 0.6 * drop_h).sum())} landed upright",
              flush=True)
        self._current_context = None

    def _apply_getup_resets(self, new_states, env_ids):
        """Three-way reset dispatch: recovery, fall-init, or normal.

        Returns a bool mask over env_ids marking envs whose state history must
        be seeded from their CURRENT (fallen) pose rather than from reference
        motion -- a fallen robot has no reference frames to look back on.
        """
        n = env_ids.shape[0]
        fall_mask = torch.zeros(n, device=self.device, dtype=torch.bool)
        self.recovery_counter[env_ids] = 0

        # 1. Recovery episodes CONTINUE from the pose the robot died in, so it
        #    practises recovering from its own failures and not only from
        #    sampled falls. Restricted to envs that actually terminated -- an
        #    env that merely timed out did not fall over.
        rec_mask = torch.zeros(n, device=self.device, dtype=torch.bool)
        if self.config.recovery_episode_prob > 0.0:
            draw = torch.rand(n, device=self.device) < self.config.recovery_episode_prob
            rec_mask = draw & self.terminate_buf[env_ids]
            if rec_mask.any():
                # reset_envs() overwrites state, so "keep the crash pose" means
                # writing the current pose back rather than skipping the write.
                cur = self.simulator.get_robot_state(env_ids[rec_mask])
                idx = rec_mask.nonzero(as_tuple=True)[0]
                new_states.root_pos[idx] = cur.root_pos
                new_states.root_rot[idx] = cur.root_rot
                new_states.dof_pos[idx] = cur.dof_pos
                for a in ("root_vel", "root_ang_vel", "dof_vel"):
                    if getattr(new_states, a, None) is not None:
                        getattr(new_states, a)[idx] = 0.0
                self.recovery_counter[env_ids[rec_mask]] = self.config.recovery_steps

        # 2. Fall-init: teleport to a settled pose from the bank.
        if self.config.fall_init_prob > 0.0 and self._fall_states is not None:
            draw = torch.rand(n, device=self.device) < self.config.fall_init_prob
            f_mask = draw & ~rec_mask
            if f_mask.any():
                idx = f_mask.nonzero(as_tuple=True)[0]
                pick = torch.randint(
                    0, self._fall_states["root_z_rel"].shape[0],
                    (idx.shape[0],), device=self.device)
                # Keep each env's own XY; the pose supplies orientation, joints
                # and the root's height ABOVE ITS OWN LOWEST BODY. Adding the
                # local ground height puts the robot on the floor wherever it
                # is, instead of at whatever absolute z it happened to settle
                # at in some other env.
                ground = self.terrain.get_ground_heights(
                    new_states.root_pos[idx, :2]
                ).squeeze(-1)
                new_states.root_pos[idx, 2] = (
                    ground + self._fall_states["root_z_rel"][pick]
                )
                new_states.root_rot[idx] = self._fall_states["root_rot"][pick]
                new_states.dof_pos[idx] = self._fall_states["dof_pos"][pick]
                for a in ("root_vel", "root_ang_vel", "dof_vel"):
                    if getattr(new_states, a, None) is not None:
                        getattr(new_states, a)[idx] = 0.0
                self.recovery_counter[env_ids[f_mask]] = self.config.recovery_steps
                fall_mask = f_mask

        self._fall_reset_mask[:] = False
        self._fall_reset_mask[env_ids[fall_mask | rec_mask]] = True
        return fall_mask | rec_mask

    def _update_recovery_count(self):
        if self._getup_enabled():
            self.recovery_counter = (self.recovery_counter - 1).clamp_min(0)

    def _nan_debug_probe(self):
        """Catch the first non-finite physics state at its source (PM_NAN_DEBUG=1).

        The failure this hunts: hours into training, `normal expects all
        elements of std >= 0.0` -- a NaN in the observations reaching the
        policy. By then the causal event is long gone. This probe runs right
        after every physics step, keeps a 64-step ring buffer of per-env
        extremes, and on the FIRST non-finite value dumps, for each offending
        env: which tensors/bodies/dofs went non-finite, the full previous
        (finite) state, the action, whether the episode was a fall-init
        (recovery_counter), time since reset, and which motion clip it was
        imitating. Then it stops the run so the dump reflects the first event,
        not the wreckage.
        """
        import os

        st = self.simulator.get_robot_state()
        dev = self.device
        if not hasattr(self, "_ndbg"):
            K, n = 64, self.num_envs
            self._ndbg = {
                "step": 0, "K": K,
                "dv_val": torch.zeros(K, n, device=dev),
                "dv_idx": torch.zeros(K, n, dtype=torch.long, device=dev),
                "bv_val": torch.zeros(K, n, device=dev),
                "bv_idx": torch.zeros(K, n, dtype=torch.long, device=dev),
                "root_z": torch.zeros(K, n, device=dev),
                "act": torch.zeros(K, n, device=dev),
                "prev": None,
            }
        S = self._ndbg
        k = S["step"] % S["K"]
        dv = st.dof_vel.abs()
        S["dv_val"][k], S["dv_idx"][k] = dv.max(dim=-1)
        bv = st.rigid_body_vel.norm(dim=-1)
        S["bv_val"][k], S["bv_idx"][k] = bv.max(dim=-1)
        S["root_z"][k] = st.root_pos[:, 2]
        S["act"][k] = self._current_processed_action.abs().max(dim=-1).values
        S["step"] += 1

        bad = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        parts = {}
        for name in ("dof_pos", "dof_vel", "root_pos", "root_rot",
                     "rigid_body_pos", "rigid_body_vel", "rigid_body_ang_vel"):
            t = getattr(st, name, None)
            if t is None:
                continue
            m = ~torch.isfinite(t)
            if m.any():
                envmask = m.reshape(m.shape[0], -1).any(dim=-1)
                parts[name] = envmask
                bad |= envmask
        if not bad.any():
            # snapshot survives to the dump as "the last finite state"
            S["prev"] = {
                "dof_pos": st.dof_pos.clone(), "dof_vel": st.dof_vel.clone(),
                "rigid_body_pos": st.rigid_body_pos.clone(),
                "rigid_body_vel": st.rigid_body_vel.clone(),
            }
            return

        ids = bad.nonzero(as_tuple=True)[0]
        dof_names = list(self.robot_config.control.control_info.keys())
        body_names = list(self.robot_config.kinematic_info.body_names)
        mm = getattr(self, "motion_manager", None)
        dump = {
            "probe_step": S["step"],
            "bad_envs": ids.cpu(),
            "nonfinite_fields": sorted(parts.keys()),
            "nonfinite_env_by_field": {kk: vv[ids].cpu() for kk, vv in parts.items()},
            "progress_buf": self.progress_buf[ids].cpu(),
            "recovery_counter": getattr(self, "recovery_counter", None) is not None
                and self.recovery_counter[ids].cpu() or None,
            "motion_ids": (mm is not None and hasattr(mm, "motion_ids"))
                and mm.motion_ids[ids].cpu() or None,
            "motion_times": (mm is not None and hasattr(mm, "motion_times"))
                and mm.motion_times[ids].cpu() or None,
            "history_ring_head": k,
            "history": {kk: S[kk][:, ids].cpu()
                        for kk in ("dv_val", "dv_idx", "bv_val", "bv_idx",
                                   "root_z", "act")},
            "prev_state": {kk: vv[ids].cpu()
                           for kk, vv in (S["prev"] or {}).items()},
            "cur_dof_pos": st.dof_pos[ids].cpu(),
            "cur_dof_vel": st.dof_vel[ids].cpu(),
            "cur_body_pos": st.rigid_body_pos[ids].cpu(),
            "cur_body_vel": st.rigid_body_vel[ids].cpu(),
            "raw_action": self._current_raw_action[ids].cpu(),
            "processed_action": self._current_processed_action[ids].cpu(),
            "dof_names": dof_names,
            "body_names": body_names,
        }
        out = os.path.abspath("nan_debug_dump.pt")
        torch.save(dump, out)
        e0 = int(ids[0])
        print(f"[NAN-DEBUG] non-finite physics state, {len(ids)} env(s): "
              f"{ids.tolist()[:8]}", flush=True)
        print(f"[NAN-DEBUG] fields: {sorted(parts.keys())}", flush=True)
        print(f"[NAN-DEBUG] env {e0}: progress={int(self.progress_buf[e0])} "
              f"recovery_counter="
              f"{int(self.recovery_counter[e0]) if hasattr(self, 'recovery_counter') else 'n/a'}",
              flush=True)
        if S["prev"] is not None:
            pv = S["prev"]["dof_vel"][e0].abs()
            top = pv.argmax()
            print(f"[NAN-DEBUG] env {e0} last finite step: max|dof_vel|="
                  f"{float(pv.max()):.1f} rad/s at dof "
                  f"{dof_names[int(top)] if int(top) < len(dof_names) else int(top)}",
                  flush=True)
        hist = S["dv_val"][:, e0]
        order = [(k - i) % S["K"] for i in range(min(10, S["step"]))]
        print(f"[NAN-DEBUG] env {e0} max|dof_vel| last 10 steps (newest first): "
              f"{[round(float(hist[j]), 1) for j in order]}", flush=True)
        print(f"[NAN-DEBUG] dump saved: {out}", flush=True)
        raise RuntimeError(
            "PM_NAN_DEBUG tripwire: non-finite physics state; see nan_debug_dump.pt")

    def _apply_random_getup_reset(self, new_states, env_ids):
        """Override a random fraction of envs with random orientation and joint positions.

        For each selected env: random root quaternion (any orientation) and joint positions
        sampled uniformly within DOF limits. Root position z is raised slightly so the robot
        doesn't start embedded in the ground.
        """
        num_resets = env_ids.shape[0]
        prob = self.config.random_getup_prob
        # Clear the get-up flag for all envs being reset; set it below only for
        # the subset that actually gets a random-orientation spawn.
        self.is_getup_env[env_ids] = False
        getup_mask = torch.rand(num_resets, device=self.device) < prob
        if not getup_mask.any():
            return

        getup_indices = getup_mask.nonzero(as_tuple=True)[0]
        self.is_getup_env[env_ids[getup_indices]] = True

        # Random quaternion via Gaussian → normalize (uniform on S3)
        rand_quat = torch.randn(len(getup_indices), 4, device=self.device)
        rand_quat = rand_quat / rand_quat.norm(dim=-1, keepdim=True)  # xyzw

        # Random DOF positions uniform within limits
        dof_low = self.robot_config.kinematic_info.dof_limits_lower.to(self.device)
        dof_high = self.robot_config.kinematic_info.dof_limits_upper.to(self.device)
        rand_dof = dof_low + torch.rand(len(getup_indices), dof_low.shape[0], device=self.device) * (dof_high - dof_low)

        new_states.root_rot[getup_indices] = rand_quat
        new_states.dof_pos[getup_indices] = rand_dof
        # Zero velocities so the robot starts stationary
        if new_states.root_vel is not None:
            new_states.root_vel[getup_indices] = 0.0
        if new_states.root_ang_vel is not None:
            new_states.root_ang_vel[getup_indices] = 0.0
        if new_states.dof_vel is not None:
            new_states.dof_vel[getup_indices] = 0.0

    def _get_ref_reset_envs(
        self, env_ids, force_default_mask, disable_motion_resample=False
    ):
        """Determine which envs should use reference motion reset and reset motion manager.

        This method is responsible for resetting the motion_manager by calling
        motion_manager.sample_motions(). Control components should be reset AFTER
        this method is called so they have access to fresh motion_ids and motion_times.

        Args:
            env_ids: Environment IDs to check
            force_default_mask: Boolean mask to force default reset
            disable_motion_resample: If True, use existing motion_ids/times instead of resampling

        Returns:
            ref_env_ids: Environments to reset with reference motion
            motion_ids: Motion IDs for ref resets (or None)
            motion_times: Motion times for ref resets (or None)
        """
        # No motions - no ref resets
        if self.motion_lib.num_motions() == 0:
            empty_ids = torch.tensor([], device=self.device, dtype=torch.long)
            return empty_ids, None, None

        if force_default_mask is not None:
            assert (
                len(force_default_mask) == len(env_ids)
            ), f"force_default_mask length {len(force_default_mask)} != env_ids length {len(env_ids)}"
            ref_env_ids = env_ids[~force_default_mask]
        else:
            ref_env_ids = env_ids

        if len(ref_env_ids) > 0:
            if not disable_motion_resample:
                self.motion_manager.sample_motions(ref_env_ids)
            motion_ids = self.motion_manager.motion_ids[ref_env_ids]
            motion_times = self.motion_manager.motion_times[ref_env_ids]
        else:
            motion_ids = None
            motion_times = None

        return ref_env_ids, motion_ids, motion_times

    def _reset_state_history(
        self,
        env_ids: Tensor,
        default_mask: Tensor,
        ref_env_ids: Tensor,
        motion_ids: Optional[Tensor],
        motion_times: Optional[Tensor],
    ):
        """Reset state history buffer for specified environments.

        For default reset: repeat current state across all history slots.
        For ref reset: query motion_lib at t-dt, t-2*dt, ... to get historical states.

        Args:
            env_ids: All environment indices being reset.
            default_mask: Boolean mask indicating which envs use default reset.
            ref_env_ids: Environment indices using reference motion reset.
            motion_ids: Motion IDs for ref envs (or None).
            motion_times: Motion times for ref envs (or None).
        """
        default_env_ids = env_ids[default_mask]
        num_history_steps = self.state_history.num_history_steps
        # Buffer stores current + history, so total slots = num_history_steps + 1
        buffer_size = num_history_steps + 1

        # Default reset: repeat current simulator state to all buffer slots
        if len(default_env_ids) > 0:
            current_state = self.simulator.get_robot_state()
            ground_heights = self.terrain.get_ground_heights(
                current_state.rigid_body_pos[default_env_ids, 0]
            ).squeeze(-1)
            body_contacts = current_state.rigid_body_contacts[default_env_ids][
                :, self.contact_body_ids
            ].bool()
            self.state_history.reset_from_single_state(
                env_ids=default_env_ids,
                rigid_body_pos=current_state.rigid_body_pos[default_env_ids],
                rigid_body_rot=current_state.rigid_body_rot[default_env_ids],
                rigid_body_vel=current_state.rigid_body_vel[default_env_ids],
                rigid_body_ang_vel=current_state.rigid_body_ang_vel[default_env_ids],
                dof_pos=current_state.dof_pos[default_env_ids],
                dof_vel=current_state.dof_vel[default_env_ids],
                ground_heights=ground_heights,
                body_contacts=body_contacts,
            )

        # Reference reset: fill buffer with current state at index 0 and historical states at index 1+
        # This ensures historical_* properties (which return [:, 1:]) give exactly num_history_steps elements
        if len(ref_env_ids) > 0 and motion_ids is not None and motion_times is not None:
            # motion_ids shape: [len(ref_env_ids)]
            # motion_times shape: [len(ref_env_ids)]
            num_ref_envs = len(ref_env_ids)

            # Create time offsets: [0, -dt, -2*dt, ..., -N*dt] for buffer_size slots
            # Index 0 = current (t), Index 1..N = historical (t-dt, t-2*dt, ..., t-N*dt)
            time_offsets = -self.dt * torch.arange(buffer_size, device=self.device)

            # Expand for batch query: [num_ref_envs, buffer_size]
            expanded_motion_ids = motion_ids.unsqueeze(1).expand(-1, buffer_size)
            expanded_motion_times = motion_times.unsqueeze(1) + time_offsets.unsqueeze(
                0
            )

            # Clamp times to valid range
            motion_lengths = self.motion_lib.motion_lengths[motion_ids]
            expanded_motion_times = expanded_motion_times.clamp(min=0.0)
            expanded_motion_times = torch.min(
                expanded_motion_times,
                motion_lengths.unsqueeze(1).expand(-1, buffer_size),
            )

            # Flatten for motion_lib query
            flat_motion_ids = expanded_motion_ids.reshape(-1)
            flat_motion_times = expanded_motion_times.reshape(-1)

            # Query motion library
            historical_state = self.motion_lib.get_motion_state(
                flat_motion_ids, flat_motion_times
            )

            # Motion library data is recorded on flat terrain (height = 0)
            # Only simulator-based states need terrain height queries
            historical_ground_heights = torch.zeros(
                num_ref_envs, buffer_size, device=self.device
            )

            # Get contacts from motion library if available, otherwise zeros
            if historical_state.rigid_body_contacts is not None:
                flat_contacts = historical_state.rigid_body_contacts[
                    :, self.contact_body_ids
                ].bool()
                historical_body_contacts = flat_contacts.view(
                    num_ref_envs, buffer_size, -1
                )
            else:
                historical_body_contacts = torch.zeros(
                    num_ref_envs,
                    buffer_size,
                    len(self.contact_body_ids),
                    dtype=torch.bool,
                    device=self.device,
                )

            # Reshape back to [num_ref_envs, buffer_size, ...]
            self.state_history.reset_from_states(
                env_ids=ref_env_ids,
                rigid_body_pos=historical_state.rigid_body_pos.view(
                    num_ref_envs, buffer_size, -1, 3
                ),
                rigid_body_rot=historical_state.rigid_body_rot.view(
                    num_ref_envs, buffer_size, -1, 4
                ),
                rigid_body_vel=historical_state.rigid_body_vel.view(
                    num_ref_envs, buffer_size, -1, 3
                ),
                rigid_body_ang_vel=historical_state.rigid_body_ang_vel.view(
                    num_ref_envs, buffer_size, -1, 3
                ),
                dof_pos=historical_state.dof_pos.view(num_ref_envs, buffer_size, -1),
                dof_vel=historical_state.dof_vel.view(num_ref_envs, buffer_size, -1),
                ground_heights=historical_ground_heights,
                body_contacts=historical_body_contacts,
                actions=None,  # Zero actions for historical reset
            )

    ###############################################################
    # Motion and Visualization Helpers
    ###############################################################
    def create_motion_manager(self):
        """Instantiate motion manager from configuration."""
        MotionManagerClass = get_class(self.config.motion_manager._target_)

        fixed_motion_ids = None
        if self.scene_lib.num_scenes() > 0:
            humanoid_motion_ids = self.scene_lib.get_humanoid_motion_ids()
            if humanoid_motion_ids is not None:
                fixed_motion_ids = torch.tensor(
                    humanoid_motion_ids, dtype=torch.long, device=self.device
                )

        self.motion_manager = MotionManagerClass(
            config=self.config.motion_manager,
            num_envs=self.num_envs,
            env_dt=self.dt,
            device=self.device,
            motion_lib=self.motion_lib,
            fixed_motion_ids_per_env=fixed_motion_ids,
        )

    def create_visualization_markers(self, headless: bool):
        """Create visualization markers based on headless flag.

        Args:
            headless: If True, no markers are created (empty dict).
                      If False, creates markers according to config.

        Returns:
            Dict of visualization markers.
        """
        if headless:
            return {}

        visualization_markers = {}

        if self.config.show_terrain_markers:
            terrain_markers = []
            for _ in range(self.terrain.num_height_points):
                terrain_markers.append(MarkerConfig(size="small"))
            terrain_markers_cfg = VisualizationMarkerConfig(
                type="sphere", color=(0.008, 0.345, 0.224), markers=terrain_markers
            )
            visualization_markers["terrain_markers"] = terrain_markers_cfg

        # Merge markers from control components
        control_markers = self.control_manager.create_visualization_markers(headless)
        visualization_markers.update(control_markers)

        return visualization_markers

    def get_state_dict(self):
        """Get environment state for checkpointing.

        Returns:
            Dictionary containing motion manager state
        """
        if self.motion_manager is not None:
            return {"motion_manager": self.motion_manager.get_state_dict()}
        return {}

    def load_state_dict(self, state_dict):
        """Load environment state from checkpoint.

        Args:
            state_dict: State dictionary from checkpoint
        """
        if self.motion_manager is not None:
            self.motion_manager.load_state_dict(state_dict["motion_manager"])

    def get_task_id(self):
        """Get task identifier for logging and checkpointing.

        Returns:
            String identifier (motion file name or 'null')
        """
        if self.motion_manager is not None:
            return self.motion_lib.motion_file.split("/")[-1]
        return "null"

    @staticmethod
    def apply_motion_weights_to_scene_weights(
        save_dir: Optional[str], motion_file: Optional[str], device: torch.device
    ) -> Optional[list]:
        """Apply motion weights from checkpoint as scene weights for curriculum learning.

        Loads motion weights from a previous training checkpoint and uses them as
        scene replication weights, allowing over-sampling of scenes corresponding to
        failed motions in curriculum learning.

        IMPORTANT: Assumes 1:1 correspondence between scenes and motions,
        where scene[i].humanoid_motion_id == i.

        Args:
            save_dir: Directory where checkpoints are saved (or None)
            motion_file: Motion file path to identify checkpoint (or None)
            device: PyTorch device

        Returns:
            List of scene weights from motion training or None if not available
        """
        from pathlib import Path

        if not save_dir or not motion_file:
            return None

        try:
            evaluated_motions = motion_file.split("/")[-1]
            checkpoint_path = Path(save_dir) / f"env_{evaluated_motions}.ckpt"

            if not checkpoint_path.exists():
                return None

            print(f"Loading motion weights from checkpoint: {checkpoint_path}")
            checkpoint_data = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )

            if "motion_manager" not in checkpoint_data:
                print(
                    "No motion_manager found in checkpoint, using uniform scene weights."
                )
                return None

            motion_weights = checkpoint_data["motion_manager"]["motion_weights"]
            print(f"Applying {len(motion_weights)} motion weights as scene weights")
            print(
                "WARNING: Assumes 1:1 scene-to-motion correspondence (scene[i].humanoid_motion_id == i)"
            )
            return motion_weights.cpu().tolist()

        except Exception as e:
            print(f"Error applying motion weights to scene weights: {e}")
            return None

    def save_state(self) -> dict:
        """Save all mutable env state for later restoration.

        Snapshots the current state of the environment including robot state,
        simulator state, progress/reset/terminate buffers, and state history.
        This is useful for temporarily interrupting normal training to run
        evaluation episodes, then restoring to continue training from where
        it left off.

        Returns:
            Dictionary containing cloned copies of all mutable state tensors
        """
        snapshot = {
            "robot_state": self.simulator.get_robot_state(),
            "markers_state": self.get_markers_state(),
            "actions": self.simulator.get_current_actions(),
            "progress_buf": self.progress_buf.clone(),
            "reset_buf": self.reset_buf.clone(),
            "terminate_buf": self.terminate_buf.clone(),
            "respawn_root_offset": self.respawn_root_offset.clone(),
            "odom_scale": self.odom_scale.clone(),
            "odom_yaw_cos_sin": self.odom_yaw_cos_sin.clone(),
        }
        if self.state_history is not None:
            snapshot["state_history"] = self.state_history.save_state()
        if self._current_noisy_obs is not None:
            from dataclasses import fields as dc_fields

            noisy = self._current_noisy_obs
            snapshot["_current_noisy_obs"] = NoisyObservations(
                **{f.name: getattr(noisy, f.name).clone() for f in dc_fields(noisy)}
            )
        if self.scene_lib.num_objects_per_scene > 0:
            snapshot["object_state"] = self.simulator.get_object_root_state()
        return snapshot

    def restore_state(self, snapshot: dict) -> None:
        """Restore env state from a previous save_state() snapshot.

        Restores all mutable state that was captured by save_state(),
        including robot positions/velocities, buffers, and state history.

        Args:
            snapshot: Dictionary from save_state() containing state tensors
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.simulator.reset_envs(
            snapshot["robot_state"], snapshot.get("object_state"), env_ids
        )

        if "state_history" in snapshot and self.state_history is not None:
            self.state_history.load_state(snapshot["state_history"])

        self.progress_buf.copy_(snapshot["progress_buf"])
        self.reset_buf.copy_(snapshot["reset_buf"])
        self.terminate_buf.copy_(snapshot["terminate_buf"])
        self.respawn_root_offset.copy_(snapshot["respawn_root_offset"])
        if "odom_scale" in snapshot:
            self.odom_scale.copy_(snapshot["odom_scale"])
            self.odom_yaw_cos_sin.copy_(snapshot["odom_yaw_cos_sin"])
        self._current_noisy_obs = snapshot.get("_current_noisy_obs")
        self._current_context = None

        # IsaacGym needs an extra step after state restore to sync internal state
        if "isaacgym" in self.simulator.config._target_.lower():
            self.simulator.step(snapshot["actions"], markers_callback=None)

    def close(self) -> None:
        """Release control-component and env-owned UI handles, then close
        the simulator. Safe to call multiple times."""
        control_manager = getattr(self, "control_manager", None)
        if control_manager is not None:
            for component in control_manager.components.values():
                component.close()

        ui = getattr(self, "_key_bindings", None)
        if ui is not None:
            ui.unregister_all()
            self._key_bindings = None

        simulator = getattr(self, "simulator", None)
        if simulator is not None:
            close = getattr(simulator, "close", None)
            if callable(close):
                close()
