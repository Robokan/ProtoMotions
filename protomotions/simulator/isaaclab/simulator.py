# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import torch

import isaaclab.sim as sim_utils

log = logging.getLogger(__name__)
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.markers import VisualizationMarkers as IsaacLabVisualizationMarkers
from isaaclab.markers import VisualizationMarkersCfg as IsaacLabVisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

try:
    # Isaac Lab 3: PhysX config moved out of isaaclab.sim
    from isaaclab_physx.physics import PhysxCfg

    _SIM_PHYSX_KW = "physics"
    _ISAACLAB_W_LAST = True  # Lab 3 uses xyzw
except ImportError:
    from isaaclab.sim import PhysxCfg

    _SIM_PHYSX_KW = "physx"
    _ISAACLAB_W_LAST = False  # Lab 2.x uses wxyz

from protomotions.components.terrains.terrain import Terrain
from protomotions.components.scene_lib import (
    SceneLib,
    MeshSceneObject,
    BoxSceneObject,
    SphereSceneObject,
    CylinderSceneObject,
)
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from protomotions.simulator.isaaclab.utils.scene import SceneCfg
from protomotions.simulator.isaaclab.config import (
    IsaacLabSimulatorConfig,
    ProtoMotionsIsaacLabMarkers,
)
from protomotions.simulator.isaaclab.utils.collision_baking import (
    ensure_baked_collision_usd,
)
from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import (
    MarkerState,
    VisualizationMarkerConfig,
    SimBodyOrdering,
    ProjectileConfig,
)
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    RootOnlyState,
    StateConversion,
    ObjectState,
    ResetState,
)


class IsaacLabSimulator(Simulator):
    config: IsaacLabSimulatorConfig

    # =====================================================
    # Group 1: Initialization & Configuration
    # =====================================================
    def __init__(
        self,
        config: IsaacLabSimulatorConfig,
        robot_config,
        terrain: Terrain,
        device: torch.device,
        simulation_app: Any,
        scene_lib: SceneLib,
        custom_key_handlers: Optional[Dict[str, callable]] = None,
    ) -> None:
        """
        Initialize the IsaacLabSimulator shell.

        Parameters:
            config (SimulatorConfig): The configuration dictionary.
            robot_config (RobotConfig): The robot configuration.
            terrain (Terrain): Terrain data for simulation.
            device (torch.device): Device to use for computation.
            simulation_app (Any): The simulation application instance.
            scene_lib (SceneLib): Scene library (always provided, can be empty).
        """
        # Lab 3 switched the public quat convention from wxyz to xyzw.
        # Must flip before Simulator.__init__ builds StateConversion.
        if _ISAACLAB_W_LAST:
            config.w_last = True

        super().__init__(
            config=config,
            robot_config=robot_config,
            scene_lib=scene_lib,
            terrain=terrain,
            device=device,
        )

        self._register_custom_user_interface_keys(custom_key_handlers or {})

        sim_kwargs = {
            "device": str(device),
            "dt": 1.0 / self.config.sim.fps,
            "render_interval": self.config.sim.decimation,
        }
        physics_backend = getattr(self.config, "physics_backend", "physx")
        if physics_backend in ("newton", "newton_mjwarp"):
            if not _ISAACLAB_W_LAST:
                raise RuntimeError(
                    "Isaac Lab Newton backend requires Isaac Lab 3 "
                    "(use .venv-isaacsim6)."
                )
            # Official Lab 3 switch: SimulationCfg(physics=NewtonCfg()).
            # ProtoMotions terrain is a trimesh plane; MuJoCo contact gen
            # rejects that mesh. Newton's own pipeline handles it.
            from isaaclab_newton.physics import (
                MJWarpSolverCfg,
                NewtonCfg,
                NewtonShapeCfg,
            )

            sim_kwargs["physics"] = NewtonCfg(
                solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False),
                default_shape_cfg=NewtonShapeCfg(margin=0.01),
            )
            log.info(
                "Isaac Lab physics backend: Newton (MJWarp, Newton contacts)"
            )
        else:
            sim_kwargs[_SIM_PHYSX_KW] = PhysxCfg(
                solver_type=self.config.sim.physx.solver_type,
                max_position_iteration_count=self.config.sim.physx.num_position_iterations,
                max_velocity_iteration_count=self.config.sim.physx.num_velocity_iterations,
                bounce_threshold_velocity=self.config.sim.physx.bounce_threshold_velocity,
                gpu_max_rigid_contact_count=self.config.sim.physx.gpu_max_rigid_contact_count,
                gpu_found_lost_pairs_capacity=self.config.sim.physx.gpu_found_lost_pairs_capacity,
                gpu_found_lost_aggregate_pairs_capacity=self.config.sim.physx.gpu_found_lost_aggregate_pairs_capacity,
                gpu_max_rigid_patch_count=self.config.sim.physx.gpu_max_rigid_patch_count,
            )
            log.info("Isaac Lab physics backend: PhysX")
        # Lab 3 Kit viewport needs an explicit visualizer + camera. Without
        # this the window opens but looks at an empty default view.
        # Headless Kit has no omni.kit.viewport — skip or SimulationContext
        # raises during visualizer init.
        if _ISAACLAB_W_LAST and not self.headless:
            try:
                from isaaclab_visualizers.kit import KitVisualizerCfg

                sim_kwargs["visualizer_cfgs"] = [
                    KitVisualizerCfg(
                        eye=(3.0, -5.0, 2.0),
                        lookat=(0.0, 0.0, 0.6),
                    )
                ]
            except ImportError:
                pass

        sim_cfg = sim_utils.SimulationCfg(**sim_kwargs)
        self._simulation_app = simulation_app
        self._sim = SimulationContext(sim_cfg)
        self._sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

        if _ISAACLAB_W_LAST and not self.headless:
            import omni.kit.app

            omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
                "omni.replicator.core", True
            )

        # Scene construction below needs _proj_config before _init_projectiles runs
        self._resolve_proj_config()

        scene_cfg = self._get_scene_cfg()

        self._scene = InteractiveScene(scene_cfg)
        if not self.headless:
            self._setup_keyboard()
        print("[INFO]: Setup complete...")

    def _create_simulation(self) -> None:
        """Create the IsaacLab simulation environment.

        Called by base class _initialize_with_markers() after visualization markers
        are set. Completes scene setup and resets simulation.
        """
        self._robot = self._scene["robot"]
        # Build a mapping from body name to contact sensor (if it exists)
        self._contact_sensor_map = {}
        for body_name in self._body_names:
            if f"contact_sensor_{body_name}" in self._scene.keys():
                self._contact_sensor_map[body_name] = self._scene[
                    f"contact_sensor_{body_name}"
                ]

        self._object = []
        self._object_contact_sensor = []
        if self.scene_lib.num_scenes() > 0:
            for obj_idx in range(self.scene_lib.num_objects_per_scene):
                self._object.append(self._scene[f"object_{obj_idx}"])
                if f"object_{obj_idx}_contact_sensor" in self._scene.keys():
                    self._object_contact_sensor.append(
                        self._scene[f"object_{obj_idx}_contact_sensor"]
                    )
                else:
                    self._object_contact_sensor.append(None)
        # Retrieve projectile rigid objects from scene
        self._projectile_objects = []
        for proj_idx in range(self._proj_config.num_projectiles):
            self._projectile_objects.append(self._scene[f"projectile_{proj_idx}"])

        if self._visualization_markers:
            self._build_markers(self._visualization_markers)
        self._sim.reset()
        # Lab 3 fabric/Kit does not show articulations until state is flushed
        # and kinematics are forwarded. Lab 2 forward() is a no-op-safe extra.
        self._scene.write_data_to_sim()
        self._sim.forward()

    def _get_scene_cfg(self) -> SceneCfg:
        """
        Construct and return the scene configuration from the current config, scene library, and terrain.

        Returns:
            SceneCfg: The constructed scene configuration.
        """
        scene_cfgs = None
        if self.scene_lib.num_scenes() > 0:
            scene_cfgs, self._initial_scene_pos = self._preprocess_object_playground()

        scene_cfg = SceneCfg(
            config=self.config,
            robot_config=self.robot_config,
            num_envs=self.config.num_envs,
            env_spacing=getattr(self.config, "env_spacing", 2.0),
            scene_cfgs=scene_cfgs,
            terrain=self.terrain,
            projectile_config=self._proj_config,
            # Physics replication must be off when envs differ: scene objects,
            # or a distinct opponent-half robot USD (viewer exhibitions).
            replicate_physics=(
                scene_cfgs is None
                and getattr(
                    self.robot_config.asset, "opponent_usd_asset_file_name", None
                )
                is None
            ),
            filter_collisions=self.config.filter_env_collisions,
        )
        return scene_cfg

    def _preprocess_object_playground(self) -> Tuple[List[Any], torch.Tensor]:
        """
        Process and build the object playground from the scene library.

        Returns:
            Tuple[List[Any], torch.Tensor]: A tuple containing the object configurations and the initial object positions.
        """
        print("=========== Building object playground")

        self._baked_path_cache: Dict[str, Path] = {}

        # Spawn objects at origin (actual positions set via reset_envs later)
        initial_obj_pos = torch.zeros(
            (self.num_envs, self.scene_lib.num_objects_per_scene, 7),
            device=self.device,
            dtype=torch.float,
        )
        # Identity quaternion in the active Isaac Lab convention.
        # Lab 2 is wxyz (index 3); Lab 3 is xyzw (index 6).
        initial_obj_pos[..., 6 if _ISAACLAB_W_LAST else 3] = 1.0

        # Build object configurations for IsaacLab
        objects_cfgs = []
        for _ in range(self.scene_lib.num_objects_per_scene):
            objects_cfgs.append([])

        for env_id, scene in enumerate(self.scene_lib.scenes):
            for obj_idx, obj in enumerate(scene.objects):
                object_options = self._get_object_options_for_randomized_asset(
                    obj, env_id=env_id
                )
                # Common properties based on object options
                rigid_props = sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=object_options.fix_base_link,
                )
                collision_props = sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002,
                    rest_offset=0.0,
                )

                # Resolve color: use object option if set, else per-type default
                obj_color = (
                    object_options.color if object_options.color is not None else None
                )

                # Handle different object types
                if isinstance(obj, MeshSceneObject):
                    main_dir_path = (
                        f"{os.path.dirname(os.path.abspath(__file__))}/../../../"
                    )
                    asset_path = Path(
                        os.path.join(main_dir_path, obj.object_path)
                    ).resolve()
                    mass_props = self._mass_props_from_options(object_options)

                    # Pre-bake collision approximation into the USD asset
                    approx = self.scene_lib.config.mesh_collision_approximation
                    if approx is not None:
                        cache_key = str(asset_path)
                        if cache_key not in self._baked_path_cache:
                            self._baked_path_cache[cache_key] = (
                                ensure_baked_collision_usd(
                                    original_path=cache_key,
                                    approximation=approx,
                                    max_convex_hulls=self.scene_lib.config.mesh_collision_max_convex_hulls,
                                    hull_vertex_limit=self.scene_lib.config.mesh_collision_hull_vertex_limit,
                                    voxel_resolution=self.scene_lib.config.mesh_collision_voxel_resolution,
                                )
                            )
                        asset_path = self._baked_path_cache[cache_key]

                    spawn_cfg = sim_utils.UsdFileCfg(
                        usd_path=str(asset_path),
                        scale=obj.scale,
                        rigid_props=rigid_props,
                        mass_props=mass_props,
                        collision_props=collision_props,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=obj_color or (0.2, 0.7, 0.3),
                            metallic=0.2,
                        ),
                    )
                elif isinstance(obj, BoxSceneObject):
                    mass_props = self._mass_props_from_options(object_options)
                    spawn_cfg = sim_utils.CuboidCfg(
                        size=(obj.width, obj.depth, obj.height),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=obj_color or (0.8, 0.3, 0.3),
                            metallic=0.2,
                        ),
                        rigid_props=rigid_props,
                        mass_props=mass_props,
                        collision_props=collision_props,
                    )
                elif isinstance(obj, SphereSceneObject):
                    mass_props = self._mass_props_from_options(object_options)
                    spawn_cfg = sim_utils.SphereCfg(
                        radius=obj.radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=obj_color or (0.3, 0.3, 0.8),
                            metallic=0.2,
                        ),
                        rigid_props=rigid_props,
                        mass_props=mass_props,
                        collision_props=collision_props,
                    )
                elif isinstance(obj, CylinderSceneObject):
                    mass_props = self._mass_props_from_options(object_options)
                    spawn_cfg = sim_utils.CylinderCfg(
                        radius=obj.radius,
                        height=obj.height,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=obj_color or (0.3, 0.8, 0.3),
                            metallic=0.2,
                        ),
                        rigid_props=rigid_props,
                        mass_props=mass_props,
                        collision_props=collision_props,
                    )
                else:
                    raise ValueError(f"Unsupported object type: {type(obj)}")

                objects_cfgs[obj_idx].append(spawn_cfg)

        return objects_cfgs, initial_obj_pos

    @staticmethod
    def _mass_props_from_options(options):
        """Build ``MassPropertiesCfg`` from :class:`ObjectOptions`.

        If ``options.mass`` is set, use explicit mass (density disabled).
        Otherwise use ``options.density`` (always set by ObjectOptions).
        """
        if options.mass is not None:
            return sim_utils.MassPropertiesCfg(mass=options.mass, density=-1)
        return sim_utils.MassPropertiesCfg(mass=-1, density=options.density)

    def _ensure_omni_appwindow(self) -> bool:
        """Make ``omni.appwindow`` importable.

        Isaac Sim 6 keeps it as a Kit extension. ``import omni`` does not
        load it, and Lab 3's experience file does not always enable it
        before scene construction.
        """
        try:
            import omni.appwindow  # noqa: F401

            return True
        except ImportError:
            pass
        try:
            import omni.kit.app

            ext_manager = omni.kit.app.get_app().get_extension_manager()
            if not ext_manager.is_extension_enabled("omni.appwindow"):
                ext_manager.set_extension_enabled_immediate("omni.appwindow", True)
            app = getattr(self, "_simulation_app", None)
            if app is not None:
                app.update()
            import omni.appwindow  # noqa: F401

            return True
        except Exception as exc:
            log.warning("Could not enable omni.appwindow: %s", exc)
            return False

    def _create_keyboard_interface(self):
        """Build an Isaac Lab Se2Keyboard, across Lab 2.x and 3.x APIs."""
        from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard

        self._ensure_omni_appwindow()

        try:
            from isaaclab.devices.keyboard.se2_keyboard_cfg import Se2KeyboardCfg

            return Se2Keyboard(cfg=Se2KeyboardCfg(sim_device=str(self.device)))
        except Exception:
            pass

        try:
            from dataclasses import dataclass

            @dataclass
            class Se2KeyboardCfg:
                v_x_sensitivity: float = 0.8
                v_y_sensitivity: float = 0.4
                omega_z_sensitivity: float = 1.0
                sim_device: str = "cuda:0"

            return Se2Keyboard(cfg=Se2KeyboardCfg())
        except Exception:
            pass

        try:
            return Se2Keyboard()
        except TypeError:
            return Se2Keyboard(
                v_x_sensitivity=0.8, v_y_sensitivity=0.4, omega_z_sensitivity=1.0
            )

    def _setup_keyboard(self) -> None:
        """Set up keyboard callbacks for control using the Se2Keyboard interface."""
        try:
            self.keyboard_interface = self._create_keyboard_interface()
        except Exception as exc:
            log.warning(
                "IsaacLab keyboard setup failed (%s). Viewer will run without hotkeys.",
                exc,
            )
            self.keyboard_interface = None
            return

        self.user_interface.add_registration_callback(
            self._register_user_interface_key_callback,
            replay_existing=True,
        )
        # Camera-env cycling (record.py) is only wired for the IsaacLab viewer.
        self.user_interface.register_key(
            "EQUAL",
            owner="simulator",
            description="Focus camera on next environment",
            on_press=self._next_camera_env,
        )
        self.user_interface.register_key(
            "MINUS",
            owner="simulator",
            description="Focus camera on previous environment",
            on_press=self._prev_camera_env,
        )

    def _register_custom_user_interface_keys(self, handlers: Dict[str, callable]) -> None:
        for key_name, handler in handlers.items():
            self.user_interface.register_key(
                key_name,
                owner="simulator.custom",
                description=f"Custom simulator key handler for {key_name}",
                on_press=handler,
            )

    def _register_user_interface_key_callback(self, handle) -> None:
        key_name = handle.key
        callback_key = self._isaaclab_callback_key(key_name)

        release_callback = getattr(
            self.keyboard_interface, "add_release_callback", None
        )

        def callback(*args, key_name=key_name, pressed=True):
            has_backend_state = bool(args) or release_callback is not None
            if args:
                pressed = self._isaaclab_event_is_pressed(args[0], default=pressed)
            self.user_interface.handle_key_event(key_name, pressed=pressed)
            if pressed and not has_backend_state:
                # Older IsaacLab keyboard callbacks are press-only. Pulse-release so
                # one-shot bindings work without leaving KeyBinding.down() stuck.
                self.user_interface.handle_key_event(key_name, pressed=False)

        try:
            self.keyboard_interface.add_callback(callback_key, callback)
            if release_callback is not None:
                release_callback(
                    callback_key,
                    lambda *args, key_name=key_name: callback(
                        *args, key_name=key_name, pressed=False
                    ),
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to register IsaacLab key '{key_name}' "
                f"as '{callback_key}'"
            ) from e

    @staticmethod
    def _isaaclab_event_is_pressed(event, *, default: bool) -> bool:
        event_type = getattr(event, "type", None)
        if event_type is not None:
            name = getattr(event_type, "name", str(event_type)).upper()
            if "RELEASE" in name:
                return False
            if "PRESS" in name:
                return True
        value = getattr(event, "value", None)
        if value is not None:
            return bool(value)
        if isinstance(event, bool):
            return event
        return default

    @staticmethod
    def _isaaclab_callback_key(key_name: str) -> str:
        if len(key_name) == 1 and key_name.isdigit():
            return f"NUMPAD_{key_name}"
        return key_name

    # =====================================================
    # Group 2: Environment Setup & Configuration
    # =====================================================
    def _finalize_setup(self) -> None:
        """
        Configure initial environment settings when the simulation is ready.
        This includes setting up joint limits and initializing state tensors.
        """
        super()._finalize_setup()

        # Update initial object positions
        if self.scene_lib.num_scenes() > 0:
            objects_start_pos = torch.zeros(
                (self.num_envs, 13), device=self.device, dtype=torch.float
            )
            for obj_idx, object in enumerate(self._object):
                objects_start_pos[:, :7] = self._initial_scene_pos[:, obj_idx, :]
                object.write_root_state_to_sim(objects_start_pos)

        self._apply_domain_randomization_if_needed()

    def _apply_domain_randomization_if_needed(self) -> None:
        all_env_ids = torch.arange(self.config.num_envs, dtype=torch.int)
        if (
            self._domain_randomization is not None
            and "friction" in self._domain_randomization
        ):
            if not hasattr(self._robot, "root_physx_view"):
                log.warning(
                    "Skipping friction domain randomization: no PhysX view "
                    "(Isaac Lab Newton backend)."
                )
            else:
                self._apply_physx_friction_randomization()

        if (
            self._domain_randomization is not None
            and "center_of_mass" in self._domain_randomization
        ):
            if not hasattr(self._robot, "root_physx_view"):
                log.warning(
                    "Skipping COM domain randomization: no PhysX view "
                    "(Isaac Lab Newton backend)."
                )
            else:
                self._apply_physx_com_randomization(all_env_ids)

        self._apply_scene_object_properties_after_spawn(all_env_ids)

    def _apply_physx_friction_randomization(self) -> None:
        # Adapted from https://github.com/isaac-sim/IsaacLab/blob/be083bf1f70466e1d41bf9ffdc405bb89394e92c/source/isaaclab/isaaclab/envs/mdp/events.py#L203
        all_env_ids = torch.arange(self.config.num_envs, dtype=torch.int)
        num_shapes_per_body = []
        for link_path in self._robot.root_physx_view.link_paths[0]:
            link_physx_view = self._robot._physics_sim_view.create_rigid_body_view(
                link_path
            )
            num_shapes_per_body.append(link_physx_view.max_shapes)
        # ensure the parsing is correct
        num_shapes = sum(num_shapes_per_body)
        expected_shapes = self._robot.root_physx_view.max_shapes
        if num_shapes != expected_shapes:
            raise ValueError(
                "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
            )

        materials = self._as_torch(
            self._robot.root_physx_view.get_material_properties()
        )
        body_names = [
            self.robot_config.kinematic_info.body_names[
                self._domain_randomization["friction"]["body_indices"][idx]
            ]
            for idx in range(
                len(self._domain_randomization["friction"]["body_indices"])
            )
        ]
        isaaclab_body_ids, _ = self._robot.find_bodies(
            body_names, preserve_order=True
        )
        for idx in range(
            len(self._domain_randomization["friction"]["body_indices"])
        ):
            # bodies may span multiple "shapes" in the physx view, so we need to assign the materials to the correct shapes
            start_idx = sum(num_shapes_per_body[: isaaclab_body_ids[idx]])
            end_idx = start_idx + num_shapes_per_body[isaaclab_body_ids[idx]]

            num_buckets = self._domain_randomization["friction"][
                "static_friction"
            ].shape[0]
            bucket_ids = torch.randint(0, num_buckets, (self.num_envs,))
            # assign the new materials
            # material samples are of shape: num_env_ids x total_num_shapes x 3
            materials[:, start_idx:end_idx, 0] = self._domain_randomization[
                "friction"
            ]["static_friction"][bucket_ids, idx].unsqueeze(-1)
            materials[:, start_idx:end_idx, 1] = self._domain_randomization[
                "friction"
            ]["dynamic_friction"][bucket_ids, idx].unsqueeze(-1)
            materials[:, start_idx:end_idx, 2] = self._domain_randomization[
                "friction"
            ]["restitution"][bucket_ids, idx].unsqueeze(-1)
        self._robot.root_physx_view.set_material_properties(
            materials, indices=all_env_ids
        )

    def _apply_physx_com_randomization(self, all_env_ids: torch.Tensor) -> None:
        coms = self._robot.root_physx_view.get_coms().clone()
        coms[
            :, self._domain_randomization["center_of_mass"]["body_indices"], :3
        ] += self._domain_randomization["center_of_mass"]["com"].to(coms.device)
        self._robot.root_physx_view.set_coms(coms, all_env_ids)

    def _apply_scene_object_properties_after_spawn(
        self, all_env_ids: torch.Tensor
    ) -> None:
        """Apply scene object properties that require live PhysX views.

        IsaacLab USD spawners do not accept per-asset physics materials, so this
        runtime path is the source of truth for scene object material overrides.
        """
        if self.scene_lib.num_scenes() == 0:
            return
        object_dr = (
            self._domain_randomization.get("object_assets")
            if self._domain_randomization is not None
            else None
        )
        apply_center_of_mass = (
            object_dr is not None and object_dr.get("center_of_mass") is not None
        )

        for obj_idx, object_view in enumerate(self._object):
            materials = None
            coms = (
                object_view.root_physx_view.get_coms().clone()
                if apply_center_of_mass
                else None
            )
            for env_id, scene in enumerate(self.scene_lib.scenes):
                object_options = self._get_object_options_for_randomized_asset(
                    scene.objects[obj_idx], env_id=env_id
                )
                material_kwargs = object_options.physics_material_kwargs()
                if material_kwargs:
                    if materials is None:
                        materials = (
                            object_view.root_physx_view.get_material_properties()
                            .clone()
                            .to("cpu")
                        )
                    if "static_friction" in material_kwargs:
                        materials[env_id, :, 0] = material_kwargs["static_friction"]
                    if "dynamic_friction" in material_kwargs:
                        materials[env_id, :, 1] = material_kwargs["dynamic_friction"]
                    if "restitution" in material_kwargs:
                        materials[env_id, :, 2] = material_kwargs["restitution"]

                if not apply_center_of_mass:
                    continue
                center_of_mass = self._get_object_center_of_mass_for_randomized_asset(
                    scene.objects[obj_idx], env_id=env_id
                )
                if center_of_mass is None:
                    continue

                center_of_mass = center_of_mass.to(coms.device)
                if coms.ndim == 2:
                    coms[env_id, :3] = center_of_mass
                else:
                    coms[env_id, 0, :3] = center_of_mass

            if materials is not None:
                object_view.root_physx_view.set_material_properties(
                    materials, indices=all_env_ids
                )
            if apply_center_of_mass:
                object_view.root_physx_view.set_coms(coms, all_env_ids)

    # =====================================================
    # Group 3: Simulation Steps & State Management
    # =====================================================
    def _physics_step(self) -> None:
        """
        Advance the simulation by stepping for a number of iterations equal to the decimation factor.
        """
        for idx in range(self.decimation):
            self._apply_control()
            self._scene.write_data_to_sim()
            self._sim.step(render=False)
            if (idx + 1) % self.decimation == 0 and not self.headless:
                self._sim.render()
            self._scene.update(dt=self._sim.get_physics_dt())

    def _apply_simulator_pd_targets(self, pd_targets: torch.Tensor) -> None:
        """Applies PD position targets using IsaacLab's internal PD controller."""
        # Lab 3 Warp kernels need __cuda_array_interface__, which torch
        # refuses on Variables that still require grad.
        self._robot.set_joint_position_target(pd_targets.detach(), joint_ids=None)

    def _apply_simulator_torques(self, torques: torch.Tensor) -> None:
        """Applies torques to the robot DOFs."""
        self._robot.set_joint_effort_target(torques.detach(), joint_ids=None)

    def _set_simulator_env_state(
        self,
        new_states: ResetState,
        new_object_states: ObjectState = None,
        env_ids: torch.Tensor = None,
    ) -> None:
        """
        Apply the provided state to the simulation by writing root and joint states.

        Parameters:
            new_states (ResetState): The new simulation state.
            new_object_states (ObjectState): The new object state.
            env_ids (torch.Tensor): Specific environment IDs to update.
        """
        init_root_state = torch.cat(
            [
                new_states.root_pos,
                new_states.root_rot,
                new_states.root_vel,
                new_states.root_ang_vel,
            ],
            dim=-1,
        ).detach()
        self._robot.write_root_state_to_sim(init_root_state, env_ids)
        self._robot.set_joint_position_target(
            new_states.dof_pos.detach(), joint_ids=None, env_ids=env_ids
        )
        self._robot.write_joint_state_to_sim(
            new_states.dof_pos.detach(), new_states.dof_vel.detach(), None, env_ids
        )
        if new_object_states is not None and len(self._object) > 0:
            init_object_root_state = torch.cat(
                [
                    new_object_states.root_pos,
                    new_object_states.root_rot,
                    new_object_states.root_vel,
                    new_object_states.root_ang_vel,
                ],
                dim=-1,
            ).reshape(len(env_ids), self.scene_lib.num_objects_per_scene, 13)
            for object_idx in range(len(self._object)):
                self._object[object_idx].write_root_state_to_sim(
                    init_object_root_state[:, object_idx], env_ids
                )
        # Push written state into PhysX/USD so the Kit viewport can see it.
        # Kinematic playback never calls sim.step(), so this flush is required.
        self._scene.write_data_to_sim()
        self._sim.forward()

    @staticmethod
    def _as_torch(value):
        """Clone a sim data buffer as a torch tensor (Lab 3 returns Warp)."""
        if torch.is_tensor(value):
            return value.clone()
        import warp as wp

        return wp.to_torch(value).clone()

    # =====================================================
    # Group 4: State Getters
    # =====================================================
    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        """
        Obtain the ordering of body and degree-of-freedom names.

        Returns:
            SimBodyOrdering: An object containing the body names and DOF names.
        """
        if _ISAACLAB_W_LAST:
            from protomotions.simulator.isaaclab.utils.actuator_groups import (
                build_isaaclab_joint_name_map,
            )

            joint_names = build_isaaclab_joint_name_map(
                self.robot_config.kinematic_info
            )
            try:
                semantic_joint_names = [
                    joint_names.backend_to_semantic[name]
                    for name in self._robot.joint_names
                ]
            except KeyError as error:
                raise ValueError(
                    f"Unexpected IsaacLab joint name: {error.args[0]}"
                ) from error
            return SimBodyOrdering(
                body_names=self._robot.body_names,
                dof_names=semantic_joint_names,
            )
        return SimBodyOrdering(
            body_names=self._robot.data.body_names,
            dof_names=self._robot.data.joint_names,
        )

    def get_body_masses(self) -> torch.Tensor:
        """Per-body masses [num_envs, num_bodies] in COMMON body ordering."""
        masses = self._robot.data.default_mass.to(self.device)
        return masses[:, self.data_conversion.body_convert_to_common]

    def _get_simulator_bodies_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the state (positions, rotations, velocities) of all simulation bodies.

        Parameters:
            env_ids (Optional[torch.Tensor]): Restrict state retrieval to specific environments if provided.

        Returns:
            RobotState: The state of the bodies.
        """
        isaacsim_bodies_positions = self._as_torch(self._robot.data.body_pos_w)
        isaacsim_bodies_rotations = self._as_torch(self._robot.data.body_quat_w)
        isaacsim_bodies_velocities = self._as_torch(self._robot.data.body_lin_vel_w)
        isaacsim_bodies_ang_velocities = self._as_torch(self._robot.data.body_ang_vel_w)

        isaacsim_bodies_positions = isaacsim_bodies_positions.view(
            self.num_envs, self._num_bodies, 3
        )
        isaacsim_bodies_rotations = isaacsim_bodies_rotations.view(
            self.num_envs, self._num_bodies, 4
        )
        isaacsim_bodies_velocities = isaacsim_bodies_velocities.view(
            self.num_envs, self._num_bodies, 3
        )
        isaacsim_bodies_ang_velocities = isaacsim_bodies_ang_velocities.view(
            self.num_envs, self._num_bodies, 3
        )
        if env_ids is not None:
            isaacsim_bodies_positions = isaacsim_bodies_positions[env_ids]
            isaacsim_bodies_rotations = isaacsim_bodies_rotations[env_ids]
            isaacsim_bodies_velocities = isaacsim_bodies_velocities[env_ids]
            isaacsim_bodies_ang_velocities = isaacsim_bodies_ang_velocities[env_ids]
        return RobotState(
            rigid_body_pos=isaacsim_bodies_positions,
            rigid_body_rot=isaacsim_bodies_rotations,
            rigid_body_vel=isaacsim_bodies_velocities,
            rigid_body_ang_vel=isaacsim_bodies_ang_velocities,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_forces(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve applied torque forces for the robot's degrees of freedom.

        Parameters:
            env_ids (Optional[torch.Tensor]): Restrict query to specific environments if provided.

        Returns:
            torch.Tensor: The DOF forces.
        """
        isaacsim_dof_forces = self._robot.data.applied_torque.clone()
        if env_ids is not None:
            isaacsim_dof_forces = isaacsim_dof_forces[env_ids]
        return RobotState(
            dof_forces=isaacsim_dof_forces, state_conversion=StateConversion.SIMULATOR
        )

    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the state (positions and velocities) of the robot's DOFs.

        Parameters:
            env_ids (Optional[torch.Tensor]): Restrict state retrieval to specific environments if provided.

        Returns:
            RobotState: The DOF state.
        """
        isaacsim_dof_pos = self._robot.data.joint_pos.clone()
        isaacsim_dof_vel = self._robot.data.joint_vel.clone()
        if env_ids is not None:
            isaacsim_dof_pos = isaacsim_dof_pos[env_ids]
            isaacsim_dof_vel = isaacsim_dof_vel[env_ids]
        return RobotState(
            dof_pos=isaacsim_dof_pos,
            dof_vel=isaacsim_dof_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_limits_for_verification(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve DOF limits from IsaacLab's internal API for verification purposes only.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (lower_limits, upper_limits)
                                              in IsaacLab's DOF ordering.
        """
        # Extract limits from the robot data
        dof_limits = self._robot.data.joint_pos_limits.clone()
        # IsaacLab stores limits as [num_envs, num_dofs, 2], we take from first env
        return dof_limits[0, :, 0].to(self.device), dof_limits[0, :, 1].to(self.device)

    def _get_simulator_bodies_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the contact force buffer for simulation bodies in sim body order.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            RobotState: Robot state containing contact forces in simulator body order.
        """
        # Get simulator body ordering
        sim_body_names = self._robot.data.body_names
        num_bodies = len(sim_body_names)

        # Pre-allocate tensor for contact forces (initialized to zeros)
        rigid_body_contact_forces = torch.zeros(
            self.num_envs, num_bodies, 3, device=self.device
        )

        # Fill in contact forces for bodies that have sensors
        for body_idx, body_name in enumerate(sim_body_names):
            if body_name in self._contact_sensor_map:
                contact_sensor = self._contact_sensor_map[body_name]
                # net_forces_w has shape [num_envs, 1, 3], extract the single body dimension
                rigid_body_contact_forces[:, body_idx, :] = (
                    contact_sensor.data.net_forces_w.clone()[:, 0, :]
                )

        if env_ids is not None:
            rigid_body_contact_forces = rigid_body_contact_forces[env_ids]
        return RobotState(
            rigid_body_contact_forces=rigid_body_contact_forces,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_object_contact_buf(
        self,
        env_ids: Optional[torch.Tensor] = None,
    ) -> ObjectState:
        """
        Retrieve the contact buffer for simulation objects.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            torch.Tensor: The object contact buffer.
        """
        if self.scene_lib.num_scenes() > 0:
            object_forces = []
            for obj_idx in range(self.scene_lib.num_objects_per_scene):
                if self._object_contact_sensor[obj_idx] is not None:
                    object_forces.append(
                        self._object_contact_sensor[obj_idx].data.force_matrix_w.clone()
                    )
                else:
                    object_forces.append(
                        torch.zeros(
                            self.num_envs,
                            1,
                            1,
                            3,
                            device=self.device,
                            dtype=torch.float,
                        )
                    )
            if env_ids is not None:
                object_forces = object_forces[env_ids]
            return torch.cat(object_forces, dim=1)
        else:
            return_tensor = torch.zeros(
                self.num_envs, 1, 1, 3, device=self.device, dtype=torch.float
            )
            if env_ids is not None:
                return_tensor = return_tensor[env_ids]
            return return_tensor

    def _get_simulator_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RootOnlyState:
        """
        Retrieve the root state (position, rotation, velocity) of the robot.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            RootOnlyState: The robot's root state.
        """
        isaacsim_root_pos = self._as_torch(self._robot.data.root_pos_w)
        isaacsim_root_rot = self._as_torch(self._robot.data.root_quat_w)
        isaacsim_root_vel = self._as_torch(self._robot.data.root_lin_vel_w)
        isaacsim_root_ang_vel = self._as_torch(self._robot.data.root_ang_vel_w)
        if env_ids is not None:
            isaacsim_root_pos = isaacsim_root_pos[env_ids]
            isaacsim_root_rot = isaacsim_root_rot[env_ids]
            isaacsim_root_vel = isaacsim_root_vel[env_ids]
            isaacsim_root_ang_vel = isaacsim_root_ang_vel[env_ids]
        return RootOnlyState(
            root_pos=isaacsim_root_pos,
            root_rot=isaacsim_root_rot,
            root_vel=isaacsim_root_vel,
            root_ang_vel=isaacsim_root_ang_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """
        Retrieve the combined root state for all simulation objects.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            ObjectState: The objects' root state.
        """
        isaacsim_root_pos = []
        isaacsim_root_rot = []
        isaacsim_root_vel = []
        isaacsim_root_ang_vel = []
        for obj_idx in range(len(self._object)):
            isaacsim_root_pos.append(self._object[obj_idx].data.root_pos_w.clone())
            isaacsim_root_rot.append(self._object[obj_idx].data.root_quat_w.clone())
            isaacsim_root_vel.append(self._object[obj_idx].data.root_lin_vel_w.clone())
            isaacsim_root_ang_vel.append(
                self._object[obj_idx].data.root_ang_vel_w.clone()
            )
        isaacsim_root_pos = torch.stack(isaacsim_root_pos, dim=1)
        isaacsim_root_rot = torch.stack(isaacsim_root_rot, dim=1)
        isaacsim_root_vel = torch.stack(isaacsim_root_vel, dim=1)
        isaacsim_root_ang_vel = torch.stack(isaacsim_root_ang_vel, dim=1)
        if env_ids is not None:
            isaacsim_root_pos = isaacsim_root_pos[env_ids]
            isaacsim_root_rot = isaacsim_root_rot[env_ids]
            isaacsim_root_vel = isaacsim_root_vel[env_ids]
            isaacsim_root_ang_vel = isaacsim_root_ang_vel[env_ids]
        return ObjectState(
            root_pos=isaacsim_root_pos,
            root_rot=isaacsim_root_rot,
            root_vel=isaacsim_root_vel,
            root_ang_vel=isaacsim_root_ang_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def get_num_actors_per_env(self) -> int:
        """
        Compute and return the number of actor instances per environment.

        Returns:
            int: Number of actors per environment.
        """
        root_pos = self._robot.data.root_pos_w
        return root_pos.shape[0] // self.num_envs

    # =====================================================
    # Group 5: Control & Computation Methods
    # =====================================================

    def _apply_root_velocity_impulse(
        self,
        linear_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Apply velocity impulse to robot root by adding to current velocities."""
        current_vel = self._robot.data.root_vel_w[env_ids]
        new_vel = current_vel.clone()
        new_vel[:, :3] += linear_velocity
        new_vel[:, 3:6] += angular_velocity
        self._robot.write_root_velocity_to_sim(new_vel, env_ids=env_ids)

    # =====================================================
    # Projectile Implementation
    # =====================================================
    def _get_projectile_positions_rotations(self) -> tuple:
        """Return projectile (positions, rotations_xyzw) from IsaacLab rigid objects."""
        n_proj = self._proj_config.num_projectiles
        pos_list = []
        rot_list = []
        for pid in range(n_proj):
            state = self._projectile_objects[pid].data.root_state_w
            pos_list.append(state[:, 0:3])
            rot_wxyz = state[:, 3:7]
            rot_xyzw = torch.cat([rot_wxyz[:, 1:4], rot_wxyz[:, 0:1]], dim=-1)
            rot_list.append(rot_xyzw)
        return torch.stack(pos_list, dim=1), torch.stack(rot_list, dim=1)

    def _create_projectiles(self, config: ProjectileConfig) -> None:
        """Projectile objects are created via SceneCfg during __init__."""
        # Already created via SceneCfg -> projectile_{idx} RigidObjectCfg entries
        pass

    def _set_projectile_root_states(
        self,
        proj_indices: torch.Tensor,
        positions: torch.Tensor,
        rotations_xyzw: torch.Tensor,
        velocities: torch.Tensor,
        ang_velocities: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Set root state for specific projectiles via per-object write API."""
        # IsaacLab uses wxyz quaternion format
        rot_wxyz = rotations_xyzw[:, [3, 0, 1, 2]]

        for pid in proj_indices.unique():
            mask = proj_indices == pid
            eids = env_ids[mask]
            state = torch.cat(
                [
                    positions[mask],
                    rot_wxyz[mask],
                    velocities[mask],
                    ang_velocities[mask],
                ],
                dim=-1,
            )
            self._projectile_objects[pid.item()].write_root_state_to_sim(
                state, env_ids=eids
            )

    # =====================================================
    # Group 6: Rendering & Visualization
    # =====================================================
    def render(self) -> None:
        """
        Render the simulation view. Initializes or updates the camera if the simulator is not in headless mode.
        """
        if not self.headless and not getattr(self, "_perspective_view_failed", False):
            try:
                if not hasattr(self, "_perspective_view"):
                    from protomotions.simulator.isaaclab.utils.perspective_viewer import (
                        PerspectiveViewer,
                    )

                    self._perspective_view = PerspectiveViewer()
                    self._init_camera()
                else:
                    self._update_camera()
            except Exception as e:
                log.warning(
                    "PerspectiveViewer unavailable, using default viewport: %s", e
                )
                self._perspective_view_failed = True
        super().render()

    def _init_camera(self) -> None:
        """
        Initialize the camera view based on the current simulation root state.
        """
        self._cam_prev_char_pos = (
            self._get_simulator_root_state(0).root_pos.cpu().numpy()
        )
        # Spawn on the +Y side: the scene's distant light shines from +Y, so
        # this puts the camera sun-at-back (characters lit, not silhouetted).
        pos = self._cam_prev_char_pos + np.array([0, 5, 1])
        self._perspective_view.set_camera_view(
            pos, self._cam_prev_char_pos + np.array([0, 0, 0.2])
        )

    def _update_camera(self) -> None:
        """
        Update the camera view based on the target's position and current camera movement.
        """
        if self._camera_target["element"] == 0:
            char_root_pos = (
                self._get_simulator_root_state(self._camera_target["env"])
                .root_pos.cpu()
                .numpy()
                .copy()
            )
            # Fix the tracked height to the robot's nominal standing height so
            # the camera follows in x/y only and doesn't bounce with the body.
            char_root_pos[2] = self.robot_config.default_root_height
            height_offset = 0.2
        else:
            in_scene_object_id = self._camera_target["element"] - 1
            char_root_pos = (
                self._get_simulator_object_root_state(self._camera_target["env"])
                .root_pos[in_scene_object_id]
                .cpu()
                .numpy()
            )
            height_offset = 0

        cam_pos = np.array(self._perspective_view.get_camera_state())
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = np.array(
            [char_root_pos[0], char_root_pos[1], char_root_pos[2] + height_offset]
        )
        new_cam_pos = np.array(
            [
                char_root_pos[0] + cam_delta[0],
                char_root_pos[1] + cam_delta[1],
                char_root_pos[2] + cam_delta[2],
            ]
        )
        self._perspective_view.set_camera_view(new_cam_pos, new_cam_target)
        self._cam_prev_char_pos[:] = char_root_pos

    def _write_viewport_to_file(self, file_name: str) -> None:
        """
        Capture the current viewport and save it to the specified file.

        Parameters:
            file_name (str): The filename for the saved image.
        """
        from omni.kit.viewport.utility import (
            get_active_viewport,
            capture_viewport_to_file,
        )

        vp_api = get_active_viewport()
        capture_viewport_to_file(vp_api, file_name)

    def grab_rgb_frame(self, resolution=(960, 540)):
        """Synchronously return the perspective camera's current RGB as an
        HxWx3 uint8 numpy array (or None if not ready).

        Uses a Replicator ``rgb`` annotator on a render product bound to the
        viewer camera — the same mechanism IsaacLab's Camera sensors use — so
        it works with offscreen rendering (``enable_cameras``) and needs no
        display or viewport window (unlike ``capture_viewport_to_file``, which
        does not write under the headless.rendering Kit experience). Lazily
        created on first call; the perspective camera must already exist (it is
        created by ``render()`` once the sim is non-headless).
        """
        import numpy as np

        if getattr(self, "_rec_annotator", None) is None:
            import omni.replicator.core as rep

            self._rec_render_product = rep.create.render_product(
                "/OmniverseKit_Persp", resolution
            )
            self._rec_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self._rec_annotator.attach(self._rec_render_product)
            # Prime: the annotator only populates after a render.
            self._sim.render()

        data = self._rec_annotator.get_data()
        arr = np.asarray(data)
        if arr.size == 0 or arr.ndim < 2:
            return None
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        return np.ascontiguousarray(arr, dtype=np.uint8)

    def is_simulation_running(self) -> bool:
        """
        Check if the simulation is running.

        Also returns False once the Isaac Sim window is closed (app shutdown
        requested), so playback loops exit cleanly instead of hanging.
        """
        return self._simulation_running and self._simulation_app.is_running()

    def close(self) -> None:
        """
        Close the simulation application and perform cleanup.
        """
        super().close()
        self._simulation_app.close()

    def _build_markers(
        self, visualization_markers: Dict[str, VisualizationMarkerConfig]
    ) -> None:
        """Build and configure visualization markers.

        Args:
            visualization_markers (Dict[str, VisualizationMarkerConfig]): Dictionary mapping marker names to their configurations
        """
        self._visualization_markers: Dict[str, ProtoMotionsIsaacLabMarkers] = {}
        if visualization_markers is None:
            return

        for marker_name, markers_cfg in visualization_markers.items():
            if markers_cfg.type == "sphere":
                marker_obj_cfg = IsaacLabVisualizationMarkersCfg(
                    prim_path=f"/Visuals/{marker_name}",
                    markers={
                        "marker": sim_utils.SphereCfg(
                            radius=1,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(
                                    markers_cfg.color[0],
                                    markers_cfg.color[1],
                                    markers_cfg.color[2],
                                )
                            ),
                        ),
                    },
                )
            elif markers_cfg.type == "arrow":
                marker_obj_cfg = IsaacLabVisualizationMarkersCfg(
                    prim_path=f"/Visuals/{marker_name}",
                    markers={
                        "marker": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                            scale=(1.0, 1.0, 1.0),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(
                                    markers_cfg.color[0],
                                    markers_cfg.color[1],
                                    markers_cfg.color[2],
                                ),
                                opacity=0.5,
                            ),
                        ),
                    },
                )
            else:
                raise ValueError(f"Marker type {markers_cfg.type} not supported")

            marker_scale = []
            for i, marker in enumerate(markers_cfg.markers):
                if markers_cfg.type == "sphere":
                    if marker.scale is not None:
                        scale = marker.scale
                    elif marker.size == "tiny":
                        scale = 0.007
                    elif marker.size == "small":
                        scale = 0.01
                    else:
                        scale = 0.05
                    marker_scale.append([scale, scale, scale])
                elif markers_cfg.type == "arrow":
                    if marker.scale is not None:
                        scale = marker.scale
                    elif marker.size == "small":
                        scale = 0.1
                    else:
                        scale = 0.5
                    marker_scale.append([scale, 0.2 * scale, 0.2 * scale])

            if len(marker_scale) == 0:
                continue

            self._visualization_markers[marker_name] = ProtoMotionsIsaacLabMarkers(
                marker=IsaacLabVisualizationMarkers(marker_obj_cfg),
                scale=torch.tensor(marker_scale, device=self.device).repeat(
                    self.num_envs, 1
                ),
            )

    def _update_simulator_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """Update the visualization markers with new state information.

        Args:
            markers_state (Dict[str, MarkerState]): Dictionary mapping marker names to their state (translation and orientation)
        """
        if markers_state is None:
            return

        for marker_name, markers_state_item in markers_state.items():
            if markers_state_item.translation.numel() == 0:
                continue
            assert (
                marker_name in self._visualization_markers
            ), f"Marker {marker_name} passed to update_markers but not defined at instantiation"
            marker_dict = self._visualization_markers[marker_name]
            marker_dict.marker.visualize(
                translations=markers_state_item.translation.view(-1, 3),
                orientations=markers_state_item.orientation.view(-1, 4),
                scales=marker_dict.scale,
            )
