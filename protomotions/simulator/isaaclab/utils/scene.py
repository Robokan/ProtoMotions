import re
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from protomotions.components.terrains.terrain import Terrain
from protomotions.robot_configs.base import RobotConfig
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from protomotions.simulator.isaaclab.utils.usd_utils import TrimeshTerrainImporter
from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig
from protomotions.simulator.base_simulator.config import ProjectileConfig
from protomotions.robot_configs.base import ControlType

try:
    import isaaclab_physx  # noqa: F401

    _ISAACLAB3 = True
except ImportError:
    _ISAACLAB3 = False


@configclass
class TrimeshTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = TrimeshTerrainImporter

    terrain_type: str = "trimesh"
    terrain_vertices: list = None
    terrain_faces: list = None


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    def __init__(
        self,
        config: IsaacLabSimulatorConfig,
        robot_config: RobotConfig,
        terrain: Optional[Terrain] = None,
        scene_cfgs=None,
        projectile_config: Optional[ProjectileConfig] = None,
        pretty=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        activate_contact_sensors = robot_config.contact_bodies is not None

        # lights
        if True:  # pretty:
            # This is way prettier, but also slower to render
            self.light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                ),
            )
        else:
            self.light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(
                    intensity=3000.0, color=(0.75, 0.75, 0.75)
                ),
            )

        num_objects_per_scene = 0
        if scene_cfgs is not None:
            num_objects_per_scene = len(scene_cfgs)
            for obj_idx, obj_configs in enumerate(scene_cfgs):
                spawn_cfg = sim_utils.MultiAssetSpawnerCfg(
                    activate_contact_sensors=activate_contact_sensors,
                    assets_cfg=obj_configs,
                    random_choice=False,
                )
                # Rigid Object
                object = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Object_{obj_idx}",
                    spawn=spawn_cfg,
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                setattr(self, f"object_{obj_idx}", object)

                # Object contact sensors are used to detect collisions between objects.
                object_contact_paths = ["/World/ground/terrain/mesh"]
                for i in range(num_objects_per_scene):
                    if i != obj_idx:
                        object_contact_paths.append(f"/World/envs/env_.*/Object_{i}")
                if activate_contact_sensors:
                    object_sensor_cfg = ContactSensorCfg(
                        prim_path=f"/World/envs/env_.*/Object_{obj_idx}",
                        # debug_vis=True,
                        filter_prim_paths_expr=object_contact_paths,
                        history_length=config.sim.decimation,
                    )
                    setattr(self, f"object_{obj_idx}_contact_sensor", object_sensor_cfg)

        # Projectile rigid objects (always created, independent of scene objects)
        if projectile_config is not None:
            proj_sizes = projectile_config.get_sizes()
            for proj_idx in range(projectile_config.num_projectiles):
                s = proj_sizes[proj_idx]
                proj_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Projectile_{proj_idx}",
                    spawn=sim_utils.CuboidCfg(
                        size=(s * 2, s * 2, s * 2),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=False,
                            enable_gyroscopic_forces=True,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(
                            density=projectile_config.density
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.02,
                            rest_offset=0.0,
                        ),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.8, 0.1, 0.1)
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0.0, 0.0, projectile_config.hide_z)
                    ),
                )
                setattr(self, f"projectile_{proj_idx}", proj_cfg)

        actuators = {}
        # Newton (Lab 3) cannot use IMPLICIT PD. Newton drives a joint from
        # joint_target_ke/kd only when its joint_target_mode is POSITION, and
        # MuJoCo builds its actuator array at model FINALIZE -- our own Newton
        # engine therefore sets the mode on the builder beforehand
        # (simulator/newton/simulator.py::_set_builder_dof_properties). Lab 3
        # binds ke/kd (a read-back shows the configured 40/5) but leaves the
        # mode at NONE/EFFORT and offers no pre-finalize hook, so the gains sit
        # inert: measured, joints missed the commanded default pose by 0.68 rad
        # mean / 3.78 rad worst while burning 26 Nm -- the legs flop. Setting
        # the mode after finalize does NOT help (verified). Explicit PD has Lab
        # compute the torque and Newton apply it: 0.097 rad / 3.4 Nm, matching
        # PhysX's 0.069 / 2.0.
        _newton_backend = getattr(config, "physics_backend", "physx") in (
            "newton",
            "newton_mjwarp",
        )
        ActuatorConfig = (
            ImplicitActuatorCfg
            if robot_config.control.control_type == ControlType.BUILT_IN_PD
            and not _newton_backend
            else IdealPDActuatorCfg
        )
        body_prim_paths = None
        if _ISAACLAB3:
            from protomotions.simulator.isaaclab.utils.actuator_groups import (
                build_isaaclab_joint_name_map,
                resolve_actuator_specs_for_control_type,
            )
            from protomotions.simulator.isaaclab.utils.mjcf_to_usd import (
                convert_robot_mjcf_to_usd,
            )
            from protomotions.simulator.isaaclab.utils.usd_body_paths import (
                resolve_robot_prim_paths,
            )

            joint_names = build_isaaclab_joint_name_map(robot_config.kinematic_info)
            isaaclab_control_info = {
                joint_names.semantic_to_backend[name]: control_info
                for name, control_info in robot_config.control.control_info.items()
            }
            for actuator_group in resolve_actuator_specs_for_control_type(
                isaaclab_control_info, robot_config.control.control_type
            ):
                actuators[actuator_group.name] = ActuatorConfig(
                    joint_names_expr=list(actuator_group.joint_names_expr),
                    **actuator_group.params,
                )

            lab3_usd = getattr(robot_config.asset, "lab3_usd_asset_file_name", None)
            keep_authored_look = not (
                getattr(robot_config.asset, "override_visual_material", True)
                and getattr(
                    robot_config.asset, "apply_default_visual_material", True
                )
            )
            visual_material = (
                None
                if keep_authored_look
                else sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.9, 0.9), metallic=0.5
                )
            )
            if lab3_usd:
                from pathlib import Path

                asset_root = Path(robot_config.asset.asset_root)
                if not asset_root.is_absolute():
                    asset_root = Path(__file__).resolve().parents[4] / asset_root
                robot_usd_path = str((asset_root / lab3_usd).resolve())
            else:
                robot_usd_path = convert_robot_mjcf_to_usd(robot_config.asset)
            contact_body_names = (
                robot_config.contact_bodies if activate_contact_sensors else []
            )
            articulation_root_prim_path, body_prim_paths = resolve_robot_prim_paths(
                robot_usd_path,
                contact_body_names,
            )
            default_joint_pos = (
                {
                    joint_names.semantic_to_backend[name]: float(
                        robot_config.default_dof_pos[i]
                    )
                    for i, name in enumerate(robot_config.kinematic_info.dof_names)
                }
                if robot_config.default_dof_pos is not None
                else {".*": 0.0}
            )
            self.robot = ArticulationCfg(
                prim_path="/World/envs/env_.*/Robot",
                articulation_root_prim_path=articulation_root_prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=robot_usd_path,
                    activate_contact_sensors=activate_contact_sensors,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=robot_config.asset.disable_gravity,
                        retain_accelerations=False,
                        linear_damping=robot_config.asset.linear_damping,
                        angular_damping=robot_config.asset.angular_damping,
                        max_linear_velocity=robot_config.asset.max_linear_velocity,
                        max_angular_velocity=robot_config.asset.max_angular_velocity,
                        max_depenetration_velocity=config.sim.physx.max_depenetration_velocity,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=robot_config.asset.self_collisions,
                        solver_position_iteration_count=config.sim.physx.num_position_iterations,
                        solver_velocity_iteration_count=config.sim.physx.num_velocity_iterations,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=config.sim.physx.contact_offset,
                        rest_offset=config.sim.physx.rest_offset,
                        collision_enabled=getattr(
                            robot_config.asset, "collision_enabled", None
                        ),
                    ),
                    visual_material=visual_material,
                ),
                # Height comes from write_root_state (Hip). Do not lift the
                # spawned USD prim — Lab 3 MJCF converts bake a worldbody
                # floor under /Geometry, and raising the prim lifts that
                # plane through the torso so the robot is shoved up ~1 m.
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),
                    joint_pos=default_joint_pos,
                    joint_vel={".*": 0.0},
                ),
                actuators=actuators,
            )
        else:
            # One actuator per DOF costs a full Python-level actuator model
            # evaluation per joint per physics step: on the 210-dof raptor that
            # was 69% of inference wall-clock (py-spy), dwarfing physics (7%)
            # and rendering (4%). DOFs sharing identical parameters behave
            # identically, so bucket them into one vectorised actuator each.
            _buckets = {}
            for dof_name, control_info in robot_config.control.control_info.items():
                stiffness = control_info.stiffness
                damping = control_info.damping
                if robot_config.control.control_type != ControlType.BUILT_IN_PD:
                    stiffness = 0.0
                    damping = 0.0
                kwargs = {
                    key: value
                    for key, value in {
                        "stiffness": stiffness,
                        "damping": damping,
                        "armature": control_info.armature,
                        "effort_limit_sim": control_info.effort_limit,
                        "velocity_limit_sim": control_info.velocity_limit,
                        "friction": control_info.friction,
                    }.items()
                    if value is not None
                }
                _buckets.setdefault(
                    tuple(sorted(kwargs.items())), []).append(dof_name)
            for _key, _dofs in _buckets.items():
                # Escape regex metacharacters: joint_names_expr is regex-matched.
                actuators[f"group_{len(actuators)}"] = ActuatorConfig(
                    joint_names_expr=[re.escape(d) for d in _dofs], **dict(_key)
                )

            # articulation
            def _robot_usd_cfg(usd_file_name, visual_material):
                return sim_utils.UsdFileCfg(
                    usd_path=f"{robot_config.asset.asset_root}/{usd_file_name}",
                    activate_contact_sensors=activate_contact_sensors,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=robot_config.asset.disable_gravity,
                        retain_accelerations=False,
                        linear_damping=robot_config.asset.linear_damping,
                        angular_damping=robot_config.asset.angular_damping,
                        max_linear_velocity=robot_config.asset.max_linear_velocity,
                        max_angular_velocity=robot_config.asset.max_angular_velocity,
                        max_depenetration_velocity=config.sim.physx.max_depenetration_velocity,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=robot_config.asset.self_collisions,
                        solver_position_iteration_count=config.sim.physx.num_position_iterations,
                        solver_velocity_iteration_count=config.sim.physx.num_velocity_iterations,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=config.sim.physx.contact_offset,
                        rest_offset=config.sim.physx.rest_offset,
                        collision_enabled=getattr(
                            robot_config.asset, "collision_enabled", None
                        ),
                    ),
                    visual_material=visual_material,
                )

            override_material = (
                sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9), metallic=0.5)
                if (
                    getattr(robot_config.asset, "override_visual_material", True)
                    and getattr(
                        robot_config.asset, "apply_default_visual_material", True
                    )  # go2-training's spelling of the same opt-out
                )
                else None
            )
            opponent_usd = getattr(
                robot_config.asset, "opponent_usd_asset_file_name", None
            )
            if opponent_usd:
                # Paired battle envs: ego half (envs 0..N/2-1) uses the base USD,
                # opponent half uses the variant, so fighters are tellable apart.
                # spawn_multi_asset assigns assets_cfg[index % len] per env, so a
                # per-env list makes the split exact. Requires replicate_physics
                # off (viewer-scale env counts only). No material override — the
                # different USD colors are the whole point.
                half = self.num_envs // 2
                robot_spawn = sim_utils.MultiAssetSpawnerCfg(
                    assets_cfg=(
                        [_robot_usd_cfg(robot_config.asset.usd_asset_file_name, None)]
                        * half
                        + [_robot_usd_cfg(opponent_usd, None)]
                        * (self.num_envs - half)
                    ),
                    random_choice=False,
                    activate_contact_sensors=activate_contact_sensors,
                )
            else:
                robot_spawn = _robot_usd_cfg(
                    robot_config.asset.usd_asset_file_name, override_material
                )

            self.robot = ArticulationCfg(
                prim_path="/World/envs/env_.*/Robot",
                spawn=robot_spawn,
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, robot_config.default_root_height),
                    joint_pos={
                        name: float(robot_config.default_dof_pos[i])
                        for i, name in enumerate(robot_config.kinematic_info.dof_names)
                    } if robot_config.default_dof_pos is not None else {".*": 0.0},
                    joint_vel={".*": 0.0},
                ),
                actuators=actuators,
            )

        # Apply disable_gravity setting for all robot types if specified
        if (
            hasattr(robot_config.asset, "disable_gravity")
            and robot_config.asset.disable_gravity
        ):
            # Only modify disable_gravity field, keeping all other settings
            new_rigid_props = self.robot.spawn.rigid_props.replace(disable_gravity=True)
            self.robot.spawn = self.robot.spawn.replace(rigid_props=new_rigid_props)

        if activate_contact_sensors:
            sensing_filter = ["/World/ground/terrain/mesh"]
            for obj_idx in range(num_objects_per_scene):
                sensing_filter.append(f"/World/envs/env_.*/Object_{obj_idx}")
            if getattr(config, "physics_backend", "physx") in (
                "newton",
                "newton_mjwarp",
            ):
                # Lab 3's Newton contact sensor resolves each filter entry to a
                # BODY label in the Newton model. The ground plane is not a body
                # there, so a ground filter fails initialization outright ("No
                # bodies matched the counterpart pattern(s)"). Unfiltered sensing
                # reports the body's NET contact force, which is what the foot
                # contact observations consume anyway.
                sensing_filter = []
            for body_name in robot_config.contact_bodies:
                if _ISAACLAB3:
                    from protomotions.simulator.isaaclab.utils.usd_body_paths import (
                        contact_sensor_prim_path,
                    )

                    prim_path = contact_sensor_prim_path(body_name, body_prim_paths)
                else:
                    prim_path = (
                        f"{robot_config.asset.usd_bodies_root_prim_path}{body_name}"
                    )
                contact_sensor_cfg = ContactSensorCfg(
                    prim_path=prim_path,
                    filter_prim_paths_expr=sensing_filter,
                    history_length=config.sim.decimation,
                )
                setattr(self, f"contact_sensor_{body_name}", contact_sensor_cfg)

        if terrain is not None:
            terrain_physics_material = sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode=terrain.sim_config.combine_mode.value,
                restitution_combine_mode=terrain.sim_config.combine_mode.value,
                static_friction=terrain.sim_config.static_friction,
                dynamic_friction=terrain.sim_config.dynamic_friction,
                restitution=terrain.sim_config.restitution,
            )
            terrain_visual_material = sim_utils.MdlFileCfg(
                mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                project_uvw=True,
            )

            vertices = terrain.vertices
            height_offset = terrain.sim_config.height_offset
            vertices[..., 2] += height_offset

            self.terrain = TrimeshTerrainImporterCfg(
                prim_path="/World/ground",
                # Pass the mesh data instead of the mesh object
                terrain_vertices=vertices.tolist(),
                terrain_faces=terrain.triangles,
                collision_group=-1,
                visual_material=terrain_visual_material,
                physics_material=terrain_physics_material,
            )
        else:
            self.terrain = None
