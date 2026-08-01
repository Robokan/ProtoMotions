# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-morphology IsaacLab simulator (MULTI_ROBOT_LEAGUE_PLAN Phase 3).

Block-partitioned for paired battle envs: side A (ego block, envs
``[0..N)``) runs ``robot_config``; side B (opponent block, ``[N..2N)``)
runs ``simulator_config.opponent_robot_config``. PhysX articulation views
must be homogeneous, so BOTH robots are spawned in EVERY env (IsaacLab
clones prims per env) and each env's inactive twin is parked far below
the arena — re-parked on every reset so free-fall drift stays bounded.

Tensor contract to the env: state getters return zero-padded common-order
tensors ``[num_envs, max(num_bodies), ...]`` / ``[num_envs, max(num_dofs)]``;
``step()`` takes ``[num_envs, max(num_actions)]`` and routes each block's
leading columns to its own articulation with its own DataConversionMapping
and PD gains. With identical robot configs on both sides the padding is a
no-op and this reduces to the single-robot behavior (validation rung 1).
"""

from typing import Dict, Optional, Tuple

import torch

from protomotions.robot_configs.base import ControlType
from protomotions.simulator.base_simulator.simulator_state import (
    DataConversionMapping,
    ObjectState,
    ResetState,
    RobotState,
    RootOnlyState,
    StateConversion,
)
from protomotions.simulator.isaaclab.simulator import IsaacLabSimulator
from protomotions.simulator.isaaclab.utils.scene import SceneCfg

TWIN_PARK_Z = -50.0


class MultiRobotSceneCfg(SceneCfg):
    """SceneCfg with a second articulation (``robot_b``) per env."""

    def __init__(self, config, robot_config, opponent_robot_config, **kwargs):
        super().__init__(config, robot_config, **kwargs)
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
        from isaaclab.assets import ArticulationCfg
        from isaaclab.sensors import ContactSensorCfg

        opp = opponent_robot_config
        actuators = {}
        ActuatorConfig = (
            ImplicitActuatorCfg
            if opp.control.control_type == ControlType.BUILT_IN_PD
            else IdealPDActuatorCfg
        )
        for dof_name, control_info in opp.control.control_info.items():
            stiffness = control_info.stiffness
            damping = control_info.damping
            if opp.control.control_type != ControlType.BUILT_IN_PD:
                stiffness = 0.0
                damping = 0.0
            actuators[dof_name] = ActuatorConfig(
                joint_names_expr=[dof_name],
                **{
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
                },
            )

        activate_contact_sensors = opp.contact_bodies is not None
        spawn = sim_utils.UsdFileCfg(
            usd_path=f"{opp.asset.asset_root}/{opp.asset.usd_asset_file_name}",
            activate_contact_sensors=activate_contact_sensors,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=opp.asset.disable_gravity,
                retain_accelerations=False,
                linear_damping=opp.asset.linear_damping,
                angular_damping=opp.asset.angular_damping,
                max_linear_velocity=opp.asset.max_linear_velocity,
                max_angular_velocity=opp.asset.max_angular_velocity,
                max_depenetration_velocity=config.sim.physx.max_depenetration_velocity,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=opp.asset.self_collisions,
                solver_position_iteration_count=config.sim.physx.num_position_iterations,
                solver_velocity_iteration_count=config.sim.physx.num_velocity_iterations,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=config.sim.physx.contact_offset,
                rest_offset=config.sim.physx.rest_offset,
            ),
            visual_material=None,
        )
        # Spawn offset keeps the twins from interpenetrating before the
        # first reset parks/places them.
        self.robot_b = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot_B",
            spawn=spawn,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(1.5, 0.0, opp.default_root_height),
                joint_pos={".*": 0.0},
                joint_vel={".*": 0.0},
            ),
            actuators=actuators,
        )

        if activate_contact_sensors:
            # Side-B sensor prims live under Robot_B; the attr prefix keeps
            # same-named bodies (both robots have a head) from colliding.
            bodies_root = opp.asset.usd_bodies_root_prim_path.replace(
                "/Robot/", "/Robot_B/"
            )
            sensing_filter = ["/World/ground/terrain/mesh"]
            for body_name in opp.contact_bodies:
                setattr(
                    self,
                    f"contact_sensor_b_{body_name}",
                    ContactSensorCfg(
                        prim_path=f"{bodies_root}{body_name}",
                        filter_prim_paths_expr=sensing_filter,
                        history_length=config.sim.decimation,
                    ),
                )


class MultiRobotIsaacLabSimulator(IsaacLabSimulator):
    """IsaacLab simulator hosting two robot morphologies in paired blocks."""

    def __init__(self, config, robot_config, **kwargs):
        opp_config = getattr(config, "opponent_robot_config", None)
        if opp_config is None:
            raise ValueError(
                "MultiRobotIsaacLabSimulator requires "
                "simulator_config.opponent_robot_config"
            )
        self.opp_robot_config = opp_config
        super().__init__(config=config, robot_config=robot_config, **kwargs)

        if self.num_envs % 2 != 0:
            raise ValueError("Multi-robot battle envs must come in pairs")
        half = self.num_envs // 2
        self._block_a = torch.arange(0, half, device=self.device)
        self._block_b = torch.arange(half, self.num_envs, device=self.device)

        ki_a = self.robot_config.kinematic_info
        ki_b = opp_config.kinematic_info
        self._nb_a, self._nb_b = ki_a.num_bodies, ki_b.num_bodies
        self._nd_a, self._nd_b = ki_a.num_dofs, ki_b.num_dofs
        self._na_a = self.robot_config.number_of_actions
        self._na_b = opp_config.number_of_actions
        self._pad_bodies = max(self._nb_a, self._nb_b)
        self._pad_dofs = max(self._nd_a, self._nd_b)
        self._pad_actions = max(self._na_a, self._na_b)

        if self._pad_actions != self._na_a:
            # Action buffers were allocated at side A's width; the padded
            # contract is [num_envs, max_actions].
            for name in ("_common_actions", "_previous_actions", "_prev_prev_actions"):
                setattr(
                    self,
                    name,
                    torch.zeros(
                        self.num_envs, self._pad_actions,
                        device=self.device, dtype=torch.float,
                    ),
                )

        if self._domain_randomization:
            raise NotImplementedError(
                "Domain randomization is not supported by the multi-robot "
                "simulator yet (single root_physx_view assumptions)"
            )

    # ------------------------------------------------------------------
    # Scene / views
    # ------------------------------------------------------------------
    def _get_scene_cfg(self):
        scene_cfgs = None
        if self.scene_lib.num_scenes() > 0:
            raise NotImplementedError(
                "Scene objects are not supported by the multi-robot simulator yet"
            )
        return MultiRobotSceneCfg(
            config=self.config,
            robot_config=self.robot_config,
            opponent_robot_config=self.opp_robot_config,
            num_envs=self.config.num_envs,
            env_spacing=getattr(self.config, "env_spacing", 2.0),
            scene_cfgs=scene_cfgs,
            terrain=self.terrain,
            projectile_config=self._proj_config,
            replicate_physics=False,  # two distinct articulations per env
            filter_collisions=self.config.filter_env_collisions,
        )

    def _create_simulation(self) -> None:
        super()._create_simulation()
        self._robot_b = self._scene["robot_b"]
        self._contact_sensor_map_b = {}
        for body_name in self.opp_robot_config.kinematic_info.body_names:
            key = f"contact_sensor_b_{body_name}"
            if key in self._scene.keys():
                self._contact_sensor_map_b[body_name] = self._scene[key]
        self._park_inactive_twins()

    def _park_inactive_twins(self) -> None:
        """Teleport each env's inactive articulation far below the arena.

        Robot A instances in the opponent block and robot B instances in the
        ego block never fight; they sit at TWIN_PARK_Z (staggered so the two
        parked shelves cannot interpenetrate) with zeroed velocities.
        """
        self._park_twin_rows(self._robot, self._block_b, TWIN_PARK_Z)
        self._park_twin_rows(self._robot_b, self._block_a, TWIN_PARK_Z - 10.0)

    def _park_twin_rows(self, robot, rows: torch.Tensor, park_z: float) -> None:
        if rows.numel() == 0:
            return
        root = robot.data.root_state_w[rows].clone()
        root[:, 2] = park_z
        root[:, 7:13] = 0.0  # zero linear + angular velocity
        robot.write_root_state_to_sim(root, rows)
        num_dofs = robot.data.joint_pos.shape[1]
        zeros = torch.zeros(rows.numel(), num_dofs, device=self.device)
        robot.set_joint_position_target(zeros, joint_ids=None, env_ids=rows)
        robot.write_joint_state_to_sim(zeros, zeros.clone(), None, rows)

    # ------------------------------------------------------------------
    # Conversion mappings / gains for side B
    # ------------------------------------------------------------------
    def _finalize_setup(self) -> None:
        super()._finalize_setup()  # side A: gains, data_conversion, verify

        ki_b = self.opp_robot_config.kinematic_info
        sim_body_names = self._robot_b.data.body_names
        sim_dof_names = self._robot_b.data.joint_names
        self.data_conversion_b = DataConversionMapping(
            body_convert_to_common=torch.tensor(
                [sim_body_names.index(n) for n in ki_b.body_names],
                dtype=torch.long, device=self.device,
            ),
            body_convert_to_sim=torch.tensor(
                [ki_b.body_names.index(n) for n in sim_body_names],
                dtype=torch.long, device=self.device,
            ),
            dof_convert_to_sim=torch.tensor(
                [ki_b.dof_names.index(n) for n in sim_dof_names],
                dtype=torch.long, device=self.device,
            ),
            dof_convert_to_common=torch.tensor(
                [sim_dof_names.index(n) for n in ki_b.dof_names],
                dtype=torch.long, device=self.device,
            ),
            sim_w_last=self.config.w_last,
        )
        self._verify_joint_limits_b()

    def _verify_joint_limits_b(self) -> None:
        ki_b = self.opp_robot_config.kinematic_info
        limits = self._robot_b.data.joint_pos_limits[0].to(self.device)
        lower = limits[:, 0][self.data_conversion_b.dof_convert_to_common]
        upper = limits[:, 1][self.data_conversion_b.dof_convert_to_common]
        max_diff = torch.maximum(
            (lower - ki_b.dof_limits_lower.to(self.device)).abs().max(),
            (upper - ki_b.dof_limits_upper.to(self.device)).abs().max(),
        )
        if max_diff > 1e-5:
            raise ValueError(
                f"Side-B joint limits disagree with MJCF by {max_diff:.2e}"
            )

    # ------------------------------------------------------------------
    # Padded per-block state getters
    # ------------------------------------------------------------------
    def _pad_cat(self, a: torch.Tensor, b: torch.Tensor, width: int) -> torch.Tensor:
        """Cat block tensors along dim 0, zero-padding dim 1 to ``width``."""
        def pad(t):
            if t.shape[1] == width:
                return t
            shape = list(t.shape)
            shape[1] = width - t.shape[1]
            return torch.cat([t, torch.zeros(shape, device=t.device, dtype=t.dtype)], dim=1)
        return torch.cat([pad(a), pad(b)], dim=0)

    @staticmethod
    def _rows(t: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return t.clone()[rows]

    @staticmethod
    def _fit_width(t: torch.Tensor, width: int) -> torch.Tensor:
        """Slice or zero-pad dim 1 to ``width`` (reset states arrive at the
        padded width once the env layer is per-side aware; until then a
        narrower ego-width tensor pads with zeros for the wider side)."""
        if t.shape[1] == width:
            return t
        if t.shape[1] > width:
            return t[:, :width]
        pad = torch.zeros(t.shape[0], width - t.shape[1], device=t.device, dtype=t.dtype)
        return torch.cat([t, pad], dim=1)

    def get_root_state(self, env_ids: Optional[torch.Tensor] = None) -> RootOnlyState:
        states = []
        for robot, rows, conv in (
            (self._robot, self._block_a, self.data_conversion),
            (self._robot_b, self._block_b, self.data_conversion_b),
        ):
            states.append(
                RootOnlyState(
                    root_pos=self._rows(robot.data.root_pos_w, rows),
                    root_rot=self._rows(robot.data.root_quat_w, rows),
                    root_vel=self._rows(robot.data.root_lin_vel_w, rows),
                    root_ang_vel=self._rows(robot.data.root_ang_vel_w, rows),
                    state_conversion=StateConversion.SIMULATOR,
                ).convert_to_common(conv)
            )
        a, b = states
        merged = RootOnlyState(
            root_pos=torch.cat([a.root_pos, b.root_pos]),
            root_rot=torch.cat([a.root_rot, b.root_rot]),
            root_vel=torch.cat([a.root_vel, b.root_vel]),
            root_ang_vel=torch.cat([a.root_ang_vel, b.root_ang_vel]),
            state_conversion=StateConversion.COMMON,
        )
        return merged[env_ids] if env_ids is not None else merged

    def get_bodies_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        states = []
        for robot, rows, conv, nb in (
            (self._robot, self._block_a, self.data_conversion, self._nb_a),
            (self._robot_b, self._block_b, self.data_conversion_b, self._nb_b),
        ):
            states.append(
                RobotState(
                    rigid_body_pos=self._rows(
                        robot.data.body_pos_w.view(self.num_envs, nb, 3), rows
                    ),
                    rigid_body_rot=self._rows(
                        robot.data.body_quat_w.view(self.num_envs, nb, 4), rows
                    ),
                    rigid_body_vel=self._rows(
                        robot.data.body_lin_vel_w.view(self.num_envs, nb, 3), rows
                    ),
                    rigid_body_ang_vel=self._rows(
                        robot.data.body_ang_vel_w.view(self.num_envs, nb, 3), rows
                    ),
                    state_conversion=StateConversion.SIMULATOR,
                ).convert_to_common(conv)
            )
        a, b = states
        merged = RobotState(
            rigid_body_pos=self._pad_cat(a.rigid_body_pos, b.rigid_body_pos, self._pad_bodies),
            rigid_body_rot=self._pad_cat(a.rigid_body_rot, b.rigid_body_rot, self._pad_bodies),
            rigid_body_vel=self._pad_cat(a.rigid_body_vel, b.rigid_body_vel, self._pad_bodies),
            rigid_body_ang_vel=self._pad_cat(
                a.rigid_body_ang_vel, b.rigid_body_ang_vel, self._pad_bodies
            ),
            state_conversion=StateConversion.COMMON,
        )
        return merged[env_ids] if env_ids is not None else merged

    def get_dof_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        states = []
        for robot, rows, conv in (
            (self._robot, self._block_a, self.data_conversion),
            (self._robot_b, self._block_b, self.data_conversion_b),
        ):
            states.append(
                RobotState(
                    dof_pos=self._rows(robot.data.joint_pos, rows),
                    dof_vel=self._rows(robot.data.joint_vel, rows),
                    state_conversion=StateConversion.SIMULATOR,
                ).convert_to_common(conv)
            )
        a, b = states
        merged = RobotState(
            dof_pos=self._pad_cat(a.dof_pos, b.dof_pos, self._pad_dofs),
            dof_vel=self._pad_cat(a.dof_vel, b.dof_vel, self._pad_dofs),
            state_conversion=StateConversion.COMMON,
        )
        return merged[env_ids] if env_ids is not None else merged

    def get_dof_forces(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        states = []
        for robot, rows, conv in (
            (self._robot, self._block_a, self.data_conversion),
            (self._robot_b, self._block_b, self.data_conversion_b),
        ):
            states.append(
                RobotState(
                    dof_forces=self._rows(robot.data.applied_torque, rows),
                    state_conversion=StateConversion.SIMULATOR,
                ).convert_to_common(conv)
            )
        a, b = states
        merged = RobotState(
            dof_forces=self._pad_cat(a.dof_forces, b.dof_forces, self._pad_dofs),
            state_conversion=StateConversion.COMMON,
        )
        return merged[env_ids] if env_ids is not None else merged

    def get_bodies_contact_buf(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        states = []
        for robot, rows, conv, nb, sensor_map in (
            (self._robot, self._block_a, self.data_conversion, self._nb_a,
             self._contact_sensor_map),
            (self._robot_b, self._block_b, self.data_conversion_b, self._nb_b,
             self._contact_sensor_map_b),
        ):
            forces = torch.zeros(self.num_envs, nb, 3, device=self.device)
            for body_idx, body_name in enumerate(robot.data.body_names):
                if body_name in sensor_map:
                    forces[:, body_idx, :] = (
                        sensor_map[body_name].data.net_forces_w.clone()[:, 0, :]
                    )
            states.append(
                RobotState(
                    rigid_body_contact_forces=forces[rows],
                    state_conversion=StateConversion.SIMULATOR,
                ).convert_to_common(conv)
            )
        a, b = states
        merged = RobotState(
            rigid_body_contact_forces=self._pad_cat(
                a.rigid_body_contact_forces, b.rigid_body_contact_forces,
                self._pad_bodies,
            ),
            state_conversion=StateConversion.COMMON,
        )
        return merged[env_ids] if env_ids is not None else merged

    def get_body_masses(self) -> torch.Tensor:
        a = self._robot.data.default_mass.to(self.device)[self._block_a][
            :, self.data_conversion.body_convert_to_common
        ]
        b = self._robot_b.data.default_mass.to(self.device)[self._block_b][
            :, self.data_conversion_b.body_convert_to_common
        ]
        return self._pad_cat(a, b, self._pad_bodies)

    def get_num_actors_per_env(self) -> int:
        return 2

    def get_default_robot_reset_state(self) -> ResetState:
        """Per-side default pose: block A rows carry robot A's default pose
        and stance height, block B rows robot B's (dof width padded)."""
        state = super().get_default_robot_reset_state()  # side A everywhere
        state.dof_pos = self._fit_width(state.dof_pos, self._pad_dofs)
        state.dof_vel = self._fit_width(state.dof_vel, self._pad_dofs)
        opp = self.opp_robot_config
        dof_b = opp.default_dof_pos.to(self.device)
        state.dof_pos[self._block_b, : dof_b.shape[0]] = dof_b.unsqueeze(0)
        state.dof_pos[self._block_b, dof_b.shape[0]:] = 0.0
        state.root_pos[self._block_b, 2] = opp.default_root_height
        return state

    # ------------------------------------------------------------------
    # Actions: per-block routing
    # ------------------------------------------------------------------
    def _apply_control(self) -> None:
        if self.control_type == ControlType.BUILT_IN_PD:
            targets = self._common_actions
            ta = targets[self._block_a][:, : self._nd_a][
                :, self.data_conversion.dof_convert_to_sim
            ]
            self._robot.set_joint_position_target(
                ta, joint_ids=None, env_ids=self._block_a
            )
            tb = targets[self._block_b][:, : self._nd_b][
                :, self.data_conversion_b.dof_convert_to_sim
            ]
            self._robot_b.set_joint_position_target(
                tb, joint_ids=None, env_ids=self._block_b
            )
        else:
            raise NotImplementedError(
                f"Multi-robot simulator supports BUILT_IN_PD only, got "
                f"{self.control_type}"
            )

    # ------------------------------------------------------------------
    # Resets: per-block writes + twin re-parking
    # ------------------------------------------------------------------
    def reset_envs(
        self,
        new_states: ResetState,
        new_object_states: Optional[ObjectState] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if (
            new_object_states is not None
            and self.scene_lib.num_objects_per_scene > 0
        ):
            raise NotImplementedError("Scene objects unsupported (multi-robot)")

        self._previous_actions[env_ids] = 0.0
        self._prev_prev_actions[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0

        half = self.num_envs // 2
        in_a = env_ids < half
        for robot, conv, nd, ids, sel in (
            (self._robot, self.data_conversion, self._nd_a, env_ids[in_a], in_a),
            (self._robot_b, self.data_conversion_b, self._nd_b,
             env_ids[~in_a], ~in_a),
        ):
            if ids.numel() == 0:
                continue
            side = ResetState(
                root_pos=new_states.root_pos[sel],
                root_rot=new_states.root_rot[sel],
                root_vel=new_states.root_vel[sel],
                root_ang_vel=new_states.root_ang_vel[sel],
                dof_pos=self._fit_width(new_states.dof_pos[sel], nd),
                dof_vel=self._fit_width(new_states.dof_vel[sel], nd),
                state_conversion=new_states.state_conversion,
            ).convert_to_sim(conv)
            root = torch.cat(
                [side.root_pos, side.root_rot, side.root_vel, side.root_ang_vel],
                dim=-1,
            )
            robot.write_root_state_to_sim(root, ids)
            robot.set_joint_position_target(side.dof_pos, joint_ids=None, env_ids=ids)
            robot.write_joint_state_to_sim(side.dof_pos, side.dof_vel, None, ids)

        # Re-park the reset envs' inactive twins: bounds free-fall drift.
        self._park_twin_rows(self._robot, env_ids[~in_a], TWIN_PARK_Z)
        self._park_twin_rows(self._robot_b, env_ids[in_a], TWIN_PARK_Z - 10.0)

        if self._push_enabled:
            self._simulation_time[env_ids] = 0.0
            self._schedule_push(env_ids)
        self._reset_projectiles(env_ids)


__all__ = ["MultiRobotSceneCfg", "MultiRobotIsaacLabSimulator"]
