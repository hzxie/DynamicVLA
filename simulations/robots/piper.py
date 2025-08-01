# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the PIPER robots.

The following configurations are available:

* :obj:`AGILEX_PIPER_CFG`: PIPER robot
* :obj:`AGILEX_PIPER_HIGH_PD_CFG`: PIPER robot with stiffer PD control

Reference: https://github.com/agilexrobotics/PIPER_ros
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

AGILEX_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "PIPER", "piper.usd")
        ),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
            rest_offset=0.0,
            min_torsional_patch_radius=0.01,
            torsional_patch_radius=0.01,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 1.57,
            "joint3": -1.57,
            "joint4": 0.0,
            "joint5": 1.2,
            "joint6": 0.0,
            "joint7": 0.035,
            "joint8": -0.035,
        },
    ),
    actuators={
        "piper_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "piper_forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint[4-6]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["joint8", "joint7"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.2,
            stiffness=2e3,
            damping=5e2,
            friction=50,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of PIPER robot."""


AGILEX_PIPER_HIGH_PD_CFG = AGILEX_PIPER_CFG.copy()
AGILEX_PIPER_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
AGILEX_PIPER_HIGH_PD_CFG.actuators["piper_shoulder"].stiffness = 400.0
AGILEX_PIPER_HIGH_PD_CFG.actuators["piper_shoulder"].damping = 125.0
AGILEX_PIPER_HIGH_PD_CFG.actuators["piper_forearm"].stiffness = 400.0
AGILEX_PIPER_HIGH_PD_CFG.actuators["piper_forearm"].damping = 125.0
"""Configuration of PIPER robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
