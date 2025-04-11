# -*- coding: utf-8 -*-
#
# @File:   robot_cfg.py
# @Author: Haozhe Xie
# @Date:   2025-03-24 16:59:09
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-04-11 20:11:06
# @Email:  root@haozhexie.com

from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class ActionCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: (
        mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg
    ) = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class FrankaActionCfg(ActionCfg):
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.107]
        ),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


def get_body_name(robot: str) -> str:
    if robot == "franka":
        return "panda_hand"
    else:
        raise ValueError("Unknown robot: %s" % robot)


def get_robot_cfg(robot: str) -> ArticulationCfg:
    if robot == "franka":
        return FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError("Unknown robot: %s" % robot)


def get_robot_action_cfg(robot: str) -> ActionCfg:
    if robot == "franka":
        return FrankaActionCfg()
    else:
        raise ValueError("Unknown robot: %s" % robot)


def get_ee_frame_cfg(robot: str) -> FrameTransformerCfg:
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    if robot == "franka":
        return FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )
    else:
        raise ValueError("Unknown robot: %s" % robot)


def get_gripper_camera_cfg(robot: str) -> dict:
    if robot == "franka":
        return {
            "prim_path": "/Robot/panda_hand/GripperCamera",
            "pos": [0.065, 0.0, 0.0],
            "quat": [0, 0.7071068, 0.7071068, 0],
            "convention": "opengl",
        }
    else:
        raise ValueError("Unknown robot: %s" % robot)
