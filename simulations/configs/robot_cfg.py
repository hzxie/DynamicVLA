# -*- coding: utf-8 -*-
#
# @File:   robot_cfg.py
# @Author: Haozhe Xie
# @Date:   2025-03-24 16:59:09
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-05-14 13:35:40
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
from robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from robots.piper import AGILEX_PIPER_HIGH_PD_CFG  # isort: skip


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
        close_command_expr={"panda_finger_.*": 0.01},
    )


@configclass
class PiperActionCfg(ActionCfg):
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint[1-6]"],
        body_name="gripper_base",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.107]
        ),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint8", "joint7"],
        open_command_expr={"joint8": -0.035, "joint7": 0.035},
        close_command_expr={"joint8": 0.0, "joint7": 0.0},
    )


def get_body_name(robot: str) -> str:
    if robot == "franka":
        return "panda_hand"
    elif robot == "piper":
        return "gripper_base"
    else:
        raise ValueError("Unknown robot: %s" % robot)


def get_robot_cfg(robot: str) -> ArticulationCfg:
    if robot == "franka":
        cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif robot == "piper":
        cfg = AGILEX_PIPER_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError("Unknown robot: %s" % robot)

    cfg.spawn.semantic_tags = [("class", "ROBOT")]
    return cfg


def get_robot_action_cfg(robot: str) -> ActionCfg:
    if robot == "franka":
        return FrankaActionCfg()
    elif robot == "piper":
        return PiperActionCfg()
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
    elif robot == "piper":
        return FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_base",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1434]),
                ),
            ],
        )
    else:
        raise ValueError("Unknown robot: %s" % robot)


def get_gripper_camera_cfg(robot: str) -> dict:
    if robot == "franka":
        prim_path = "/Robot/panda_hand/GripperCamera"
    elif robot == "piper":
        prim_path = "/Robot/gripper_base/GripperCamera"
    else:
        raise ValueError("Unknown robot: %s" % robot)

    return {
        "prim_path": prim_path,
        "pos": [0.065, 0.0, 0.0],
        "quat": [0, 0.7071068, 0.7071068, 0],
        "convention": "opengl",
    }


def get_gripper_length(robot: str) -> float:
    if robot == "franka":
        return 0.045
    elif robot == "piper":
        return 0.09
    else:
        raise ValueError("Unknown robot: %s" % robot)


def get_robot_name(usd_path: str) -> str:
    if usd_path.endswith("panda_instanceable.usd"):
        return "franka"
    elif usd_path.endswith("piper.usd"):
        return "piper"
    else:
        raise ValueError("Unknown robot name: %s" % usd_path)
