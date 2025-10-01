# -*- coding: utf-8 -*-
#
# @File:   termination_cfg.py
# @Author: Haozhe Xie
# @Date:   2025-09-26 10:24:59
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-01 14:01:56
# @Email:  root@haozhexie.com

import omni.usd
import pxr
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv
from isaaclab_tasks.manager_based.manipulation.lift import mdp


def is_object_picked(
    env: ManagerBasedRLEnv,
    goal_position: torch.Tensor,
    tolerance: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    object = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    robot = env.scene[robot_cfg.name]

    object_position_w = object.data.root_pos_w
    eef_position_w = ee_frame.data.target_pos_w[..., 0, :]

    object_eef_dist = torch.norm(eef_position_w - object_position_w, dim=1)
    goal_position_r = goal_position.to(device=robot.data.root_pos_w.device)
    eef_position_r = quat_apply(
        quat_inv(robot.data.root_quat_w), eef_position_w - robot.data.root_pos_w
    )
    eef_goal_dist = torch.norm(goal_position_r - eef_position_r, dim=1)
    return object_eef_dist < tolerance and eef_goal_dist < tolerance


def is_object_placed(
    env: ManagerBasedRLEnv,
    goal_position: torch.Tensor,
    object_size: torch.Tensor,
    container_size: torch.Tensor,
    tolerance: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    object = env.scene[object_cfg.name]
    container = env.scene[container_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    robot = env.scene[robot_cfg.name]

    object_relative_size = _get_object_relative_bbox(object_size, object.data.root_quat_w) / 2
    object_negz_mask = (object_relative_size[:, 2] > 0).unsqueeze(1)
    object_negz_size = torch.where(object_negz_mask, -object_relative_size, object_relative_size)
    lowest_point = object.data.root_pos_w + object_negz_size.sum(dim=0) / 2

    containier_relative_size = _get_object_relative_bbox(container_size, container.data.root_quat_w) / 2
    object_container_rela = lowest_point - container.data.root_pos_w
    containier_axis_lengths = torch.norm(containier_relative_size, dim=1)
    containier_axis_dirs = containier_relative_size / containier_axis_lengths[:, None]
    object_container_projections = torch.matmul(containier_axis_dirs, object_container_rela[0])

    goal_position_r = goal_position.to(device=robot.data.root_pos_w.device)
    eef_position_r = quat_apply(
        quat_inv(robot.data.root_quat_w),
        ee_frame.data.target_pos_w[..., 0, :] - robot.data.root_pos_w,
    )
    eef_goal_dist = torch.norm(goal_position_r - eef_position_r, dim=1)
    
    return torch.all(torch.abs(object_container_projections) <= containier_axis_lengths) and eef_goal_dist < tolerance


def _get_object_relative_bbox(object_size, object_quat_w):
    object_size_x_rot = quat_apply(
        object_quat_w,
        torch.tensor([[object_size[:, 0], 0.0, 0.0]], device=object_size.device),
    )
    object_size_y_rot = quat_apply(
        object_quat_w,
        torch.tensor([[0.0, object_size[:, 1], 0.0]], device=object_size.device),
    )
    object_size_z_rot = quat_apply(
        object_quat_w,
        torch.tensor([[0.0, 0.0, object_size[:, 2]]], device=object_size.device),
    )
    return torch.cat([object_size_x_rot, object_size_y_rot, object_size_z_rot], dim=0)


def get_done_term(terms: list[str]) -> str | None:
    DONE_TERMS = ["object_picked", "object_placed"]

    for term in DONE_TERMS:
        if term in terms:
            return term

    return None


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    object_dropping = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.1,
            "asset_cfg": SceneEntityCfg("object"),
        },
        time_out=True,
    )


@configclass
class PickTerminationsCfg(TerminationsCfg):
    """Termination terms for the Pick task."""

    object_picked = TerminationTermCfg(
        func=is_object_picked,
        params={"goal_position": None, "tolerance": 0.015},
        time_out=False,
    )


@configclass
class PlaceTerminationsCfg(TerminationsCfg):
    """Termination terms for the Pick task."""

    object_placed = TerminationTermCfg(
        func=is_object_placed,
        params={
            "goal_position": None,
            "object_size": None,
            "container_size": None,
            "tolerance": 0.015,
        },
        time_out=False,
    )


def get_termination_cfg(task: str, args: dict = {}) -> TerminationsCfg:
    done_term = None
    if task == "pick":
        cfg = PickTerminationsCfg()
        done_term = cfg.object_picked
    elif task == "place":
        cfg = PlaceTerminationsCfg()
        done_term = cfg.object_placed
    else:
        cfg = TerminationsCfg()

    # Update the parameters of the done term
    if done_term is not None:
        for k, v in args.items():
            if k in done_term.params:
                done_term.params[k] = v

    return cfg
