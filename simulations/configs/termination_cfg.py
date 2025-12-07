# -*- coding: utf-8 -*-
#
# @File:   termination_cfg.py
# @Author: Haozhe Xie
# @Date:   2025-09-26 10:24:59
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-12-06 10:45:26
# @Email:  root@haozhexie.com

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift import mdp

from simulations import helpers
from typing import Dict


def is_object_picked(
    env: ManagerBasedRLEnv,
    goal_position: torch.Tensor,
    tolerance: float,
    objects: list[str] = ["object"],
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    assert len(objects) == 1, "Only single object picking is supported."
    object = env.scene[objects[0]]
    ee_frame = env.scene[ee_frame_cfg.name]
    robot = env.scene[robot_cfg.name]

    object_position_w = object.data.root_pos_w
    eef_position_w = ee_frame.data.target_pos_w[..., 0, :]

    object_eef_dist = torch.norm(eef_position_w - object_position_w, dim=1)
    goal_position_r = goal_position.to(device=robot.data.root_pos_w.device)
    eef_position_r = helpers.get_robot_relative_position(
        ee_frame.data.target_pos_w[..., 0, :] - robot.data.root_pos_w,
        robot.data.root_quat_w,
    )
    eef_goal_dist = torch.norm(goal_position_r - eef_position_r, dim=1)
    return object_eef_dist < tolerance and eef_goal_dist < tolerance


def are_objects_placed(
    env: ManagerBasedRLEnv,
    goal_position: torch.Tensor,
    objects: list[str],
    object_sizes: Dict[str, torch.Tensor],
    container_size: torch.Tensor,
    tolerance: float,
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    objects_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    container = env.scene[container_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    robot = env.scene[robot_cfg.name]
    env_origins = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    container_position = helpers.get_robot_relative_position(
        container.data.root_pos_w - env_origins, robot_quat
    )
    containier_size = helpers.get_object_relative_bbox(
        container_size, container.data.root_quat_w, robot_quat
    )
    for obj in objects:
        object = env.scene[obj]
        object_position = helpers.get_robot_relative_position(
            object.data.root_pos_w - env_origins, robot_quat
        )
        object_size = helpers.get_object_relative_bbox(
            object_sizes[obj], object.data.root_quat_w, robot_quat
        )
        objects_placed = torch.logical_and(
            objects_placed,
            helpers.is_object_placed(
                object_position,
                object_size,
                container_position,
                containier_size,
            ),
        )

    goal_position_r = goal_position.to(device=env_origins.device)
    eef_position_r = helpers.get_robot_relative_position(
        ee_frame.data.target_pos_w[..., 0, :] - env_origins, robot_quat
    )
    eef_goal_dist = torch.norm(goal_position_r - eef_position_r, dim=1)
    return torch.logical_and(objects_placed, eef_goal_dist < tolerance)


def are_objects_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    objects: list[str],
) -> torch.Tensor:
    object_dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for obj in objects:
        _dropped = mdp.root_height_below_minimum(
            env, minimum_height, SceneEntityCfg(obj)
        )
        object_dropped = torch.logical_or(object_dropped, _dropped)

    return object_dropped


def are_objects_unreachable(
    env: ManagerBasedRLEnv,
    max_reach_dist: float,
    objects: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    object_unreachable = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    robot = env.scene[robot_cfg.name]
    env_origins = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    for obj in objects:
        object = env.scene[obj]
        object_position = helpers.get_robot_relative_position(
            object.data.root_pos_w - env_origins, robot_quat
        )
        obj_dist = torch.norm(object_position)
        object_unreachable = torch.logical_or(
            object_unreachable,
            obj_dist > max_reach_dist
        )
    return object_unreachable



def get_done_term(terms: list[str]) -> str | None:
    DONE_TERMS = ["object_picked", "objects_placed"]

    for term in DONE_TERMS:
        if term in terms:
            return term

    return None


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    object_dropping = TerminationTermCfg(
        func=are_objects_dropped,
        params={
            "minimum_height": 0.1,
            "objects": ["object"],
        },
        time_out=True,
    )
    object_unreachable = TerminationTermCfg(
        func=are_objects_unreachable,
        params={
            "max_reach_dist": 0,
            "objects": ["object"],
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

    objects_placed = TerminationTermCfg(
        func=are_objects_placed,
        params={
            "goal_position": None,
            "objects": None,
            "object_sizes": None,
            "container_size": None,
            "tolerance": 0.015,
        },
        time_out=False,
    )
    container_dropping = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.1,
            "asset_cfg": SceneEntityCfg("container"),
        },
        time_out=True,
    )


def get_termination_cfg(task: str, args: dict = {}) -> TerminationsCfg:
    done_term = None
    if task == "pick":
        cfg = PickTerminationsCfg()
        done_term = cfg.object_picked
    elif task in ["place", "long_horizon"]:
        cfg = PlaceTerminationsCfg()
        done_term = cfg.objects_placed
    else:
        cfg = TerminationsCfg()

    for k, v in args.items():
        if k in cfg.object_dropping.params:
            cfg.object_dropping.params[k] = v
        if k in cfg.object_unreachable.params:
            cfg.object_unreachable.params[k] = v

    # Update the parameters of the done term
    if done_term is not None:
        for k, v in args.items():
            if k in done_term.params:
                done_term.params[k] = v

    return cfg
