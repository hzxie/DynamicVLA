# -*- coding: utf-8 -*-
#
# @File:   termination_cfg.py
# @Author: Haozhe Xie
# @Date:   2025-09-26 10:24:59
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-09-27 11:21:48
# @Email:  root@haozhexie.com

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv
from isaaclab_tasks.manager_based.manipulation.lift import mdp


def is_object_picked(
    env: ManagerBasedRLEnv,
    goal_position: torch.Tensor,
    tolerance: float = 0.015,
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
        params={"goal_position": None},
        time_out=False,
    )


def get_termination_cfg(task: str, args: dict = {}) -> TerminationsCfg:
    if task == "pick":
        cfg = PickTerminationsCfg()
        for k, v in args.items():
            cfg.object_picked.params[k] = v
    elif task == "place":
        cfg = TerminationsCfg()
    else:
        cfg = TerminationsCfg()

    return cfg
