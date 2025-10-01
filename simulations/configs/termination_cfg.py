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

    # object_obb = _get_obb_vertices(
    #     object_size, object.data.root_pos_w, object.data.root_quat_w
    # )
    # container_obb = _get_obb_vertices(
    #     container_size, container.data.root_pos_w, container.data.root_quat_w
    # )
    # object_container_dist = torch.norm(container_position_w - object_position_w, dim=1)

    goal_position_r = goal_position.to(device=robot.data.root_pos_w.device)
    eef_position_r = quat_apply(
        quat_inv(robot.data.root_quat_w),
        ee_frame.data.target_pos_w[..., 0, :] - robot.data.root_pos_w,
    )
    eef_goal_dist = torch.norm(goal_position_r - eef_position_r, dim=1)
    # print(object_container_dist, eef_goal_dist, tolerance)
    # return object_container_dist < tolerance and eef_goal_dist < tolerance
    return eef_goal_dist < tolerance


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


# def _get_obb_vertices(
#     size: torch.Tensor, center: torch.Tensor, quat: torch.Tensor
# ) -> torch.Tensor:
#     batch_size = center.shape[0]
#     signs = torch.tensor(
#         [
#             [-1, -1, -1],
#             [-1, -1, 1],
#             [-1, 1, -1],
#             [-1, 1, 1],
#             [1, -1, -1],
#             [1, -1, 1],
#             [1, 1, -1],
#             [1, 1, 1],
#         ],
#         device=center.device,
#         dtype=center.dtype,
#     )  # [8,3]
#     signs = signs.unsqueeze(0).expand(batch_size, 8, 3)  # [B,8,3]
#     local_verts = signs * size.unsqueeze(1) / 2  # [B,8,3]
#     rotated_verts = _quat_rotate_batch(quat, local_verts)  # [B,8,3]
#     return rotated_verts + center.unsqueeze(1)  # [B,8,3]


# def _quat_rotate_batch(quat: torch.Tensor, vecs: torch.Tensor) -> torch.Tensor:
#     w, x, y, z = quat[:, 0:1], quat[:, 1:2], quat[:, 2:3], quat[:, 3:4]  # [B,1]
#     R = torch.zeros((quat.shape[0], 3, 3), device=vecs.device, dtype=vecs.dtype)
#     R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
#     R[:, 0, 1] = 2 * (x * y - z * w)
#     R[:, 0, 2] = 2 * (x * z + y * w)
#     R[:, 1, 0] = 2 * (x * y + z * w)
#     R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
#     R[:, 1, 2] = 2 * (y * z - x * w)
#     R[:, 2, 0] = 2 * (x * z - y * w)
#     R[:, 2, 1] = 2 * (y * z + x * w)
#     R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
#     # [B,N,3] @ [B,3,3] -> [B,N,3]
#     return torch.bmm(vecs, R.transpose(1, 2))


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
