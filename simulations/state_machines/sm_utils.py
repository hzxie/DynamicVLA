# -*- coding: utf-8 -*-
#
# @File:   sm_utils.py
# @Author: Haozhe Xie
# @Date:   2025-09-29 17:04:03
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-09-29 21:50:59
# @Email:  root@haozhexie.com

import numpy as np
import torch
import warp as wp
from isaaclab.utils.math import quat_box_minus

# initialize warp
wp.init()


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication."""
    x1, y1, z1, w1 = q1.unbind(1)
    x2, y2, z2, w2 = q2.unbind(1)
    return torch.stack(
        [
            w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
            w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
            w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
            w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
        ],
        dim=1,
    )


def is_object_static(object_velocity: torch.Tensor) -> torch.Tensor:
    STATIC_VELOCITY_THRESHOLD = 0.1
    return torch.norm(object_velocity, dim=1) < STATIC_VELOCITY_THRESHOLD


def get_grasp_position(
    object_projected_size: torch.Tensor,
    object_position: torch.Tensor,
    object_velocity: torch.Tensor,
    gripper_length: float,
) -> torch.Tensor:
    WAITING_TIME = 0.23
    TABLE_HEIGHT_THRES = 0.006
    OBJECT_HEIGHT_DISPLACEMENT = 0.008
    grasp_position = object_position.clone() + object_velocity * WAITING_TIME
    object_height = torch.norm(object_projected_size[:, 2])
    grasp_position_z = grasp_position[:, 2]

    # Plan B: Try to grasp the center of the object
    grasp_position_z = torch.where(
        object_height > OBJECT_HEIGHT_DISPLACEMENT * 2,
        grasp_position_z - OBJECT_HEIGHT_DISPLACEMENT,
        grasp_position_z,
    )
    grasp_position_z = torch.where(
        object_height > gripper_length * 2,
        object_height - gripper_length,
        grasp_position_z,
    )

    # ensure grasping height higher than table
    grasp_position[:, 2] = torch.where(
        grasp_position_z < TABLE_HEIGHT_THRES,
        TABLE_HEIGHT_THRES,
        grasp_position_z,
    )
    return grasp_position


def get_grasp_quat(
    object_projected_size: torch.Tensor,
    object_velocity: torch.Tensor,
    final_eef_pose: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    # NOTE: Rotation around the z-axis (0, 0, 1)
    #       Quat = [
    #         x * sin(theta / 2),
    #         y * sin(theta / 2),
    #         z * sin(theta / 2),
    #         w * cos(theta / 2)
    #       ]

    # Consider the object quaternion to determine the grasp quaternion for static objects
    if is_object_static(object_velocity):
        object_size_z_rot = torch.abs(object_projected_size[:, 2])
        object_size_z_max = torch.argmax(object_size_z_rot)
        keep_indices = [i for i in range(3) if i != object_size_z_max]
        object_size_xy_rot = object_projected_size[keep_indices, :2]
        object_size_xy_norm = torch.norm(object_size_xy_rot, dim=1)
        if abs(object_size_xy_norm[0] / object_size_xy_norm[1] - 1) < 0.05:
            short_axis = 0
        else:
            short_axis = torch.argmin(object_size_xy_norm)
        grasp_direction = torch.tensor(
            [
                [object_size_xy_rot[short_axis][0]],
                [object_size_xy_rot[short_axis][1]],
            ],
            device=device,
        )
    else:
        grasp_direction = [object_velocity[:, 0], object_velocity[:, 1]]

    # Determine the grasp quaternion according to the velocity
    gsp_theta = torch.arctan2(grasp_direction[0], grasp_direction[1])
    gsp_theta = torch.where(gsp_theta >= np.pi / 2, gsp_theta - np.pi, gsp_theta)
    gsp_theta = torch.where(gsp_theta <= -np.pi / 2, gsp_theta + np.pi, gsp_theta)
    gsp_theta = torch.where(gsp_theta > np.pi / 2 - 1e-2, -np.pi / 2, gsp_theta)
    gsp_quat = torch.stack(
        [
            torch.zeros_like(gsp_theta),
            torch.zeros_like(gsp_theta),
            torch.sin(gsp_theta / 2),
            torch.cos(gsp_theta / 2),
        ],
        dim=1,
    )  # xyzw

    # Consider the basic rotation of the gripper
    return quat_multiply(gsp_quat, final_eef_pose[:, 3:7])


def get_pose_angle(target_quat: torch.Tensor, ee_quat: torch.Tensor) -> torch.Tensor:
    return torch.abs(quat_box_minus(target_quat, ee_quat)[:, 2]) * 180 / np.pi


@wp.func
def get_length(vec: wp.vec3) -> float:
    return wp.length(vec)


@wp.func
def get_offset(vec1: wp.vec3, vec2: wp.vec3) -> wp.vec3:
    return wp.abs(vec1 - vec2)


@wp.func
def get_z_offset(vec1: wp.vec3, vec2: wp.vec3) -> float:
    return wp.abs(vec1[2] - vec2[2])


@wp.func
def is_offset_below_threshold(
    offset: wp.vec3, threshold: wp.vec3, object_vel: wp.vec3
) -> bool:
    is_static = wp.length(object_vel) < 0.05
    if is_static:
        return offset[0] <= threshold[0] and offset[1] <= threshold[1]
    else:
        return offset[2] <= threshold[2]
