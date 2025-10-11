# -*- coding: utf-8 -*-
#
# @File:   helpers.py
# @Author: Haozhe Xie
# @Date:   2025-10-03 19:04:52
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-04 16:39:47
# @Email:  root@haozhexie.com

import torch


def get_object_relative_bbox(object_size, object_quat_w, robot_quat):
    from isaaclab.utils.math import quat_apply

    batch_size = object_quat_w.size(0)
    object_size_rot = torch.eye(3, device=object_size.device) * object_size
    object_size_x_rot = quat_apply(
        object_quat_w,
        object_size_rot[0:1, :].repeat(batch_size, 1),
    )
    object_size_y_rot = quat_apply(
        object_quat_w,
        object_size_rot[1:2, :].repeat(batch_size, 1),
    )
    object_size_z_rot = quat_apply(
        object_quat_w,
        object_size_rot[2:3, :].repeat(batch_size, 1),
    )
    return torch.cat(
        [
            get_robot_relative_position(object_size_x_rot, robot_quat).unsqueeze(1),
            get_robot_relative_position(object_size_y_rot, robot_quat).unsqueeze(1),
            get_robot_relative_position(object_size_z_rot, robot_quat).unsqueeze(1),
        ],
        dim=1,
    )


def get_robot_relative_position(point, robot_quat):
    from isaaclab.utils.math import quat_apply, quat_inv

    # inv_quat = scipy.spatial.transform.Rotation.from_quat(robot_quat).inv()
    # inv_offset = inv_quat.apply(point)
    return quat_apply(quat_inv(robot_quat), point)


def is_object_placed(
    object_position: torch.Tensor,
    object_projected_size: torch.Tensor,
    container_position: torch.Tensor,
    container_projected_size: torch.Tensor,
    tolerance: float = 0.015,
) -> torch.Tensor:
    # Horizonal
    container_relative_size = container_projected_size / 2 + tolerance
    container_axis_lengths = torch.norm(container_relative_size, dim=2)
    container_axis_dirs = container_relative_size / container_axis_lengths.unsqueeze(2)

    object_container_rela = object_position - container_position
    object_container_projections = torch.matmul(
        container_axis_dirs, object_container_rela.unsqueeze(-1)
    ).squeeze(-1)

    object_container_projections_xy = object_container_projections[:, :2]
    container_axis_lengths_xy = container_axis_lengths[:, :2]
    is_horizonal_in_container = torch.all(
        torch.abs(object_container_projections_xy) <= container_axis_lengths_xy, dim=1
    )

    # Vertical
    object_relative_size = object_projected_size / 2
    object_lowest_z = object_position[:, 2] - torch.sum(torch.abs(object_relative_size[:, :, 2]), dim=1)
    container_highest_z = container_position[:, 2] + torch.sum(torch.abs(container_relative_size[:, :, 2]), dim=1)
    is_vertical_in_container = object_lowest_z <= container_highest_z

    return torch.logical_and(is_horizonal_in_container, is_vertical_in_container)
