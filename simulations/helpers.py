# -*- coding: utf-8 -*-
#
# @File:   helpers.py
# @Author: Haozhe Xie
# @Date:   2025-10-03 19:04:52
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-03 19:24:56
# @Email:  root@haozhexie.com

import torch


def get_object_relative_bbox(object_size, object_quat_w, robot_quat):
    from isaaclab.utils.math import quat_apply

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
    object_size_x_rot = get_robot_relative_position(object_size_x_rot, robot_quat)
    object_size_y_rot = get_robot_relative_position(object_size_y_rot, robot_quat)
    object_size_z_rot = get_robot_relative_position(object_size_z_rot, robot_quat)
    return torch.cat([object_size_x_rot, object_size_y_rot, object_size_z_rot], dim=0)


def get_robot_relative_position(point, robot_quat):
    from isaaclab.utils.math import quat_apply, quat_inv

    # inv_quat = scipy.spatial.transform.Rotation.from_quat(robot_quat).inv()
    # inv_offset = inv_quat.apply(point)
    return quat_apply(quat_inv(robot_quat), point)
