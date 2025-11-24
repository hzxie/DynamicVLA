# -*- coding: utf-8 -*-
#
# @File:   helpers.py
# @Author: Haozhe Xie
# @Date:   2025-10-03 19:04:52
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-11-24 10:16:45
# @Email:  root@haozhexie.com

import math
import numpy as np
import torch

from scipy.spatial.transform import Rotation as R


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
    object_lowest_z = object_position[:, 2] - torch.sum(
        torch.abs(object_relative_size[:, :, 2]), dim=1
    )
    container_highest_z = container_position[:, 2] + torch.sum(
        torch.abs(container_relative_size[:, :, 2]), dim=1
    )
    is_vertical_in_container = object_lowest_z <= container_highest_z

    return torch.logical_and(is_horizonal_in_container, is_vertical_in_container)


def get_object_tags(object_type, object_states, robot_pose, skip_tags, tag_thresholds):
    robot_quat_xyzw = np.roll(robot_pose["quat"].astype(np.float32), -1)
    _get_relative_pos = lambda point: R.from_quat(robot_quat_xyzw).apply(
        point - robot_pose["pos"], inverse=True
    )
    TAG_FUNCTIONS = {
        "HEIGHT": lambda x: x["pos"][2],
        "AREA": lambda x: x["size"][0] * x["size"][1],
        "VOLUME": lambda x: np.prod(x["size"]),
        "POSITION_FROM_LEFT": lambda x: _get_relative_pos(x["pos"])[1],
        "POSITION_FROM_BOTTOM": lambda x: -_get_relative_pos(x["pos"])[0],
        "DISTANCE_FROM_ROBOT": lambda x: -np.linalg.norm(x["pos"] - robot_pose["pos"]),
        "VELOCITY": lambda x: (np.linalg.norm(x["lin_vel"]) if "lin_vel" in x else 0),
    }
    assert object_type in [
        "objects",
        "containers",
    ], f"Unknown object type: {object_type}"

    if len(object_states) > 1:
        for tag, func in TAG_FUNCTIONS.items():
            if skip_tags is None or tag not in skip_tags:
                object_states = _get_state_tag(
                    object_type, object_states, tag, func, tag_thresholds
                )
        # Generate additional direction tags
        if skip_tags is None or "VELOCITY" not in skip_tags:
            object_states = _get_direction_tags(
                object_type, object_states, robot_quat_xyzw
            )
    # Remove duplicate tags (causing confusion in instruction generation)
    return _get_unique_tags([os["tags"] for os in object_states])


def _get_state_tag(object_type, object_states, tag_name, tag_func, tag_thresholds):
    RANK_TAGS = {
        "FIRST": {
            "HEIGHT": "the tallest %s",
            "AREA": "the %s with the largest area",
            "VOLUME": "the %s with the largest volume",
            "POSITION_FROM_LEFT": "the %s that is farthest to the robot's left at the start",
            "POSITION_FROM_BOTTOM": "the %s closest to the robot mounting edge at the start",
            "DISTANCE_FROM_ROBOT": "the %s closest to the robot at the start",
            "VELOCITY": "the %s with the highest initial velocity",
        },
        "LAST": {
            "HEIGHT": "the shortest %s",
            "AREA": "the %s with the smallest area",
            "VOLUME": "the %s with the smallest volume",
            "POSITION_FROM_LEFT": "the %s that is farthest to the robot's right at the start",
            "POSITION_FROM_BOTTOM": "the %s farthest from the robot mounting edge at the start",
            "DISTANCE_FROM_ROBOT": "the %s farthest from the robot at the start",
            "VELOCITY": "the %s with the lowest initial velocity",
        },
        "MEDIUM": {
            "HEIGHT": "the %s of medium height",
            "AREA": "the %s with medium area",
            "VOLUME": "the %s with medium volume",
            "POSITION_FROM_LEFT": "the %s in the middle from left to right at the start",
            "POSITION_FROM_BOTTOM": "the %s with medium distance to the robot mounting edge at the start",
            "DISTANCE_FROM_ROBOT": "the %s with medium distance to the robot at the start",
            "VELOCITY": "the %s with medium initial velocity",
        },
    }

    n = len(object_states)
    sorted_states = sorted(object_states, key=tag_func, reverse=True)
    last_value = tag_func(sorted_states[-1])
    cur_rank = 1
    for i, item in enumerate(sorted_states):
        object_name = object_type.rstrip("s")  # singular form
        cur_value = tag_func(item)
        if abs(cur_value - last_value) > tag_thresholds.get(tag_name.lower()):
            cur_rank = i + 1
        if cur_rank == 1:
            item["tags"].append(RANK_TAGS["FIRST"][tag_name] % object_name)
        elif cur_rank == n and n != 1:
            item["tags"].append(RANK_TAGS["LAST"][tag_name] % object_name)
        elif cur_rank == 2 and n == 3:
            item["tags"].append(RANK_TAGS["MEDIUM"][tag_name] % object_name)

        last_value = cur_value

    return object_states


def _get_direction_tags(object_type, object_states, robot_quat):
    DIRECTION_TAGS = [
        "the %s moving in the robot's forward direction",                 # front
        "the %s moving in the robot's forward-left direction",            # front-left
        "the %s moving in the robot's left direction",                    # left
        "the %s moving in the robot's backward-left direction",           # back-left
        "the %s moving in the robot's backward direction",                # back
        "the %s moving in the robot's backward-right direction",          # back-right
        "the %s moving in the robot's right direction",                   # right
        "the %s moving in the robot's forward-right direction",           # front-right
    ]

    object_name = object_type.rstrip("s")  # singular form
    for state in object_states:
        if "lin_vel" not in state or np.linalg.norm(state["lin_vel"]) < 1e-3:
            state["tags"].append("stationary %s" % object_name)
            continue

        rel_velocity = R.from_quat(robot_quat).inv().as_matrix() @ state["lin_vel"]
        angle = math.degrees(math.atan2(rel_velocity[1], rel_velocity[0])) % 360
        # idx = int((angle + 22.5) // 45) % 8  # Old version
        # front: [345°, 360) U [0°, 15°); back:  [165°, 195°); left:  [75°, 105°); 
        # right: [255, 285°)
        if angle >= 345 or angle < 15:
            idx = 0  # front
        elif angle >= 15 and angle < 75:
            idx = 1  # front-left
        elif angle >= 75 and angle < 105:
            idx = 2  # left (20°)
        elif angle >= 105 and angle < 165:
            idx = 3  # back-left
        elif angle >= 165 and angle < 195:
            idx = 4  # back (20°)
        elif angle >= 195 and angle < 255:
            idx = 5  # back-right
        elif angle >= 255 and angle < 285:
            idx = 6  # right (20°)
        elif angle >= 285 and angle < 345:
            idx = 7  # front-right

        state["tags"].append(DIRECTION_TAGS[idx] % object_name)

    return object_states


def _get_unique_tags(object_tags):
    assert isinstance(object_tags, list)
    if len(object_tags) == 0:
        return []

    target_tags = set(object_tags[0])
    other_tags = set(tag for obj in object_tags[1:] for tag in obj)
    return list(target_tags - other_tags)
