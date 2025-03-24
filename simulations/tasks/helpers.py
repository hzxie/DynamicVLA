# -*- coding: utf-8 -*-
#
# @File:   helpers.py
# @Author: Haozhe Xie
# @Date:   2025-03-22 21:01:04
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-03-22 21:02:04
# @Email:  root@haozhexie.com

import numpy as np
import torch


def get_waiting_time(
    object_position: torch.Tensor, object_velocity: torch.Tensor, future_position
):
    # Decide if we need to change the position
    object_position = object_position[0]
    object_velocity = object_velocity[0]
    future_position = torch.tensor(future_position, device="cuda:0")
    velocity_norm_sq = torch.dot(object_velocity, object_velocity)
    if velocity_norm_sq == 0:
        return True, float("inf")
    # If not, compute the time
    displacement = future_position - object_position
    t_closest = torch.dot(displacement, object_velocity) / velocity_norm_sq
    closest_point = object_position + t_closest * object_velocity
    distance = torch.norm(closest_point - future_position)
    if distance > 0.05 or t_closest.item() < 0:
        return True, float("inf")
    else:
        return False, t_closest.item()


def get_future_position(
    object_position: torch.Tensor, object_velocity: torch.Tensor, waiting_time: float
):
    # this is different, the waiting time is scheduled
    future_position = object_position[0] + object_velocity[0] * waiting_time
    return future_position.tolist(), waiting_time


def orientation_to_quaternion(object_velocity: torch.Tensor):
    # regularize
    object_velocity = object_velocity[0].cpu().numpy()
    if object_velocity[0] < 0:
        velo_x = -object_velocity[0]
        velo_y = -object_velocity[1]
    else:
        velo_x = object_velocity[0]
        velo_y = object_velocity[1]

    # velocity to orientation
    A = np.array([velo_x, velo_y, 0.0])
    B = np.array([0, 1, 0])
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    cos_theta = dot_product / (norm_A * norm_B)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    # To quaternion
    roll = np.radians(179.995)
    pitch = 0
    yaw = np.arccos(cos_theta) - (np.pi / 2)
    cr, cp, cy = torch.cos(torch.tensor([roll, pitch, yaw]) / 2)
    sr, sp, sy = torch.sin(torch.tensor([roll, pitch, yaw]) / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return float(w), float(x), float(y), float(z)
