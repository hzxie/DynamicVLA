# -*- coding: utf-8 -*-
#
# @File:   event_cfg.py
# @Author: Haozhe Xie
# @Date:   2026-01-11 08:01:18
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2026-01-11 12:52:19
# @Email:  root@haozhexie.com

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils


def pertube_linear_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot_position = env.scene[robot_cfg.name].data.root_pos_w
    object_position = env.scene[asset_cfg.name].data.root_pos_w

    # Only apply perturbation before lifting
    for dim, (low, high) in range.items():
        dim_idx = {"x": 0, "y": 1, "z": 2}[dim]
        is_lifted = abs(robot_position[:, 2] - object_position[:, 2]) > 0.05
        random_dir = torch.empty(env.num_envs, device=env.device).uniform_(-1.0, 1.0).sign()
        random_offset = torch.empty(env.num_envs, device=env.device).uniform_(low, high) * random_dir
        random_offset = random_offset * (~is_lifted).to(torch.float32)

        if env_ids is None:
            object_position[:, dim_idx] += random_offset
        else:
            object_position[env_ids, dim_idx] += random_offset


def apply_external_force_torque_xy(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
):
    # get target object
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques for x and y
    size = (len(env_ids), num_bodies, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    forces[:, :, 2] = 0.0
    torques[:, :, 0] = 0.0

    # set the forces and torques into the buffers
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class PertubationEventCfg(EventCfg):
    linear_velocity_perturbation = EventTermCfg(
        func=apply_external_force_torque_xy,
        mode="interval",
        interval_range_s=(0.1, 0.2),  # interval is sampled uniformly from this range
        params={
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            "force_range": (-0.5, 0.5),
            "torque_range": (-0.1, 0.1),
        },
    )


def get_event_cfg(perturbation_range) -> EventCfg:
    if perturbation_range is not None:
        cfg = PertubationEventCfg()
        # cfg.linear_velocity_perturbation.params["range"]["x"] = perturbation_range
        # cfg.linear_velocity_perturbation.params["range"]["y"] = perturbation_range
    else:
        cfg = EventCfg()

    return cfg
