# -*- coding: utf-8 -*-
#
# @File:   event_cfg.py
# @Author: Haozhe Xie
# @Date:   2026-01-11 08:01:18
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2026-01-11 14:43:41
# @Email:  root@haozhexie.com

import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass


def apply_external_force_torque_xy(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset = env.scene[asset_cfg.name]
    asset_position = asset.data.root_pos_w
    robot_position = env.scene[robot_cfg.name].data.root_pos_w
    is_lifted = abs(robot_position[:, 2] - asset_position[:, 2]) > 0.05

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )
    size = (len(env_ids), num_bodies, 3)
    random_fdir = math_utils.sample_uniform(-1.0, 1.0, size, asset.device).sign()
    forces = math_utils.sample_uniform(*force_range, size, asset.device) * random_fdir
    random_tdir = math_utils.sample_uniform(-1.0, 1.0, size, asset.device).sign()
    torques = math_utils.sample_uniform(*torque_range, size, asset.device) * random_tdir
    forces[:, :, 2] = 0.0
    torques[:, :, 0] = 0.0

    # set the forces and torques into the buffers
    asset.set_external_force_and_torque(
        forces, torques, env_ids=env_ids[~is_lifted], body_ids=asset_cfg.body_ids
    )


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
    perturbation = EventTermCfg(
        func=apply_external_force_torque_xy,
        mode="interval",
        interval_range_s=(0.5, 1),  # interval is sampled uniformly from this range
        params={
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            "force_range": (-0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )


def get_event_cfg(perturbation_cfg) -> EventCfg:
    if (
        perturbation_cfg is not None
        and "force" in perturbation_cfg
        and "torque" in perturbation_cfg
    ):
        cfg = PertubationEventCfg()
        cfg.perturbation.params["force_range"] = perturbation_cfg["force"]
        cfg.perturbation.params["torque_range"] = perturbation_cfg["torque"]
    else:
        cfg = EventCfg()

    return cfg
