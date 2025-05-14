# -*- coding: utf-8 -*-
#
# @File:   evaluate.py
# @Author: Haozhe Xie
# @Date:   2025-05-06 15:21:20
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-05-14 14:21:24
# @Email:  root@haozhexie.com

import argparse
import json
import logging
import os
import sys

import cv2
import gymnasium as gym
import h5py
import isaaclab.app
import numpy as np
import random
import scipy.spatial.transform
import torch
import yaml
import zmq

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.dirname(__file__))

import simulations.simulate as sim


def get_message_queues(host, img_port, act_port):
    context = zmq.Context()

    img_socket = context.socket(zmq.PUB)
    img_socket.bind("tcp://%s:%d" % (host, img_port))

    act_socket = context.socket(zmq.PULL)
    act_socket.bind("tcp://%s:%d" % (host, act_port))

    return img_socket, act_socket


def get_test_env(
    env_cfg,
    num_envs,
    scene_dir,
    object_dir,
    physics_time_step,
    timeout,
    device,
    disable_fabric,
    path_tracing,
):
    import omni.replicator.core as rep

    # Load the environment configuration
    with open(env_cfg, "r") as fp:
        cfg = json.load(fp)

    # Create the environment
    robot_name, env_cfg = _get_env_cfg(
        cfg, num_envs, scene_dir, object_dir, device, disable_fabric
    )
    env_cfg.dt = physics_time_step
    env_cfg.episode_length_s = timeout
    env = gym.make("Robot-Env-Cfg-v0", cfg=env_cfg, seed=cfg["seed"])
    # Reset environment at start
    env.reset(seed=cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # Enable Path Tracing
    if path_tracing:
        rep.settings.set_render_pathtraced()

    return robot_name, env


def _get_env_cfg(cfg, num_envs, scene_dir, object_dir, device, disable_fabric):
    import configs.env_cfg
    import configs.robot_cfg
    import configs.scene_cfg
    import isaaclab_tasks

    gym.register(
        id="Robot-Env-Cfg-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": "configs.env_cfg:EnvCfg",
        },
        disable_env_checker=True,
    )
    env_cfg = isaaclab_tasks.utils.parse_cfg.parse_env_cfg(
        "Robot-Env-Cfg-v0",
        device=device,
        num_envs=num_envs,
        use_fabric=not disable_fabric,
    )

    scene_usd_path = os.path.join(
        scene_dir, os.path.basename(cfg["scene"]["house"]["spawn"]["usd_path"])
    )
    logging.info("Loading scene from %s" % scene_usd_path)
    env_cfg.scene = configs.scene_cfg.set_house_asset(
        env_cfg.scene, os.path.join(scene_dir, scene_usd_path)
    )

    # Set up the robot arm
    robot_name = configs.robot_cfg.get_robot_name(
        cfg["scene"]["robot"]["spawn"]["usd_path"]
    )
    env_cfg = configs.env_cfg.set_robot(
        robot_name,
        env_cfg,
        {
            "pos": cfg["scene"]["robot"]["init_state"]["pos"],
            "quat": cfg["scene"]["robot"]["init_state"]["rot"],
        },
    )
    # Set up cameras in the scene
    env_cfg.scene = _set_up_scene_cameras(env_cfg.scene, cfg["scene"])

    # Set the light intensity and color
    assert "distant_light" in cfg["scene"]
    env_cfg.scene = _set_up_scene_distant_light(
        env_cfg.scene, cfg["scene"]["distant_light"]
    )

    # Dynamically add objects to scene
    env_cfg.scene = _set_up_scene_objects(env_cfg.scene, cfg["scene"], object_dir)
    return robot_name, env_cfg


def _set_up_scene_cameras(scene_cfg, cfg):
    import configs.scene_cfg

    # Set up the top-view camera
    for k, v in cfg.items():
        if not k.endswith("_cam"):
            continue

        # Remove prefix: '/World/envs/env_.*'
        prim_path = v["prim_path"]
        prim_path = prim_path[prim_path.rfind("/Robot") :]

        scene_cfg = configs.scene_cfg.add_scene_camera(
            scene_cfg,
            k,
            configs.scene_cfg.get_camera_cfg(
                {
                    "prim_path": prim_path,
                    "fps": 1 / v["update_period"],
                    "width": v["width"],
                    "height": v["height"],
                    "data_types": v["data_types"],
                    "focal_length": v["spawn"]["focal_length"],
                    "focus_distance": v["spawn"]["focus_distance"],
                    "horizontal_aperture": v["spawn"]["horizontal_aperture"],
                    "clip": {
                        "near": v["spawn"]["clipping_range"][0],
                        "far": v["spawn"]["clipping_range"][1],
                    },
                    "pos": v["offset"]["pos"],
                    "quat": v["offset"]["rot"],
                    "convention": v["offset"]["convention"],
                }
            ),
        )

    return scene_cfg


def _set_up_scene_distant_light(scene_cfg, cfg):
    import configs.object_cfg
    import configs.scene_cfg

    scene_cfg = configs.scene_cfg.set_light_asset(
        scene_cfg,
        position=cfg["init_state"]["pos"],
        temperature=cfg["spawn"]["color_temperature"],
        intensity=cfg["spawn"]["intensity"],
    )
    return scene_cfg


def _set_up_scene_objects(scene_cfg, cfg, object_dir):
    import configs.object_cfg

    for k, v in cfg.items():
        if not (
            isinstance(v, dict)
            and "class_type" in v
            and v["class_type"]
            == "isaaclab.assets.rigid_object.rigid_object:RigidObject"
        ):
            continue

        usd_file_path = None
        if "usd_path" in v["spawn"]:
            usd_folder = os.path.basename(os.path.dirname(v["spawn"]["usd_path"]))
            usd_file_path = os.path.join(
                object_dir, usd_folder, os.path.basename(v["spawn"]["usd_path"])
            )
            logging.info("Loading object from %s" % usd_file_path)
            assert os.path.exists(usd_file_path)

        object_cfg = configs.object_cfg.get_object_cfg(
            {
                "pos": v["init_state"]["pos"],
                "quat": v["init_state"]["rot"],
                "lin_vel": v["init_state"]["lin_vel"],
                "ang_vel": v["init_state"]["ang_vel"],
            },
            configs.object_cfg.get_spawner_cfg(
                usd_file_path,
                v["spawn"]["mass_props"]["mass"],
                v["spawn"]["semantic_tags"],
            ),
        )
        if k == "object":
            scene_cfg = configs.scene_cfg.set_target_object(scene_cfg, object_cfg)
        else:
            scene_cfg = configs.scene_cfg.set_target_object(scene_cfg, k, object_cfg)

    return scene_cfg


def main(simulation_app, args):
    logging.info("Starting evaluation server...")
    # Set up Zero MQ context and sockets
    img_mq, act_mq = get_message_queues(args.host, args.img_port, args.act_port)
    logging.info(
        "ZeroMQs are listensing on %s:%d for images and %s:%d for actions"
        % (args.host, args.img_port, args.host, args.act_port)
    )

    # Set up test environment
    logging.info("Recovering test environment from %s" % args.env_cfg)
    robot_name, env = get_test_env(
        args.env_cfg,
        args.num_envs,
        args.scene_dir,
        args.object_dir,
        args.physics_time_step,
        args.timeout,
        args.device,
        args.disable_fabric,
        args.path_tracing,
    )

    # Simulation loop
    robot_origin = (
        torch.from_numpy(np.array(env.unwrapped.cfg.scene.robot.init_state.pos))
        .unsqueeze(0)
        .float()
        .to(env.unwrapped.device)
    )
    robot_quat = (
        torch.from_numpy(np.array(env.unwrapped.cfg.scene.robot.init_state.rot))
        .unsqueeze(0)
        .float()
        .to(env.unwrapped.device)
    )

    last_action = None
    while simulation_app.is_running():
        cam_views = sim.get_camera_views(env.unwrapped.scene.sensors)
        img_mq.send_pyobj(cam_views)
        try:
            while True:
                action = act_mq.recv(flags=zmq.NOBLOCK)
                last_action = action
        except zmq.Again:
            # No more messages in the queue
            pass

        # If no action is received, use the previous action to make the 
        # simulation continuous
        if last_action is None:
            curr_state = sim.get_curr_state(
                env.unwrapped.scene["ee_frame"].data,
                None,
                env.unwrapped.scene.env_origins + robot_origin,
                robot_quat,
            )
            last_action = torch.cat(
                [
                    curr_state["end_effector"]["pos"],
                    curr_state["end_effector"]["quat"],
                    torch.ones(args.num_envs, 1, device=env.unwrapped.device),
                ],
                dim=1,
            )

        env.step(last_action)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    SHARED_PARAMETERS = ["num_envs", "save"]

    parser = argparse.ArgumentParser(description="Evaluation Server Runner")
    # Arguments for the IsaacLab
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the data from camera at index specified by ``--camera_id``.",
    )
    isaaclab.app.AppLauncher.add_app_launcher_args(parser)
    isaaclab_args, script_args = parser.parse_known_args()

    # Arguments for the script
    parser.add_argument("--path_tracing", action="store_true")
    parser.add_argument("--physics_time_step", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument(
        "--scene_dir", default=os.path.join(PROJECT_HOME, os.pardir, "scenes")
    )
    parser.add_argument(
        "--object_dir", default=os.path.join(PROJECT_HOME, os.pardir, "objects")
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(PROJECT_HOME, os.pardir, "datasets")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--img_port", default=3186, type=int, help="Port for image stream"
    )
    parser.add_argument(
        "--act_port", default=3188, type=int, help="Port for action stream"
    )
    parser.add_argument("--env_cfg", required=True)
    args = parser.parse_args(script_args)
    # Copy the shared parameters from isaaclab_args to args
    for sp in SHARED_PARAMETERS:
        if sp in isaaclab_args:
            setattr(args, sp, getattr(isaaclab_args, sp))

    app_launcher = isaaclab.app.AppLauncher(isaaclab_args)
    main(app_launcher.app, args)
    app_launcher.app.close()
