# -*- coding: utf-8 -*-
#
# @File:   simulate.py
# @Author: Haozhe Xie
# @Date:   2025-03-22 20:59:36
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-04-02 15:52:19
# @Email:  root@haozhexie.com
"""
Script to run an environment with an action state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p simulations/simulate.py --enable_cameras

"""

import argparse
import logging
import os
import random
import scipy.spatial.transform
import sys

import gymnasium as gym
import isaaclab.app

# import omni.replicator.core
import numpy as np
import torch
import yaml

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.dirname(__file__))


def get_env_cfg(scene_dir, sim_cfg, robot):
    # The following packages MUST be imported after the simulation app is created
    import configs.env_cfg
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
    env_cfg: configs.env_cfg.EnvCfg = isaaclab_tasks.utils.parse_cfg.parse_env_cfg(
        "Robot-Env-Cfg-v0",
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )

    table = None
    while table is None:
        # Dynamically create basic scene from USD files
        usd_file = os.path.join(scene_dir, random.choice(os.listdir(scene_dir)))
        logging.info("Loading scene from %s", usd_file)
        env_cfg.scene = configs.scene_cfg.set_house_asset(
            env_cfg.scene, os.path.join(scene_dir, usd_file)
        )
        tables = configs.scene_cfg.get_table_assets(usd_file)
        if len(tables) != 0:
            table = random.choice(tables)

    # Determine the robot pose
    robot_pose = random.choice([a for a in table["anchors"] if a["side"] == "long"])
    # Set up the third-view camera
    cam_pose = random.choice([a for a in table["anchors"] if a["side"] == "short"])
    env_cfg.scene = configs.scene_cfg.add_scene_camera(
        env_cfg.scene,
        "side_camera",
        configs.scene_cfg.get_camera_cfg(
            sim_cfg["camera"].copy(),
            _get_camera_relative_pose(
                cam_pose, robot_pose, (table["bbox"].min + table["bbox"].max) / 2.0
            ),
        ),
    )

    # Set the light intensity and color
    light_cfg = _get_light_cfg(sim_cfg["lighting"])
    logging.info(
        "Setting light temperature to %d and intensity to %d"
        % (light_cfg["temperature"], light_cfg["intensity"])
    )
    env_cfg.scene = configs.scene_cfg.set_light_asset(env_cfg.scene, **light_cfg)

    # TODO: Dynamically add objects to scene
    # env_cfg.scene = configs.scene_cfg.add_object_to_scene(env_cfg.scene)

    # Dummy robot and end-effector position for debugging
    # robot_pose = {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}

    # Set up the robot arm
    final_ee_position = [
        0.0,
        0.0,
        0.0,
    ]  # TODO: Determine the final end-effector position
    configs.env_cfg.set_robot(robot, env_cfg, robot_pose, final_ee_position)
    # configs.robot_cfg.get_gripper_camera_prim_path(robot)

    return env_cfg


def _get_camera_relative_pose(cam_pose, robot_pose, table_center):
    import configs.scene_cfg

    inv_r = scipy.spatial.transform.Rotation.from_quat(
        [
            robot_pose["quat"][1],
            robot_pose["quat"][2],
            robot_pose["quat"][3],
            robot_pose["quat"][0],
        ]
    ).inv()

    # Relative position of the camera to the robot
    dx, dy, dz = inv_r.apply(cam_pose["pos"] - robot_pose["pos"])

    # Relative rotation of the camera to the robot
    cx, cy, cz = inv_r.apply(np.array(table_center) - robot_pose["pos"])
    cam_quat = configs.scene_cfg.get_quat_from_look_at([dx, dy, dz], [cx, cy, cz])

    return {
        "pos": [dx, dy, dz + 0.75],  # Move the camera above the table top
        "quat": cam_quat,
    }


def _get_light_cfg(light_cfg):
    light_position = [
        random.randint(*light_cfg["position"]["x"]),
        random.randint(*light_cfg["position"]["y"]),
        random.randint(*light_cfg["position"]["z"]),
    ]
    light_temperature = random.randint(*light_cfg["temperature"])
    light_intensity = random.randint(*light_cfg["intensity"])
    return {
        "position": light_position,
        "temperature": light_temperature,
        "intensity": light_intensity,
    }


def main(simulation_app, args):
    with open(args.sim_cfg_file) as fp:
        sim_cfg = yaml.load(fp, Loader=yaml.FullLoader)

    # Create environment
    cfg = get_env_cfg(args.scene_dir, sim_cfg, args.robot)
    env = gym.make("Robot-Env-Cfg-v0", cfg=cfg)
    # Reset environment at start
    env.reset()

    # Perform actions in the environment
    while simulation_app.is_running():
        actions = torch.from_numpy(env.action_space.sample())
        env.step(actions)

    # close the environment
    env.close()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Isaac Simulation Runner")
    # Arguments for the IsaacLab
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save the data from camera at index specified by ``--camera_id``.",
    )
    # IssacSim Environment Initialization
    isaaclab.app.AppLauncher.add_app_launcher_args(parser)
    isaaclab_args, script_args = parser.parse_known_args()
    app_launcher = isaaclab.app.AppLauncher(isaaclab_args)

    # Arguments for the script
    parser.add_argument("--robot", default="franka")
    parser.add_argument(
        "--scene_dir", default=os.path.join(PROJECT_HOME, os.pardir, "USD")
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(PROJECT_HOME, os.pardir, "datasets")
    )
    parser.add_argument(
        "--sim_cfg_file",
        default=os.path.join(PROJECT_HOME, "simulations", "configs", "sim_cfg.yaml"),
    )
    args = parser.parse_args(script_args)

    main(app_launcher.app, args)
    app_launcher.app.close()
