# -*- coding: utf-8 -*-
#
# @File:   simulate.py
# @Author: Haozhe Xie
# @Date:   2025-03-22 20:59:36
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-03-24 20:34:17
# @Email:  root@haozhexie.com
"""
Script to run an environment with an action state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p simulations/simulate.py --enable_cameras

"""

import argparse
import os
import random
import sys

import gymnasium as gym
import isaaclab.app

# import omni.replicator.core
import numpy as np
import torch

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.dirname(__file__))


def get_env_cfg(scene_dir, robot="franka"):
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
    # Dynamically create basic scene from USD files
    usd_file = random.choice(os.listdir(scene_dir))
    env_cfg.scene = configs.scene_cfg.set_house_asset(
        env_cfg.scene, os.path.join(scene_dir, usd_file)
    )
    # TODO: Dynamically add objects to scene
    # env_cfg.scene = configs.scene_cfg.add_object_to_scene(env_cfg.scene)
    # TODO: Determine the end-effector position
    ee_position = [0, 0, 0]
    # TODO: Set the robot and end-effector frame
    configs.env_cfg.set_robot(robot, env_cfg, ee_position)

    return env_cfg


def main(simulation_app, args):
    # Create environment
    cfg = get_env_cfg(args.scene_dir)
    env = gym.make("Robot-Env-Cfg-v0", cfg=cfg)
    # Reset environment at start
    env.reset()
    # import pdb; pdb.set_trace()
    # close the environment
    env.close()


if __name__ == "__main__":
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
    args = parser.parse_args(script_args)

    main(app_launcher.app, args)
    app_launcher.app.close()
