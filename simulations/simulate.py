# -*- coding: utf-8 -*-
#
# @File:   simulate.py
# @Author: Haozhe Xie
# @Date:   2025-03-22 20:59:36
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-04-07 15:36:21
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
import sys

import gymnasium as gym
import isaaclab.app

# import omni.replicator.core
import numpy as np
import scipy.spatial.transform
import torch
import yaml

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.dirname(__file__))


def get_env_cfg(scene_dir, object_dir, sim_cfg, robot):
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
        tables = configs.scene_cfg.get_table_assets(usd_file, sim_cfg["scene"]["table"])
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
            _get_camera_relative_pose(cam_pose, robot_pose, table["bbox"]),
        ),
    )
    # Set up the gripper camera on the robot arm
    env_cfg.scene = configs.scene_cfg.add_scene_camera(
        env_cfg.scene,
        "gripper_camera",
        configs.scene_cfg.get_camera_cfg(
            sim_cfg["camera"].copy(), configs.robot_cfg.get_gripper_camera_cfg(robot)
        ),
    )

    # Set the light intensity and color
    light_cfg = _get_light_cfg(sim_cfg["lighting"])
    logging.info(
        "Setting light temperature to %d and intensity to %d"
        % (light_cfg["temperature"], light_cfg["intensity"])
    )
    env_cfg.scene = configs.scene_cfg.set_light_asset(env_cfg.scene, **light_cfg)

    # Dynamically add objects to scene
    env_cfg.scene = configs.scene_cfg.set_target_object(
        env_cfg.scene,
        _get_object_cfg(
            table["bbox"],
            robot_pose["pos"],
            moving_time=sim_cfg["scene"]["object"]["moving_time"],
        ),
    )
    # TODO: Add more objects to the scene
    # env_cfg.scene = configs.scene_cfg.add_object_to_scene(env_cfg.scene)

    # Set up the robot arm
    configs.env_cfg.set_robot(robot, env_cfg, robot_pose)

    return env_cfg


def _get_camera_relative_pose(cam_pose, robot_pose, table_bbox):
    import configs.scene_cfg

    rbt_quat = robot_pose["quat"]
    inv_r = scipy.spatial.transform.Rotation.from_quat(
        [rbt_quat[1], rbt_quat[2], rbt_quat[3], rbt_quat[0]]
    ).inv()
    # Relative position of the camera to the robot
    dx, dy, dz = inv_r.apply(cam_pose["pos"] - robot_pose["pos"])

    # Relative rotation of the camera to the robot
    tbl_center = (table_bbox.min + table_bbox.max) / 2.0
    cx, cy, cz = inv_r.apply(np.array(tbl_center) - robot_pose["pos"])
    # cz = -robot_pose["pos"][2]
    cam_quat = configs.scene_cfg.get_quat_from_look_at([dx, dy, dz], [cx, cy, cz])

    # Determine the height of the camera (1/5 of the longer side of the table)
    tbl_size = table_bbox.max - table_bbox.min
    dz += max(tbl_size[:2]) / 5

    return {
        "pos": [dx, dy, dz],  # Move the camera above the table top
        "quat": cam_quat,
        "convention": "world",
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


def _get_object_cfg(table_bbox, rbt_pos=None, static=False, moving_time=[1, 5]):
    PADDING = 0.02

    object_cfg = {}
    tbl_z = table_bbox.max[2] + PADDING
    object_cfg["pos"] = np.array(
        [
            random.uniform(table_bbox.min[0] + PADDING, table_bbox.max[0] - PADDING),
            random.uniform(table_bbox.min[1] + PADDING, table_bbox.max[1] - PADDING),
            tbl_z,
        ]
    )
    if not static:
        assert (
            rbt_pos is not None
        ), "Robot position must be provided for dynamic objects."
        # Generate a random position between the table center and the robot arm
        tbl_ctr = (table_bbox.min + table_bbox.max) / 2.0
        rnd_rto = random.uniform(0, 1)
        rnd_pos = tbl_ctr + rnd_rto * (rbt_pos - tbl_ctr)
        rnd_pos[2] = tbl_z
        # Determine the linear velocity of the object
        rnd_tme = random.uniform(*moving_time)
        object_cfg["lin_vel"] = (rnd_pos - object_cfg["pos"]) / rnd_tme

    return object_cfg


def get_state_machine(task, sm_args={}):
    import state_machines.pick_sm

    STATE_MACHINES = {
        "pick": state_machines.pick_sm.PickStateMachine,
    }
    if task not in STATE_MACHINES:
        raise ValueError(f"Unknown task: %s." % task)

    return STATE_MACHINES[task](**sm_args)


def get_curr_state(ee_state, object_state, env_origins):
    return {
        "end_effector": {
            "pos": ee_state.target_pos_w[..., 0, :] - env_origins,
            "quat": ee_state.target_quat_w[..., 0, :],
        },
        "object": {
            "pos": object_state.root_pos_w - env_origins,
            "quat": object_state.root_quat_w,
            "velocity": object_state.root_lin_vel_w,
            "acceleration": object_state.body_lin_acc_w,
        },
    }


def main(simulation_app, args):
    with open(args.sim_cfg_file) as fp:
        sim_cfg = yaml.load(fp, Loader=yaml.FullLoader)

    # Create a new environment
    env_cfg = get_env_cfg(args.scene_dir, args.object_dir, sim_cfg, args.robot)
    env = gym.make("Robot-Env-Cfg-v0", cfg=env_cfg)
    # Reset environment at start
    env.reset()

    # Initialize the state machine
    state_machine = get_state_machine(
        args.task,
        {
            "dt": env_cfg.sim.dt * env_cfg.decimation,
            "num_envs": env.unwrapped.num_envs,
            "device": env.unwrapped.device,
        },
    )

    # Perform actions in the environment
    while simulation_app.is_running():
        robot_origin = (
            torch.from_numpy(env_cfg.scene.robot.init_state.pos[None, :])
            .float()
            .to(env.unwrapped.device)
        )
        robot_quat = (
            torch.from_numpy(env_cfg.scene.robot.init_state.rot[None, :])
            .float()
            .to(env.unwrapped.device)
        )
        curr_state = get_curr_state(
            env.unwrapped.scene["ee_frame"].data,
            env.unwrapped.scene["object"].data,
            env.unwrapped.scene.env_origins + robot_origin,
        )
        action = state_machine.compute(curr_state, robot_quat)

        is_finished = env.step(action)[-2]
        if is_finished.any():
            state_machine.reset_idx(is_finished.nonzero(as_tuple=False).squeeze(-1))

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
        "--scene_dir", default=os.path.join(PROJECT_HOME, os.pardir, "scenes")
    )
    parser.add_argument(
        "--object_dir", default=os.path.join(PROJECT_HOME, os.pardir, "objects")
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(PROJECT_HOME, os.pardir, "datasets")
    )
    parser.add_argument("--task", default="pick")
    parser.add_argument(
        "--sim_cfg_file",
        default=os.path.join(PROJECT_HOME, "simulations", "configs", "sim_cfg.yaml"),
    )
    args = parser.parse_args(script_args)

    main(app_launcher.app, args)
    app_launcher.app.close()
