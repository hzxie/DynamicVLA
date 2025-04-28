# -*- coding: utf-8 -*-
#
# @File:   simulate.py
# @Author: Haozhe Xie
# @Date:   2025-03-22 20:59:36
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-04-28 19:24:26
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

import cv2
import gymnasium as gym
import isaaclab.app
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
        # usd_file = "D:/Projects/DynamicVLA/scenes/058205e1-6ec4-4342-a609-1ecce3551c3b.usd"
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

    robot_quat = robot_pose["quat"]
    inv_r = scipy.spatial.transform.Rotation.from_quat(
        [robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]]
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


def _get_object_cfg(table_bbox, rbt_pos=None, static=False, moving_time=[1, 2]):
    import configs.object_cfg

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
        rnd_rto = random.uniform(-0.5, 0.5)
        rnd_pos = tbl_ctr + rnd_rto * (rbt_pos - tbl_ctr)
        rnd_pos[2] = tbl_z
        # Determine the linear velocity of the object
        rnd_tme = random.uniform(*moving_time)
        object_cfg["lin_vel"] = (rnd_pos - object_cfg["pos"]) / rnd_tme
        object_cfg["quat"] = configs.object_cfg.get_object_init_quat(
            object_cfg["lin_vel"]
        )

    return configs.object_cfg.get_object_cfg(
        object_cfg,
        configs.object_cfg.get_spawner_cfg(
            # "D:/Projects/DynamicVLA/objects/apple/apple_10.usd"
        ),
    )


def get_state_machine(task, sm_args={}):
    import state_machines.pick_sm

    STATE_MACHINES = {
        "pick": state_machines.pick_sm.PickStateMachine,
    }
    if task not in STATE_MACHINES:
        raise ValueError(f"Unknown task: %s." % task)

    return STATE_MACHINES[task](**sm_args)


def get_curr_state(ee_state, object_state, env_origins, robot_quat):
    quat_opengl = robot_quat[:, [1, 2, 3, 0]]  # xyzw
    return {
        "end_effector": {
            "pos": _get_robot_relative_position(
                ee_state.target_pos_w[..., 0, :] - env_origins, quat_opengl
            ),
            "quat": ee_state.target_quat_w[..., 0, :],
        },
        "object": {
            "pos": _get_robot_relative_position(
                object_state.root_pos_w - env_origins, quat_opengl
            ),
            "quat": object_state.root_quat_w,  # TODO
            "velocity": _get_robot_relative_position(
                object_state.root_lin_vel_w, quat_opengl
            ),
        },
    }


def _get_robot_relative_position(point, robot_quat):
    # inv_quat = scipy.spatial.transform.Rotation.from_quat(robot_quat).inv()
    # inv_offset = inv_quat.apply(point)

    # pytorch3d/transforms/rotation_conversions.html#quaternion_invert
    inv_quat = robot_quat * torch.tensor([[-1, -1, -1, 1]], device=robot_quat.device)
    # pytorch3d/transforms/rotation_conversions.html#quaternion_apply
    t = 2.0 * torch.cross(inv_quat[..., :3], point, dim=-1)
    inv_offset = (
        point + inv_quat[..., 3:] * t + torch.cross(inv_quat[..., :3], t, dim=-1)
    )
    return inv_offset


def get_camera_frames(sensors):
    # NOTE: import isaaclab.utils does not work
    from isaaclab.utils import convert_dict_to_backend

    values = {}
    for name, sensor in sensors.items():
        if type(sensor).__name__ == "Camera":
            values[name] = convert_dict_to_backend(
                {k: v for k, v in sensor.data.output.items()}, backend="numpy"
            )

    return values


def get_stitched_frames(cameras, env_id=0):
    MAX_DEPTH = 25

    stitched_frames = []
    for name, camera in cameras.items():
        frames = []
        for k, v in camera.items():
            if v.ndim == 2:
                frame = np.repeat(v[env_id, :, :, None], 3, axis=-1)
            elif v.shape[-1] == 1:
                frame = np.repeat(v[env_id, :, :, :], 3, axis=-1)
            elif v.shape[-1] >= 3:
                frame = np.repeat(v[env_id, :, :, :3], 1, axis=-1)
            else:
                raise ValueError(f"Unknown camera data shape: {v.shape}")

            # Normalize the depth image to 0-255
            if k == "distance_to_image_plane":
                frame = np.clip(frame, 0, MAX_DEPTH)
                frame = (frame / np.max(frame) * 255).astype(np.uint8)

            frames.append(frame)

        stitched_frames.append(np.concatenate(frames, axis=1))

    return np.concatenate(stitched_frames, axis=0)


def print_state_on_frame(frame, curr_state, next_state, robot_quat, env_id=0):
    TEXT_MARGIN = 10
    TEXT_SCALE = 0.5
    TEXT_THICKNESS = 1
    TEXT_COLOR = (255, 255, 255)
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

    lines = _get_state_text(curr_state, next_state, robot_quat, env_id).split("\n")
    img_height, img_width = frame.shape[:2]
    # Print the text on the image
    y = TEXT_MARGIN
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(
            line, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS
        )
        x = img_width - text_width - TEXT_MARGIN
        frame = cv2.putText(
            np.ascontiguousarray(frame),
            line,
            (x, y + text_height),
            TEXT_FONT,
            TEXT_SCALE,
            TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )
        y += text_height + TEXT_MARGIN

    return frame


def _get_state_text(curr_state, next_state, robot_quat, env_id=0):
    grasp_rot = scipy.spatial.transform.Rotation.from_quat(
        next_state["grasp_quat"][env_id].cpu().numpy()
    ).as_euler("xyz", degrees=True)
    robot_rot = scipy.spatial.transform.Rotation.from_quat(
        robot_quat[env_id, [1, 2, 3, 0]].cpu().numpy()
    ).as_euler("xyz", degrees=True)

    text = "SM. State: %d\n" % next_state["sm_state"][env_id].item()
    text += "Rbt. Rot: %.3f %.3f %.3f\n" % (
        robot_rot[0],
        robot_rot[1],
        robot_rot[2],
    )
    text += "EE. Pos: %.3f %.3f %.3f\n" % (
        curr_state["end_effector"]["pos"][env_id, 0].item(),
        curr_state["end_effector"]["pos"][env_id, 1].item(),
        curr_state["end_effector"]["pos"][env_id, 2].item(),
    )
    text += "Obj. Pos: %.3f %.3f %.3f\n" % (
        curr_state["object"]["pos"][env_id, 0].item(),
        curr_state["object"]["pos"][env_id, 1].item(),
        curr_state["object"]["pos"][env_id, 2].item(),
    )
    text += "Obj. Vel: %.3f %.3f %.3f\n" % (
        curr_state["object"]["velocity"][env_id, 0].item(),
        curr_state["object"]["velocity"][env_id, 1].item(),
        curr_state["object"]["velocity"][env_id, 2].item(),
    )
    text += "Gsp. Pos: %.3f %.3f %.3f\n" % (
        next_state["grasp_postion"][env_id, 0].item(),
        next_state["grasp_postion"][env_id, 1].item(),
        next_state["grasp_postion"][env_id, 2].item(),
    )
    text += "Gsp. Rot: %.3f %.3f %.3f\n" % (
        grasp_rot[0],
        grasp_rot[1],
        grasp_rot[2],
    )
    return text


def is_final_position_reached(object_pos, ee_pos, final_pos):
    DIST_THRESHOLD = 0.02

    ee_offset = (ee_pos - final_pos).abs().sum(dim=1)
    obj_offset = (object_pos - final_pos).abs().sum(dim=1)
    return torch.bitwise_and(ee_offset < DIST_THRESHOLD, obj_offset < DIST_THRESHOLD)


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
    frame_count = 0
    while simulation_app.is_running():
        # Add an option to disable the state machine to accelerate the simulation
        if args.disable_sm:
            env.step(torch.from_numpy(env.action_space.sample()))
            continue

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
            robot_quat,
        )
        next_state = state_machine.compute(curr_state)  # xyz, quat (wxyz), gripper
        frames = get_camera_frames(env.unwrapped.scene.sensors)
        if args.debug:
            frame = get_stitched_frames(frames)[..., ::-1]
            frame = print_state_on_frame(frame, curr_state, next_state, robot_quat)
            cv2.imwrite(
                os.path.join(args.output_dir, "%06d.jpg" % frame_count),
                frame,
            )

        frame_count += 1
        # Check whether the simulation is finished
        _ = env.step(next_state["action"])
        # Ideally, _[-2] indicates the simulation is finished, which does not work.
        is_finished = is_final_position_reached(
            curr_state["object"]["pos"],
            curr_state["end_effector"]["pos"],
            state_machine.final_object_pose[:, :3],
        )
        if is_finished.any():
            import pdb

            pdb.set_trace()
            state_machine.reset_idx(is_finished.nonzero(as_tuple=False).squeeze(-1))

    # close the environment
    env.close()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    SHARED_PARAMETERS = ["num_envs", "save"]

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
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--disable_sm", action="store_true", default=False)
    args = parser.parse_args(script_args)
    # Copy the shared parameters from isaaclab_args to args
    for sp in SHARED_PARAMETERS:
        if sp in isaaclab_args:
            setattr(args, sp, getattr(isaaclab_args, sp))

    main(app_launcher.app, args)
    app_launcher.app.close()
