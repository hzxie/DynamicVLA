# -*- coding: utf-8 -*-
#
# @File:   translate_dataset_seq.py
# @Author: Haozhe Xie
# @Date:   2025-07-28 18:09:15
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-08-09 16:30:41
# @Email:  root@haozhexie.com

import argparse
import json
import logging
import os
import random
import shutil
import sys

import h5py
import isaaclab.app
import numpy as np
import torch
import yaml

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)
sys.path.append(os.path.dirname(__file__))

import simulations.evaluate as eval
import simulations.simulate as sim


def get_camera_config(sim_cfg_file, robot_usd_path):
    import configs.robot_cfg

    if sim_cfg_file is None or not os.path.exists(sim_cfg_file):
        return None

    cam_cfg = {}
    robot_name = configs.robot_cfg.get_robot_name(robot_usd_path)
    with open(sim_cfg_file) as fp:
        sim_cfg = yaml.load(fp, Loader=yaml.FullLoader)

    sim_cfg["camera"]["class_type"] = "isaaclab.sensors.camera.camera:Camera"
    if "cameras" in sim_cfg["scene"] and "camera" in sim_cfg:
        # Wrist camera configuration
        cam_cfg["wrist_cam"] = sim_cfg["camera"].copy()
        cam_cfg["wrist_cam"].update(configs.robot_cfg.get_wrist_camera_cfg(robot_name))
        # Other cameras configuration
        for cam in sim_cfg["scene"]["cameras"]:
            cam_name = cam["name"]
            cam_cfg[cam_name] = sim_cfg["camera"].copy()
            cam_cfg[cam_name].update(sim.get_camera_pose(cam))

    return {"scene": cam_cfg}


def simulate(env, sim_states, robot_origin, robot_quat, final_position, debug=False):
    OFFSET = 5
    state_seq = np.concatenate(
        [sim_states["ee_pos"][()], sim_states["ee_quat"][()]], axis=1
    )
    action_seq = sim_states["action"][()]
    # Add extra frames to the action sequence (to reach final position)
    action_seq = np.concatenate(
        [
            action_seq,
            action_seq[-1:].repeat(OFFSET, axis=0),
        ],
        axis=0,
    )
    state_seq = np.concatenate(
        [
            state_seq,
            action_seq[-1:, :-1].repeat(OFFSET, axis=0),
        ],
        axis=0,
    )

    # The simulation loop
    env_states = []
    for act_conter in range(action_seq.shape[0] - OFFSET):
        curr_state = sim.get_curr_state(
            env.unwrapped.scene["ee_frame"].data,
            env.unwrapped.scene.state["articulation"]["robot"]["joint_position"],
            env.unwrapped.scene["object"].data,
            None,
            None,
            env.unwrapped.scene.env_origins + robot_origin,
            robot_quat,
            env.unwrapped.device,
        )
        cam_view = sim.get_camera_views(
            env.unwrapped.scene.sensors, ["rgb", "depth", "seg"]
        )

        action = action_seq[act_conter]
        # Replace the action with the next state
        action[:7] = state_seq[act_conter + OFFSET, :7]
        action = _get_action_tensor(
            action[None, :],
            env.unwrapped.num_envs,
            env.unwrapped.device,
        )
        env.step(action)
        # Check if the final position is reached
        is_done = sim.is_final_position_reached(
            curr_state["object"]["pos"],
            curr_state["end_effector"]["pos"],
            final_position,
        )
        env_states.append(
            {
                "cam_views": cam_view,
                "curr_state": curr_state,
                "next_state": {"action": action},
                "is_done": is_done,
            }
        )

    env_states = sim.get_env_states(env_states, env.unwrapped.num_envs)
    return [
        (es, is_done[env_id].item())
        for env_id, es in enumerate(env_states)
        if is_done[env_id].item() or debug
    ]


def _get_action_tensor(action, num_envs, device):
    if isinstance(action, np.ndarray):
        action = torch.from_numpy(action).to(device)
    elif isinstance(action, torch.tensor):
        action = action.to(device)
    else:
        logging.warning("Unsupported action type: %s" % type(action))
        action = None

    if action.size(0) != num_envs or action.size(1) != 8:
        logging.warning(
            "Received action with shape %s, expected (%d, 8)" % (action.shape, num_envs)
        )
        action = None

    return action


def main(args):
    sequences = sorted([f for f in os.listdir(args.dataset_dir) if f.endswith(".h5")])
    if args.range is not None:
        start, end = args.range
        logging.info("Processing sequences from %d to %d" % (start, end))
        sequences = sequences[start:end]

    for seq in sequences:
        # Load the dataset sequence
        with h5py.File(os.path.join(args.dataset_dir, seq), "r") as f:
            env_states = {k: f[k][()] for k in f.keys()}

        # Set up test environment
        env_cfg = os.path.join(args.dataset_dir, "%s.json" % seq[:-3])
        logging.info("Recovering test environment from %s" % env_cfg)
        with open(env_cfg, "r") as fp:
            env_cfg = json.load(fp)

        # Remove old camera configurations
        env_cfg = {
            k: v
            for k, v in env_cfg.items()
            if not isinstance(v, dict)
            or "class_type" not in v
            or not isinstance(v["class_type"], str)
            or not v["class_type"].startswith("isaaclab.sensors.camera")
        }
        # Recover camera configuration from the simulation config file
        if args.enable_cameras and args.sim_cfg_file is not None:
            env_cfg["scene"].update(
                get_camera_config(
                    args.sim_cfg_file, env_cfg["scene"]["robot"]["spawn"]["usd_path"]
                )
            )

        # Fix random seed for reproducibility
        random.seed(env_cfg["seed"])
        np.random.seed(env_cfg["seed"])
        torch.manual_seed(env_cfg["seed"])
        # Set up the instruction and environment
        env = eval.get_test_env(
            env_cfg,
            args.num_envs,
            args.scene_dir,
            args.object_dir,
            args.physics_time_step,
            args.timeout,
            args.device,
            args.disable_fabric,
            args.path_tracing,
        )
        env.reset(seed=env_cfg["seed"])

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
        final_position = eval.get_final_position(env_cfg, env.unwrapped.device)

        env_states = simulate(
            env, env_states, robot_origin, robot_quat, final_position, args.debug
        )
        for es in env_states:
            env_state, success = es
            if args.save:
                assert success  # Only save successful episodes
                shutil.copyfile(
                    os.path.join(args.dataset_dir, "%s.json" % seq[:-3]),
                    os.path.join(args.output_dir, "%s-tr.json" % seq[:-3]),
                )
                with h5py.File(
                    os.path.join(args.output_dir, "%s-tr.h5" % seq[:-3]), "w"
                ) as fp:
                    for k, v in env_state.items():
                        fp.create_dataset(k, data=v, compression="gzip")

            if args.debug:
                sim.dump_video(
                    sim.get_frames(env_state, ["ee_pos", "object_pos", "object_vel"]),
                    os.path.join(
                        args.output_dir,
                        "%s-%s.mp4" % (seq[:-4], "SUCCESS" if success else "FAIL"),
                    ),
                )

        env.close()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    SHARED_PARAMETERS = ["num_envs", "save"]

    parser = argparse.ArgumentParser(description="Dataset Sequence Replay Script")
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
    parser.add_argument("--physics_time_step", type=float, default=0.04)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument(
        "--sim_cfg_file",
        default=os.path.join(PROJECT_HOME, "simulations", "configs", "sim_cfg.yaml"),
    )
    parser.add_argument(
        "--scene_dir", default=os.path.join(PROJECT_HOME, os.pardir, "scenes")
    )
    parser.add_argument(
        "--object_dir", default=os.path.join(PROJECT_HOME, os.pardir, "objects")
    )
    parser.add_argument(
        "--dataset_dir", default=os.path.join(PROJECT_HOME, os.pardir, "datasets")
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(PROJECT_HOME, os.pardir, "datasets")
    )
    parser.add_argument("--range", type=int, nargs=2, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(script_args)
    # Copy the shared parameters from isaaclab_args to args
    for sp in SHARED_PARAMETERS:
        if sp in isaaclab_args:
            setattr(args, sp, getattr(isaaclab_args, sp))

    app_launcher = isaaclab.app.AppLauncher(isaaclab_args)
    # Pass "enable_cameras" to this script
    # Ref: https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/app/app_launcher.html
    args.enable_cameras = app_launcher._enable_cameras
    if not args.enable_cameras:
        logging.warning(
            "Cameras are disabled. No images will be produced during simulation."
        )
        answer = input("Do you want to continue? (y/N) ").strip().lower()
        if answer != "y":
            exit(0)

    main(args)
    app_launcher.app.close()
