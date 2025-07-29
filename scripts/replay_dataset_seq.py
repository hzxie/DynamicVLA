# -*- coding: utf-8 -*-
#
# @File:   replay_dataset_seq.py
# @Author: Haozhe Xie
# @Date:   2025-07-28 07:17:57
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-07-29 21:37:11
# @Email:  root@haozhexie.com

import argparse
import json
import logging
import os
import random
import sys

import h5py
import isaaclab.app
import numpy as np
import torch

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)
sys.path.append(os.path.dirname(__file__))

import simulations.simulate as sim
import simulations.evaluate as eval


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


def simulate(env, env_states):
    state_seq = np.concatenate(
        [env_states["ee_pos"][()], env_states["ee_quat"][()]], axis=1
    )
    action_seq = env_states["action"][()]
    cam_views = []

    OFFSET = 0
    # The simulation loop
    for act_conter in range(action_seq.shape[0] - OFFSET):
        # scene_state = env.unwrapped.scene.state
        cam_view = sim.get_camera_views(env.unwrapped.scene.sensors, ["rgb"])
        cam_views.append(cam_view)

        action = action_seq[act_conter]
        # Replace the action with the next state
        # action[:7] = state_seq[act_conter + OFFSET, :7]
        action = _get_action_tensor(
            action[None, :],
            env.unwrapped.num_envs,
            env.unwrapped.device,
        )
        env.step(action)

    return cam_views


def main(args):
    # Load the dataset sequence
    with h5py.File(os.path.join(args.dataset_dir, "%s.h5" % args.seq), "r") as f:
        env_states = {k: f[k][()] for k in f.keys()}

    # Set up test environment
    env_cfg = os.path.join(args.dataset_dir, "%s.json" % args.seq)
    logging.info("Recovering test environment from %s" % env_cfg)
    with open(env_cfg, "r") as fp:
        env_cfg = json.load(fp)

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
    cam_views = simulate(env, env_states)
    # Save the frames if needed
    if args.save and len(cam_views) > 1:
        episode_file_path = os.path.join(args.output_dir, "%s.mp4" % args.seq)
        logging.info(
            "Saving videos (%d frames) to %s" % (len(cam_views), episode_file_path)
        )
        sim.dump_video(
            sim.get_frames(eval.get_frames(cam_views), state_keys=[]),
            episode_file_path,
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
        "--scene_dir", default=os.path.join(PROJECT_HOME, os.pardir, "scenes")
    )
    parser.add_argument(
        "--object_dir", default=os.path.join(PROJECT_HOME, os.pardir, "objects")
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_HOME, "runs", "evaluation"),
    )
    parser.add_argument(
        "--dataset_dir", default=os.path.join(PROJECT_HOME, os.pardir, "datasets")
    )
    parser.add_argument("--seq", required=True)
    args = parser.parse_args(script_args)
    # Copy the shared parameters from isaaclab_args to args
    for sp in SHARED_PARAMETERS:
        if sp in isaaclab_args:
            setattr(args, sp, getattr(isaaclab_args, sp))

    app_launcher = isaaclab.app.AppLauncher(isaaclab_args)
    main(args)
    app_launcher.app.close()
