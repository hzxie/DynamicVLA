# -*- coding: utf-8 -*-
#
# @File:   translate_dataset_seq.py
# @Author: Haozhe Xie
# @Date:   2025-07-28 18:09:15
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-12-06 08:35:01
# @Email:  root@haozhexie.com

import argparse
import json
import logging
import os
import random
import sys

import h5py
import numpy as np
import torch
import yaml
from isaaclab.app import AppLauncher

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)
sys.path.append(os.path.join(PROJECT_HOME, "simulations"))

import simulations.evaluate as eval
import simulations.helpers as helpers
import simulations.simulate as sim


def get_camera_config(sim_cfg_file, robot_usd_path):
    import configs.robot_cfg
    import configs.scene_cfg

    if sim_cfg_file is None or not os.path.exists(sim_cfg_file):
        return None

    cam_cfg = {}
    robot_name = configs.robot_cfg.get_robot_name(robot_usd_path)
    with open(sim_cfg_file) as fp:
        sim_cfg = yaml.load(fp, Loader=yaml.FullLoader)

    sim_cfg["camera"]["class_type"] = "isaaclab.sensors.camera.camera:Camera"
    if "cameras" in sim_cfg["scene"] and "camera" in sim_cfg:
        # Wrist camera configuration
        cam_cfg["wrist_cam"] = configs.scene_cfg.get_camera_cfg(
            sim_cfg["camera"].copy(), configs.robot_cfg.get_wrist_camera_cfg(robot_name)
        ).to_dict()
        # Other cameras configuration
        for cam in sim_cfg["scene"]["cameras"]:
            cam_name = cam["name"]
            cam_cfg[cam_name] = configs.scene_cfg.get_camera_cfg(
                sim_cfg["camera"].copy(), sim.get_camera_pose(cam)
            ).to_dict()

    return cam_cfg


def simulate(env, sim_states, debug=False):
    import configs.termination_cfg

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
    term_mgr = env.env.termination_manager
    done_term = configs.termination_cfg.get_done_term(term_mgr.active_terms)
    for act_conter in range(action_seq.shape[0] - OFFSET):
        curr_state = sim.get_curr_state(
            ee_state=env.unwrapped.scene["ee_frame"].data,
            object_state=env.unwrapped.scene["object"].data,
            env_origins=env.unwrapped.scene["robot"].data.root_pos_w,
            robot_quat=env.unwrapped.scene["robot"].data.root_quat_w,
            device=env.unwrapped.device,
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
        is_done = term_mgr.get_term(done_term)
        env_states.append(
            {
                "cam_views": cam_view,
                "curr_state": curr_state,
                "next_state": {"action": action},
                "is_done": is_done,
            }
        )
        if is_done.all():
            break

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


def is_cam_occluded(env_states):
    semantic_tags = helpers.get_semantic_tags()
    for key in env_states.keys():
        if not key.endswith("_seg"):
            continue

        es = np.stack(env_states[key])
        if semantic_tags["OBJECT_MAIN"] not in np.unique(es):
            return True

    return False


def is_object_occluded(scene_cfg, env_states, object_type, n_steps=25):
    assert object_type in ["object", "container"]

    semantic_tags = helpers.get_semantic_tags()
    semantic_tags_rev = {v: k for k, v in semantic_tags.items()}
    n_exp_objects = len([k for k in scene_cfg.keys() if k.startswith(object_type)])
    for key in env_states.keys():
        if not key.endswith("_seg") or key.startswith("wrist_cam"):
            continue

        es = np.stack(env_states[key])
        n_frames = es.shape[0]
        for i in range(3, min(n_steps, n_frames)):  # Skip  the first n frames (init)
            semantic_labels = np.unique(es[i])
            n_act_objects = len(
                [
                    sl
                    for sl in semantic_labels
                    if semantic_tags_rev.get(sl, "").startswith(object_type.upper())
                ]
            )
            if n_act_objects < n_exp_objects:
                return True

    return False


def remove_spatial_tags(tags):
    # _get_state_tag() in simulations/helpers.py
    KEYWORDS = [
        "tall",
        "short",
        "height",
        "aera",
        "volume",
        "at the start",
        "initial velocity",
    ]
    return [t for t in tags if not any(kw in t for kw in KEYWORDS)]


def main(args):
    sequences = sorted([f for f in os.listdir(args.dataset_dir) if f.endswith(".h5")])
    if args.range is not None:
        start, end = args.range
        logging.info("Processing sequences from %d to %d" % (start, end))
        sequences = sequences[start:end]

    for seq in sequences:
        output_file = os.path.join(args.output_dir, "%s-tr.json" % seq[:-3])
        if os.path.exists(output_file):
            continue

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
            cam_cfg = get_camera_config(
                args.sim_cfg_file, env_cfg["scene"]["robot"]["spawn"]["usd_path"]
            )
            for k, v in cam_cfg.items():
                env_cfg["scene"][k] = v

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
            args.tolerance,
            args.device,
            args.disable_fabric,
            args.path_tracing,
        )
        env.reset(seed=env_cfg["seed"])

        env_states = simulate(env, env_states, args.debug)
        for env_state, success in env_states:
            if (
                args.save
                and not is_cam_occluded(env_state)
                and not is_object_occluded(env_cfg["scene"], env_state, "object")
                and not is_object_occluded(env_cfg["scene"], env_state, "container")
                and len(env_cfg["instruction"]["objects"]) != 0
            ):
                assert success or args.debug  # Only save successful episodes
                with open(
                    os.path.join(args.output_dir, "%s-tr.json" % seq[:-3]), "w"
                ) as fp:
                    json.dump(sim.get_object_without_numpy(env_cfg), fp, indent=2)

                with h5py.File(
                    os.path.join(args.output_dir, "%s-tr.h5" % seq[:-3]), "w"
                ) as fp:
                    for k, v in env_state.items():
                        fp.create_dataset(k, data=v, compression="gzip")

            if args.debug and args.enable_cameras:
                logging.debug(
                    "Cam Occluded: %s; Object Occluded: %s; Container Occluded: %s"
                    % (
                        is_cam_occluded(env_state),
                        is_object_occluded(env_cfg["scene"], env_state, "object"),
                        is_object_occluded(env_cfg["scene"], env_state, "container"),
                    )
                )
                sim.dump_video(
                    sim.get_frames(env_state, ["ee_pos", "object_pos", "object_vel"]),
                    os.path.join(
                        args.output_dir,
                        "%s-%s.mp4" % (seq[:-3], "SUCCESS" if success else "FAIL"),
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
    AppLauncher.add_app_launcher_args(parser)
    isaaclab_args, script_args = parser.parse_known_args()

    # Arguments for the script
    parser.add_argument("--path_tracing", action="store_true")
    parser.add_argument("--physics_time_step", type=float, default=0.04)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--tolerance", type=float, default=0.03)
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

    app_launcher = AppLauncher(isaaclab_args)
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
