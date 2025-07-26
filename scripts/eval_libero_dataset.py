# -*- coding: utf-8 -*-
#
# @File:   eval_libero_dataset.py
# @Author: Haozhe Xie
# @Date:   2025-07-11 14:31:21
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-07-24 12:38:51
# @Email:  root@haozhexie.com
# @Ref: https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py
#
# NOTE: UPDATES NEEDED in LIBERO/libero/libero/benchmark/__init__.py L164:
# -> torch.load(init_states_path, weights_only=False)

import argparse
import datetime
import logging
import os
import pickle
import sys

import cv2
import numpy as np
from libero.libero import benchmark as libero_benchmark
from libero.libero import envs as libero_envs
from libero.libero import get_libero_path

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)
import inference

import utils.helpers


def get_libero_env(env_name, task_id, seed):
    LIBERO_PATH = get_libero_path("bddl_files")
    LIBERO_IMG_SIZE = 256
    assert env_name in ["libero_goal", "libero_object", "libero_spatial", "libero_10"]

    benchmarks = libero_benchmark.get_benchmark_dict()
    task_suite = benchmarks[env_name]()
    assert task_id < task_suite.n_tasks
    task = task_suite.get_task(task_id)

    env = libero_envs.OffScreenRenderEnv(
        **{
            "bddl_file_name": os.path.join(
                LIBERO_PATH, task.problem_folder, task.bddl_file
            ),
            "camera_heights": LIBERO_IMG_SIZE,
            "camera_widths": LIBERO_IMG_SIZE,
        }
    )
    env.seed(seed)
    env.reset()
    logging.info("Environment initialized with task: %s." % (task.name))
    env.set_init_state(task_suite.get_task_init_states(task_id)[0])

    return env, task.language


def get_observation(cfg, obs, instruction):
    IMAGE_MAPPER = {
        "agentview_image": {
            "name": "observation.images.base",
            "size": cfg["observation.images.base"].shape[-2:],
        },
        "robot0_eye_in_hand_image": {
            "name": "observation.images.wrist",
            "size": cfg["observation.images.wrist"].shape[-2:],
        },
    }

    images = {
        v["name"]: np.ascontiguousarray(
            cv2.resize(obs[k][::-1, ::-1], v["size"])[None, ...]
        )
        for k, v in IMAGE_MAPPER.items()
    }
    # NOTE: Convert xyzw -> wxyz for quat
    # Because it will be feed to inference..get_transformed_observation,
    # which expects the quaternion in wxyz format.
    ee_pose = {
        "end_effector": {
            "pos": obs["robot0_eef_pos"][None, :].astype(np.float32),
            "quat": obs["robot0_eef_quat"][None, [3, 0, 1, 2]].astype(np.float32),
            "gripper": obs["robot0_gripper_qpos"][None, :].astype(np.float32),
        }
    }
    return {
        **images,
        "observation.state": ee_pose,
        "task": instruction,
    }


def get_action(vla_model, observation, use_delta_action):
    _count = getattr(get_action, "count", 0)
    _state = getattr(get_action, "state", None)
    for ifk in vla_model.config.input_features.keys():
        if ifk not in observation:
            logging.warning("Ingoring observation without key: %s" % ifk)
            return None

    observation = inference.get_transformed_observation(
        observation, "rotvec", vla_model.config.input_features
    )
    # Update the state every chunk_size steps
    if _count % vla_model.config.chunk_size == 0:
        _state = observation["observation.state"].cpu().numpy()
        setattr(get_action, "state", _state)

    action = vla_model.select_action(observation).cpu().numpy()
    if use_delta_action:
        action_dim = action.shape[-1] - 1
        action[:, :action_dim] += _state[:, :action_dim]

    setattr(get_action, "count", _count + 1)
    return action


def get_episode_name(env_name, task_id, done):
    return "%s-%02d-%s-%s.mp4" % (
        env_name,
        task_id,
        datetime.datetime.now().strftime("%m%d-%H%M%S"),
        "SUCCESS" if done else "FAIL",
    )


def main(vla_model, vla_weights, env_name, task_id, output_dir, seed, debug):
    # Initialize the VLA model
    logging.info("Loading VLA model: %s with weights: %s" % (vla_model, vla_weights))
    vla_model = inference.get_vla_model(
        model_name=vla_model, pretrained_model=vla_weights
    )
    vla_model.reset()
    logging.info(
        "Input features: %s; Output features: %s"
        % (vla_model.config.input_features, vla_model.config.output_features)
    )

    # Initialize the LIBERO environment
    logging.info("Initializing LIBERO environment [Name=%s, Task ID=%d] ...")
    env, task_instruction = get_libero_env(env_name, task_id, seed)

    # Run the evaluation loop
    DUMMY_ACTION = [0.0] * 6 + [-1.0]
    N_MAX_STEPS = 600
    N_WAIT_STEPS = 10

    logging.info("Starting evaluation the VLA model ...")
    done = False
    frames = []
    actions = []
    for step_idx in range(N_MAX_STEPS):
        if step_idx < N_WAIT_STEPS:
            obs, _, done, _ = env.step(DUMMY_ACTION)
            continue

        action = get_action(
            vla_model,
            get_observation(vla_model.config.input_features, obs, task_instruction),
            use_delta_action=True,
        )
        actions.append(action)
        obs, _, done, _ = env.step(action.squeeze(0))
        frames.append(obs["agentview_image"][::-1, ::-1])
        if done:
            logging.info("Task completed in %d steps." % (step_idx + 1))
            break

    env.close()
    utils.helpers.dump_video(
        frames, os.path.join(output_dir, get_episode_name(env_name, task_id, done))
    )
    if debug:
        output_path = os.path.join(
            output_dir,
            "%s-%02d-%s.pkl"
            % (env_name, task_id, datetime.datetime.now().strftime("%m%d-%H%M%S")),
        )
        with open(output_path, "wb") as fp:
            pickle.dump(
                {
                    "env": env_name,
                    "task": task_id,
                    "inst": task_instruction,
                    "action": actions,
                },
                fp,
            )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="LIBERO Evaluation Runner")
    parser.add_argument(
        "--model", type=str, required=True, help="The name of VLA model to use"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="The path to the pretrained VLA model",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="libero_10",
        help="The name of the LIBERO environment. Options: "
        "[libero_goal, libero_object, libero_spatial, libero_10]",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=0,
        help="The task ID in the LIBERO environment",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_HOME, "runs", "evaluation"),
        help="Directory to save the evaluation results",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for detailed logging"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    main(
        args.model,
        args.weights,
        args.env,
        args.task,
        args.output_dir,
        args.seed,
        args.debug,
    )
