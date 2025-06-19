# -*- coding: utf-8 -*-
#
# @File:   inference.py
# @Author: Haozhe Xie
# @Date:   2025-05-14 14:25:25
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-06-18 19:41:09
# @Email:  root@haozhexie.com

import argparse
import logging
import os
import pathlib
import time
import uuid

import lerobot.common.policies.diffusion.modeling_diffusion
import lerobot.common.policies.pi0.modeling_pi0
import lerobot.common.policies.pi0fast.modeling_pi0fast
import lerobot.configs.types
import numpy as np
import scipy.spatial.transform
import torch
import zmq
import zmq.utils.monitor


def is_socket_connected(context, sock, timeout=1000):
    MONITOR_ENDPOINT = "inproc://monitor-%s.sock" % str(uuid.uuid4())[-12:]
    sock.monitor(MONITOR_ENDPOINT, zmq.EVENT_CONNECTED)

    monitor_sock = context.socket(zmq.PAIR)
    monitor_sock.connect(MONITOR_ENDPOINT)
    poller = zmq.Poller()
    poller.register(monitor_sock, zmq.POLLIN)

    socks = dict(poller.poll(timeout))
    if monitor_sock in socks:
        evt = monitor_sock.recv_multipart()
        event = zmq.utils.monitor.parse_monitor_message(evt)
        if event["event"] == zmq.EVENT_CONNECTED:
            return True

    return False


def get_zmq_sockets(host, img_port, act_port):
    context = zmq.Context()
    obs_socket = context.socket(zmq.SUB)
    obs_socket.connect("tcp://%s:%d" % (host, img_port))
    if not is_socket_connected(context, obs_socket):
        raise RuntimeError("Failed to connect to tcp://%s:%d" % (host, img_port))

    obs_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    obs_socket.RCVHWM = 1  # Only receive the latest observation

    action_socket = context.socket(zmq.PUSH)
    action_socket.connect("tcp://%s:%d" % (host, act_port))
    if not is_socket_connected(context, action_socket):
        raise RuntimeError("Failed to connect to tcp://%s:%d" % (host, act_port))

    return obs_socket, action_socket


def get_latest_observation(obs_socket):
    image = None
    while True:
        try:
            image = obs_socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            break

    return image


def get_vla_model(model_name, pretrained_model):
    IMG_SIZE = (3, 224, 224)
    ACTION_DIM = 7
    STATE_DIM = 7

    if os.path.exists(pretrained_model):
        pretrained_model = pathlib.Path(pretrained_model)

    if model_name == "diffusion":
        vla_model = lerobot.common.policies.diffusion.modeling_diffusion.DiffusionPolicy.from_pretrained(
            pretrained_model
        )
    elif model_name == "pi0":
        vla_model = lerobot.common.policies.pi0.modeling_pi0.PI0Policy.from_pretrained(
            pretrained_model
        )
    elif model_name == "pi0fast":
        vla_model = lerobot.common.policies.pi0fast.modeling_pi0fast.PI0FASTPolicy.from_pretrained(
            pretrained_model
        )
    else:
        raise ValueError("Unknown model: %s" % model_name)

    # Set up the input and output features
    vla_model.config.output_features = {
        "action": lerobot.configs.types.PolicyFeature(
            type=lerobot.configs.types.FeatureType.ACTION, shape=(ACTION_DIM,)
        )
    }
    vla_model.config.input_features = {
        "observation.image": lerobot.configs.types.PolicyFeature(
            type=lerobot.configs.types.FeatureType.VISUAL, shape=IMG_SIZE
        ),
        "observation.state": lerobot.configs.types.PolicyFeature(
            type=lerobot.configs.types.FeatureType.STATE, shape=(STATE_DIM,)
        ),
    }
    return vla_model


def get_transformed_observation(observation, device="cuda"):
    cam_rgb = observation["observation.image"]["side_cam"]["rgb"].astype(np.float32)
    ee_pose = np.concatenate(
        [
            observation["observation.state"]["end_effector"]["pos"],
            scipy.spatial.transform.Rotation.from_quat(
                observation["observation.state"]["end_effector"]["quat"]
            ).as_euler("xyz", degrees=False),
            observation["observation.state"]["joints"][:, -1:],
        ],
        axis=-1,
    ).astype(np.float32)

    return {
        "observation.image": torch.from_numpy(cam_rgb).permute(0, 3, 1, 2).to(device),
        "observation.state": torch.from_numpy(ee_pose).to(device),
        "task": [observation["task"]],
    }


def get_action(vla_model, observation):
    for ifk in vla_model.config.input_features.keys():
        if ifk not in observation:
            logging.warning("Ingoring observation without key: %s" % ifk)
            return None

    with torch.inference_mode():
        action = (
            vla_model.select_action(get_transformed_observation(observation))
            .cpu()
            .numpy()
        )
        ee_pos = action[:, :3]
        ee_quat = scipy.spatial.transform.Rotation.from_euler(
            "xyz", action[:, 3:6], degrees=False
        ).as_quat()[
            :, [3, 0, 1, 2]
        ]  # Convert to quaternion (w, x, y, z) format
        gripper = action[:, -1:]

    return np.concatenate([ee_pos, ee_quat, gripper], axis=-1).astype(np.float32)


def main(vla_model, vla_weights, host, img_port, act_port):
    # Initialize the VLA model
    logging.info("Loading VLA model: %s with weights: %s" % (vla_model, vla_weights))
    vla_model = get_vla_model(model_name=vla_model, pretrained_model=vla_weights)
    vla_model.reset()
    logging.info(
        "Input features: %s; Output features: %s"
        % (vla_model.config.input_features, vla_model.config.output_features)
    )

    # Initialize the ZeroMQ sockets
    logging.info("Connecting to ZeroMQ sockets ...")
    obs_socket, act_socket = None, None
    while obs_socket is None or act_socket is None:
        try:
            obs_socket, act_socket = get_zmq_sockets(host, img_port, act_port)
            logging.info(
                "Connected to %s:%d for images and %s:%d for actions"
                % (args.host, args.img_port, args.host, args.act_port)
            )
        except RuntimeError as e:
            logging.error("Failed to connect to sockets: %s" % e)
            time.sleep(5)

    logging.info("Starting evaluation the VLA model ...")
    instruction = None
    while True:
        observation = get_latest_observation(obs_socket)
        if observation is None:
            continue

        # Determine the instruction from the observation
        if "task" in observation:
            instruction = observation["task"]
            logging.info("Received new task: %s" % instruction)
            vla_model.reset()  # Reset the model with the new instruction
        elif instruction is not None:
            observation["task"] = instruction
        else:
            continue

        assert "task" in observation, "Observation must contain 'task' key"
        action = get_action(vla_model, observation)
        if action is not None:
            act_socket.send_pyobj(action, flags=zmq.NOBLOCK)
            logging.info("Sending action: %s" % action)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="Evaluation Client Runner")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--img_port", default=3186, type=int, help="Port for image stream"
    )
    parser.add_argument(
        "--act_port", default=3188, type=int, help="Port for action stream"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="The name of VLA model to use"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="The path to the pretrained VLA model",
    )
    args = parser.parse_args()
    main(args.model, args.weights, args.host, args.img_port, args.act_port)
