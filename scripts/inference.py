# -*- coding: utf-8 -*-
#
# @File:   inference.py
# @Author: Haozhe Xie
# @Date:   2025-05-14 14:25:25
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-07-31 15:33:41
# @Email:  root@haozhexie.com

import argparse
import datetime
import logging
import os
import pathlib
import pickle
import sys
import time
import uuid

import numpy as np
import torch
import torchvision.transforms.v2.functional as F
import zmq
import zmq.utils.monitor

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir)
)
import utils.helpers


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
    pretrained_cfg = None
    if os.path.exists(pretrained_model):
        logging.info("Loading VLA model from local path: %s" % pretrained_model)
        pretrained_model = pathlib.Path(pretrained_model).expanduser().resolve()
        pretrained_cfg = pretrained_model / "config.json"

    vla_cfg = utils.helpers.get_policy_cfg(model_name, cfg_file=pretrained_cfg)
    vla_model = utils.helpers.get_policy_class(model_name).from_pretrained(
        pretrained_model, config=vla_cfg
    )
    if torch.cuda.is_available():
        vla_model = vla_model.cuda()

    vla_model.eval()
    return vla_model


def get_transformed_observation(observation, rotation, feat_cfg, device="cuda"):
    images = {
        k: v.astype(np.float32) / 255.0
        for k, v in observation.items()
        if k.startswith("observation.images.") and k in feat_cfg
    }
    ee_pose = observation["observation.state"]["end_effector"]
    ee_pose = np.concatenate(
        [
            ee_pose["pos"],
            utils.helpers.get_rotation_vector(ee_pose["quat"], rotation),
            (ee_pose["gripper"] if "gripper" in ee_pose else np.empty((1, 0))),
        ],
        axis=-1,
    ).astype(np.float32)

    return {
        **{
            k: F.resize(
                torch.from_numpy(v).permute(0, 3, 1, 2).to(device),
                feat_cfg[k].shape[-2:],
            )
            for k, v in images.items()
        },
        "observation.state": torch.from_numpy(ee_pose).to(device),
        "task": [observation["task"]],
    }


def get_action(vla_model, observation, rotation, use_delta_action, debug=False):
    N_DUMMY_STEPS = 5
    _count = getattr(get_action, "count", -N_DUMMY_STEPS)
    _state = getattr(get_action, "state", None)
    for ifk in vla_model.config.input_features.keys():
        if ifk not in observation:
            logging.warning("Ingoring observation without key: %s" % ifk)
            return None

    setattr(get_action, "count", _count + 1)
    # Skip the first few steps to allow model to warm up
    if _count < 0:
        return None

    observation = get_transformed_observation(
        observation, rotation, vla_model.config.input_features
    )
    # Update the state every chunk_size steps
    if _count % vla_model.config.chunk_size == 0:
        _state = observation["observation.state"].cpu().numpy()
        setattr(get_action, "state", _state)
        if debug:
            states = getattr(get_action, "states", [])
            states.append(_state)
            setattr(get_action, "states", states)

    action = vla_model.select_action(observation).cpu().numpy()
    if use_delta_action:
        action_dim = action.shape[-1] - 1
        action[:, :action_dim] += _state[:, :action_dim]

    ee_pos = action[:, :3]
    ee_quat = utils.helpers.get_quaternion(action[:, 3:-1], rotation)
    gripper = action[:, -1:]
    action = np.concatenate([ee_pos, ee_quat, gripper], axis=-1)
    return action


def dump_vla_states(vla_name, states, actions, output_dir):
    output_path = os.path.join(
        output_dir,
        "%s-%s.pkl" % (vla_name, datetime.datetime.now().strftime("%m%d-%H%M%S")),
    )
    with open(output_path, "wb") as fp:
        pickle.dump({"state": states, "action": actions}, fp)


def main(
    vla_model,
    vla_weights,
    rotation,
    use_delta_action,
    host,
    img_port,
    act_port,
    n_total_tests,
    output_dir,
):
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
    n_tests = 0
    actions = []
    instruction = None
    while True:
        if n_tests >= n_total_tests:
            logging.info("Reached the maximum number of tests: %d" % n_total_tests)
            break

        observation = get_latest_observation(obs_socket)
        if observation is None:
            continue

        # Determine the instruction from the observation
        if "task" in observation:
            if len(actions) > 0:
                n_tests += 1

            instruction = observation["task"]
            logging.info("[Test%02d] Received new task: %s" % (n_tests, instruction))
            # Save the debug states/actions
            if output_dir and actions:
                dump_vla_states(
                    os.path.basename(vla_weights),
                    getattr(get_action, "states", []),
                    actions,
                    output_dir,
                )
                setattr(get_action, "states", [])
            # Reset the model with the new instruction
            action = None
            actions = []
            vla_model.reset()
            if hasattr(get_action, "count"):
                delattr(get_action, "count")
        elif instruction is not None:
            observation["task"] = instruction
        else:
            continue

        assert "task" in observation, "Observation must contain 'task' key"
        action = get_action(
            vla_model, observation, rotation, use_delta_action, output_dir is not None
        )
        if action is not None:
            act_socket.send_pyobj(action, flags=zmq.NOBLOCK)
            logging.debug("Sending action: %s" % (np.round(action, 2),))
            actions.append(action)


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
        "-m", "--model", type=str, required=True, help="The name of VLA model to use"
    )
    parser.add_argument(
        "-r",
        "--rotation",
        type=str,
        default="quat",
    )
    parser.add_argument(
        "-d",
        "--delta",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--weights",
        type=str,
        required=True,
        help="The path to the pretrained VLA model",
    )
    parser.add_argument(
        "-n",
        "--n_tests",
        type=int,
        default=10,
        help="The number of tests to run",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="The directory to save the VLA's output",
    )
    args = parser.parse_args()
    main(
        args.model,
        args.weights,
        args.rotation,
        args.delta,
        args.host,
        args.img_port,
        args.act_port,
        args.n_tests,
        args.output_dir,
    )
