# -*- coding: utf-8 -*-
#
# @File:   inference.py
# @Author: Haozhe Xie
# @Date:   2025-05-14 14:25:25
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-16 20:45:35
# @Email:  root@haozhexie.com

import argparse
import json
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


def get_vla_model(pretrained_model, use_delta_action, streaming, skip_n_actions):
    pretrained_cfg = None
    if not os.path.exists(pretrained_model):
        raise FileNotFoundError(
            "Pretrained model path does not exist: %s" % pretrained_model
        )

    logging.info("Loading VLA model from local path: %s" % pretrained_model)
    pretrained_model = pathlib.Path(pretrained_model).expanduser().resolve()
    pretrained_cfg = pretrained_model / "config.json"
    with open(pretrained_cfg, "r") as fp:
        model_cfg = json.load(fp)

    vla_cfg = utils.helpers.get_policy_cfg(cfg_file=pretrained_cfg)
    # Check whether the delta action setting is consistent
    if hasattr(vla_cfg, "use_delta_action"):
        assert vla_cfg.use_delta_action == use_delta_action
    # Update the config for inference
    if hasattr(vla_cfg, "skip_n_actions"):
        vla_cfg.skip_n_actions = skip_n_actions
    else:
        assert skip_n_actions == 0, "The model does not support action skipping."

    vla_model_class = utils.helpers.get_policy_class(model_cfg["type"])
    # Enable streaming inference if needed
    if streaming and hasattr(vla_cfg, "enable_streaming"):
        vla_cfg.enable_streaming = True
        vla_model = vla_model_class.get_streaming_model(pretrained_model, vla_cfg)
        logging.info("Enable streaming inference for the VLA model.")
    else:
        vla_model = vla_model_class.from_pretrained(pretrained_model, config=vla_cfg)
        vla_model.eval()
        if torch.cuda.is_available():
            vla_model = vla_model.cuda()

    if (
        "delta_timestamps" not in model_cfg
        or model_cfg["delta_timestamps"] is None
        or "observation" not in model_cfg["delta_timestamps"]
    ):
        assert vla_model.config.n_obs_steps == 1
        # Use current observation by default
        model_cfg["delta_timestamps"] = [0]
    else:
        assert vla_model.config.n_obs_steps == len(
            model_cfg["delta_timestamps"]["observation"]
        )
        model_cfg["delta_timestamps"] = model_cfg["delta_timestamps"]["observation"]

    logging.info("Observation timestamps: %s" % model_cfg["delta_timestamps"])
    return vla_model, {"delta_timestamps": model_cfg["delta_timestamps"]}


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
    observation = None
    while True:
        try:
            observation = obs_socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            break

    return observation


def run_tests(
    obs_socket,
    act_socket,
    vla_cfg,
    vla_model,
    vla_weights,
    vla_epoch_idx,
    rotation,
    use_delta_action,
    output_dir,
):
    n_tests = 0
    test_results = {}
    instruction = None
    actions = []
    observations = []
    while True:
        observation = get_latest_observation(obs_socket)
        if observation is None:
            continue

        # Test case or test suite is done
        if "success_rates" in observation:
            logging.info(
                "Test suite done with success rates: %s" % observation["success_rates"]
            )
            break
        elif "success" in observation:
            _handle_test_case_finished(
                observation,
                test_results,
                vla_model,
                vla_weights,
                vla_epoch_idx,
                actions,
                output_dir,
            )
            n_tests += 1
            logging.info(
                "[Test%02d] Test done with %s in %d steps and %d actions."
                % (
                    n_tests,
                    "SUCCESS" if observation["success"] else "FAILURE",
                    len(observation["ee_path"]),
                    len(observation["actions"]),
                )
            )
            # Send ack to the server
            act_socket.send_pyobj({"ack": n_tests}, flags=zmq.NOBLOCK)
            continue

        if "task" in observation:
            instruction = observation["task"]
            logging.info(
                "[Test%02d] Received new task: %s" % (n_tests, instruction.strip())
            )
            # Reset the model with the new instruction
            action = None
            actions = []
            observations = []
            vla_model.reset()
            setattr(_get_action, "states", [])
            if hasattr(_get_action, "count"):
                delattr(_get_action, "count")
        elif instruction is not None:
            observation["task"] = instruction
        else:
            continue

        observations.append(observation)
        # Generate and send an action
        action = _get_action(
            vla_model,
            _get_observations(observations, vla_cfg["delta_timestamps"]),
            rotation,
            use_delta_action,
            output_dir is not None,
        )
        if action is not None:
            act_socket.send_pyobj({"action": action}, flags=zmq.NOBLOCK)
            logging.debug("Sending action: %s", np.round(action, 2))
            actions.append(action)

    return n_tests, test_results


def _handle_test_case_finished(
    observation,
    test_results,
    vla_model,
    vla_weights,
    vla_epoch_idx,
    actions,
    output_dir,
):
    observation["actions"] = actions
    env_name = observation["env_name"]
    if env_name not in test_results:
        test_results[env_name] = []

    test_results[env_name].append(observation)
    # Save the debug states/actions
    if output_dir:
        _output_dir = os.path.join(
            output_dir, os.path.basename(vla_weights), "%04d" % vla_epoch_idx
        )
        os.makedirs(_output_dir, exist_ok=True)
        _dump_vla_states(
            observation["eps_name"],
            vla_model.config,
            getattr(_get_action, "states", []),
            actions,
            _output_dir,
        )


def _dump_vla_states(episode_name, vla_cfg, states, actions, output_dir):
    output_path = os.path.join(output_dir, "%s.pkl" % episode_name)
    with open(output_path, "wb") as fp:
        pickle.dump({"vla": vla_cfg, "state": states, "action": actions}, fp)


def _get_observations(observations, delta_timestamps):
    selected_observations = []
    last_idx = len(observations) - 1
    for dt in delta_timestamps:
        idx = last_idx + dt
        idx = max(0, min(last_idx, idx))
        selected_observations.append(observations[idx])

    return selected_observations


def _get_action(vla_model, observations, rotation, use_delta_action, debug=False):
    N_DUMMY_STEPS = 3
    _count = getattr(_get_action, "count", -N_DUMMY_STEPS)
    _state = getattr(_get_action, "state", None)
    for ifk in vla_model.config.input_features.keys():
        for observation in observations:
            if ifk not in observation:
                logging.warning("Ignoring observation without key: %s" % ifk)
                return None

    setattr(_get_action, "count", _count + 1)
    # Skip the first few steps to allow model to warm up
    if _count < 0:
        return None

    # NOTE: All tensors are on CPU if streaming is enabled.
    #       Because IPC with CUDA tensors is not supported.
    device = "cuda" if not vla_model.config.enable_streaming else "cpu"
    observations = _get_transformed_observations(
        observations, rotation, vla_model.config.input_features, device
    )

    # print(observations["index"], _count)
    # Update the state every chunk_size steps
    if _count % vla_model.config.n_action_steps == 0:
        _state = observations["observation.state"][:, -1, :].cpu().numpy()
        setattr(_get_action, "state", _state)
        if debug:
            states = getattr(_get_action, "states", [])
            states.append(_state)
            setattr(_get_action, "states", states)

    action = vla_model.select_action(observations).cpu().numpy()
    # If the model does not support delta action, we manually convert it here
    if not hasattr(vla_model.config, "use_delta_action") and use_delta_action:
        action_dim = action.shape[-1] - 1
        action[:, :action_dim] += _state[:, :action_dim]

    ee_pos = action[:, :3]
    ee_quat = utils.helpers.get_quaternion(action[:, 3:-1], rotation)
    gripper = action[:, -1:]
    action = np.concatenate([ee_pos, ee_quat, gripper], axis=-1)
    return action


def _get_transformed_observations(observations, rotation, feat_cfg, device="cuda"):
    tr_observations = {}
    for o in observations:
        tr_o = _get_transformed_observation(o, rotation, feat_cfg, device)
        for k, v in tr_o.items():
            if k not in tr_observations:
                tr_observations[k] = []

            tr_observations[k].append(v)

    tr_observations = {
        k: torch.stack(v, dim=1) for k, v in tr_observations.items() if k != "task"
    }
    for k in ["task", "index", "dt_scale"]:
        tr_observations[k] = observations[-1][k] if k in observations[-1] else None

    return tr_observations


def _get_transformed_observation(observation, rotation, feat_cfg, device="cuda"):
    images = {
        k: v.astype(np.float32) / 255.0
        for k, v in observation.items()
        if k.startswith("observation.image") and k in feat_cfg
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


def get_test_stats(test_results):
    stats = {}
    for env, results in test_results.items():
        n_steps = 0
        n_actions = 0
        n_success_trials = 0
        path_length = []
        for tr in results:
            if not tr["success"]:
                continue

            n_success_trials += 1
            n_steps += len(tr["ee_path"])
            n_actions += len(tr["actions"])
            path_length.append(
                np.sum(np.linalg.norm(np.diff(tr["ee_path"], axis=0), axis=-1))
            )

        stats[env] = {
            "n_tests": len(results),
            "success_rate": n_success_trials / len(results),
            "avg_steps": n_steps / len(results),
            "avg_actions": n_actions / len(results),
            "avg_path_length": np.mean(path_length).item() if path_length else 0,
        }

    return stats


def main(
    vla_weights,
    vla_alias,
    vla_epoch_idx,
    rotation,
    use_delta_action,
    streaming,
    skip_n_actions,
    host,
    img_port,
    act_port,
    output_dir,
):
    # Initialize the VLA model
    logging.info("Loading VLA model with weights: %s" % (vla_weights))
    vla_model, vla_cfg = get_vla_model(
        vla_weights, use_delta_action, streaming, skip_n_actions
    )
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
    # Hand-shake to the evaluation server
    act_socket.send_pyobj(
        {
            "vla": (
                vla_alias if vla_alias is not None else os.path.basename(vla_weights)
            ),
            "epoch": vla_epoch_idx,
        },
        flags=zmq.NOBLOCK,
    )
    # Start the evaluation
    n_tests, test_results = run_tests(
        obs_socket,
        act_socket,
        vla_cfg,
        vla_model,
        vla_weights,
        vla_epoch_idx,
        rotation,
        use_delta_action,
        output_dir,
    )
    logging.info("Evaluation completed. Total tests run: %d" % n_tests)
    logging.info("Test results: %s" % get_test_stats(test_results))


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
        "-r",
        "--rotation",
        type=str,
        default="quat",
        help="The representation of rotation in the action space",
    )
    parser.add_argument(
        "-d",
        "--delta",
        action="store_true",
        help="Whether to use delta action in the action space",
    )
    parser.add_argument(
        "-s",
        "--streaming",
        action="store_true",
        help="Whether to enable streaming inference",
    )
    parser.add_argument(
        "-n",
        "--skip_n_actions",
        type=int,
        default=0,
        help="The number of actions to skip",
    )
    parser.add_argument(
        "-p",
        "--weights",
        type=str,
        required=True,
        help="The path to the pretrained VLA model",
    )
    parser.add_argument(
        "-a",
        "--alias",
        type=str,
        default=None,
        help="The alias of the pretrained VLA model",
    )
    parser.add_argument(
        "-i",
        "--epoch",
        type=int,
        default=0,
        help="The epoch index of the pretrained VLA model",
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
        args.weights,
        args.alias,
        args.epoch,
        args.rotation,
        args.delta,
        args.streaming,
        args.skip_n_actions,
        args.host,
        args.img_port,
        args.act_port,
        args.output_dir,
    )
