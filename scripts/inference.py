# -*- coding: utf-8 -*-
#
# @File:   inference.py
# @Author: Haozhe Xie
# @Date:   2025-05-14 14:25:25
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-05-14 19:44:45
# @Email:  root@haozhexie.com

import argparse
import h5py
import logging
import time
import uuid
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
    img_socket = context.socket(zmq.SUB)
    img_socket.connect("tcp://%s:%d" % (host, img_port))
    if not is_socket_connected(context, img_socket):
        raise RuntimeError("Failed to connect to tcp://%s:%d" % (host, img_port))

    img_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    img_socket.RCVHWM = 1  # Only receive the latest image

    action_socket = context.socket(zmq.PUSH)
    action_socket.connect("tcp://%s:%d" % (host, act_port))
    if not is_socket_connected(context, action_socket):
        raise RuntimeError("Failed to connect to tcp://%s:%d" % (host, act_port))

    return img_socket, action_socket


def get_latest_image(img_socket):
    image = None
    while True:
        try:
            image = img_socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            break

    return image


def get_action(image):
    # TODO: Replace this function with VLA model inference
    frame_idx = getattr(get_action, "index", 0)
    with h5py.File("datasets/pick_franka_apple00d_O01_5d9c640bc1f7.h5", "r") as fp:
        action = fp["action"][()]

    get_action.index = frame_idx + 1
    return action[frame_idx][None, :] if frame_idx < len(action) else None


def main(host, img_port, act_port):
    img_socket, act_socket = get_zmq_sockets(host, img_port, act_port)
    logging.info(
        "Connected to %s:%d for images and %s:%d for actions"
        % (args.host, args.img_port, args.host, args.act_port)
    )
    while True:
        image = get_latest_image(img_socket)
        if image is None:
            continue

        action = get_action(image)
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
    args = parser.parse_args()
    main(args.host, args.img_port, args.act_port)
