# -*- coding: utf-8 -*-
#
# @File:   eval_checkpoints.py
# @Author: Haozhe Xie
# @Date:   2025-08-01 07:40:13
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-20 08:27:50
# @Email:  root@haozhexie.com

import argparse
import ast
import fnmatch
import logging
import os
import re
import shutil
import subprocess
import time

import torch.utils.tensorboard


def get_new_checkpoints(ckpt_dir, ckpt_pattern=None):
    known_checkpoints = getattr(get_new_checkpoints, "checkpoints", [])
    new_checkpoints = []
    for root, _, files in os.walk(ckpt_dir):
        for file in files:
            ckpt_path = os.path.join(root, file)
            if ckpt_pattern and not fnmatch.fnmatch(ckpt_path, ckpt_pattern):
                continue

            fm_time = time.time() - os.path.getmtime(ckpt_path)
            # Make sure that the file has been transferred for at least 2 minutes
            if (
                file.endswith(".safetensors")
                and ckpt_path not in known_checkpoints
                and fm_time >= 120
            ):
                logging.info(
                    "Found new checkpoint %s modified at %s minute(s) ago"
                    % (ckpt_path, fm_time // 60)
                )
                new_checkpoints.append(ckpt_path)

    setattr(get_new_checkpoints, "checkpoints", known_checkpoints + new_checkpoints)
    return sorted(new_checkpoints)


def get_ckpt_info(ckpt_path):
    tokens = ckpt_path.split(os.path.sep)
    assert len(tokens) >= 2, "Invalid checkpoint path: %s" % ckpt_path
    exp_name = tokens[-2]
    epoch_idx = re.search(r"epoch(\d+)", tokens[-1])
    try:
        epoch_idx = int(epoch_idx.group(1))
    except Exception:
        epoch_idx = None

    return exp_name, epoch_idx


def test_checkpoint(
    vla_weights,
    work_dir,
    rotation,
    use_delta_action,
    streaming,
    host,
    img_port,
    act_port,
):
    # Construct the checkpoints in the work directory
    os.makedirs(work_dir, exist_ok=True)
    shutil.copy(vla_weights, os.path.join(work_dir, "model.safetensors"))
    shutil.copy(
        os.path.join(os.path.dirname(vla_weights), "config.json"),
        os.path.join(work_dir, "config.json"),
    )

    matches = re.search(r"/([^/]+)/model\.epoch(\d+)\.safetensors$", vla_weights)
    # fmt: off
    args = [
        "python", os.path.join(os.path.dirname(__file__), "inference.py"),
        "--host", host,
        "--img_port", str(img_port),
        "--act_port", str(act_port),
        "-a", matches.group(1),
        "-i", matches.group(2),
        "-p", work_dir,
        "-r", rotation,
    ]
    # fmt: on
    if use_delta_action:
        args.append("-d")
    if streaming:
        args.append("-s")

    test_results = None
    try:
        logging.info("Executing command:\n%s" % " ".join(args))
        output = subprocess.check_output(args, stderr=subprocess.STDOUT, text=True)
    except Exception as ex:
        logging.exception(ex)
        logging.exception(ex.output)
        return None

    try:
        test_results = re.search(r"Test results: ({.*})", output)
        test_results = ast.literal_eval(test_results.group(1))
    except Exception as ex:
        logging.error("Failed to parse test results from output:\n%s" % output)
        logging.exception(ex)

    return test_results


def add_tensorboard_scalars(test_results, ep_idx, tb_writer):
    average = {}
    for env, results in test_results.items():
        for k, v in results.items():
            tb_writer.add_scalar("%s/%s" % (env, k), v, ep_idx)
            if k not in average:
                average[k] = []

            average[k].append(v)

    for k, v in average.items():
        tb_writer.add_scalar("overall/%s" % k, sum(v) / len(v), ep_idx)


def main(
    log_dir,
    work_dir,
    ckpt_dir,
    ckpt_pattern,
    rotation,
    use_delta_action,
    streaming,
    host,
    img_port,
    act_port,
):
    # Load previously evaluated checkpoints
    eval_ckpts_file_path = os.path.join(work_dir, "checkpoints.txt")
    if os.path.exists(eval_ckpts_file_path):
        with open(eval_ckpts_file_path, "r") as fp:
            setattr(
                get_new_checkpoints,
                "checkpoints",
                [line for line in fp.read().splitlines() if os.path.exists(line)],
            )

    # TensorBoard writers for different experiments
    tb_writers = {}
    # Evaluate checkpoints in a loop
    while True:
        new_checkpoints = get_new_checkpoints(ckpt_dir, ckpt_pattern)
        if not new_checkpoints:
            logging.info("No new checkpoints found.")
            time.sleep(60)
            continue

        for nc in new_checkpoints:
            exp_name, ep_idx = get_ckpt_info(nc)
            if exp_name is None or ep_idx is None:
                logging.warning("Ignoring checkpoint %s due to invalid format." % nc)
                continue

            logging.info("Testing checkpoint %s" % (nc))
            test_results = test_checkpoint(
                nc,
                work_dir,
                rotation,
                use_delta_action,
                streaming,
                host,
                img_port,
                act_port,
            )
            with open(eval_ckpts_file_path, "a") as fp:
                fp.write("%s\n" % nc)
                fp.write("%s\n" % test_results)

            if test_results is None:
                logging.error("Failed to get test results for checkpoint %s" % nc)
                continue

            logging.info("Test results for checkpoint %s: %s" % (nc, test_results))
            if exp_name not in tb_writers:
                tb_writers[exp_name] = torch.utils.tensorboard.SummaryWriter(
                    os.path.join(log_dir, exp_name)
                )

            add_tensorboard_scalars(test_results, ep_idx, tb_writers[exp_name])


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
        "--log_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), os.path.pardir, "runs", "logs"),
        help="The path to the TensorBoard directory",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="/tmp/vla-checkpoint",
        help="The path to the work directory",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="The path to the checkpoint directory",
    )
    parser.add_argument(
        "-p",
        "--ckpt_pattern",
        type=str,
        default=None,
        help="The pattern to match checkpoints in the directory",
    )
    args = parser.parse_args()
    main(
        args.log_dir,
        args.work_dir,
        args.ckpt_dir,
        args.ckpt_pattern,
        args.rotation,
        args.delta,
        args.streaming,
        args.host,
        args.img_port,
        args.act_port,
    )
