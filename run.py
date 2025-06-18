# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Haozhe Xie
# @Date:   2025-03-14 15:09:46
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-06-18 18:42:33
# @Email:  root@haozhexie.com

import argparse
import datetime
import importlib
import logging
import os
import pprint
import sys

import cv2
import easydict
import torch
import yaml

import core
import utils.distributed

# Fix deadlock in DataLoader
cv2.setNumThreads(0)


def get_args_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="exp_name",
        help="The name of the experiment",
        default="%s" % datetime.datetime.now(),
        type=str,
    )
    parser.add_argument(
        "-c",
        "--cfg",
        dest="cfg_file",
        help="Path to the config.yaml file",
        default="config.yaml",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        help="The dataset name to train or test.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-g",
        "--gpus",
        dest="gpus",
        help="The GPU device to use (e.g., 0,1,2,3).",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-p",
        "--ckpt",
        dest="ckpt",
        help="Initialize the network from a pretrained model.",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--run",
        dest="run_id",
        help="The unique run ID for WandB",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--test", dest="test", help="Test the network.", action="store_true"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        help="The rank ID of the GPU. Automatically assigned by torch.distributed.",
        default=os.getenv("LOCAL_RANK", 0),
    )
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    # Read the experimental config
    with open(args.cfg_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
        cfg = easydict.EasyDict(cfg)

    # Parse runtime arguments
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.exp_name is not None:
        cfg.CONST.EXP_NAME = args.exp_name
    if args.dataset is not None:
        cfg.DATASET.NAME = args.dataset
    if args.ckpt is not None:
        cfg.CONST.CKPT = args.ckpt
    if args.run_id is not None:
        cfg.WANDB.RUN_ID = args.run_id
    if args.run_id is not None and args.ckpt is None:
        raise Exception("No checkpoints")

    # Print the current config
    local_rank = args.local_rank
    if local_rank == 0:
        pprint.pprint(cfg)

    # Initialize the DDP environment
    if torch.cuda.is_available() and not args.test:
        utils.distributed.set_affinity(local_rank)
        utils.distributed.init_dist(local_rank)

    # Start train/test processes
    if not args.test:
        try:
            core.train(cfg)
        finally:
            utils.distributed.cleanup_dist()
    else:
        if "CKPT" not in cfg.CONST or not os.path.exists(cfg.CONST.CKPT):
            logging.error("Please specify the file path of checkpoint.")
            sys.exit(2)
        try:
            core.test(cfg)
        finally:
            utils.distributed.cleanup_dist()


if __name__ == "__main__":
    # References: https://stackoverflow.com/a/53553516/1841143
    importlib.reload(logging)
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s"
    )
    main()
