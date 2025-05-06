# -*- coding: utf-8 -*-
#
# @File:   visualze_dataset_seq.py
# @Author: Haozhe Xie
# @Date:   2025-05-05 14:29:46
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-05-06 11:14:29
# @Email:  root@haozhexie.com

import argparse
import os
import sys

import cv2
import h5py
from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.dirname(__file__))


import simulations.simulate as sim


def main(input_dir, output_dir):
    files = [f for f in os.listdir(input_dir) if f.endswith(".h5")]
    for file in tqdm(files):
        with h5py.File(os.path.join(input_dir, file), "r") as f:
            env_states = {k: f[k][()] for k in f.keys()}

        sim.dump_video(
            sim.get_frames(env_states),
            os.path.join(args.output_dir, "%s.mp4" % os.path.splitext(file)[0]),
        )
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", default=os.path.join(PROJECT_HOME, os.pardir, "datasets")
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(PROJECT_HOME, os.pardir, "datasets")
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
