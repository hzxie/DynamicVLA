# -*- coding: utf-8 -*-
#
# @File:   create_scene_collision.py
# @Author: Haozhe Xie
# @Date:   2025-04-04 10:36:03
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-05-06 18:58:20
# @Email:  root@haozhexie.com

import argparse
import logging
import os
import sys

import isaaclab.app
from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.dirname(__file__))


def main(input_dir, output_dir):
    from pxr import Usd, UsdPhysics

    usd_files = [f for f in os.listdir(input_dir) if f.endswith(".usd")]
    for uf in tqdm(usd_files):
        input_file = os.path.join(input_dir, uf)
        output_file = os.path.join(output_dir, uf)

        stage = Usd.Stage.Open(input_file)
        prim_names = [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()]
        assert "/house" in prim_names

        # Set the default prim to the house
        stage.SetDefaultPrim(stage.GetPrimAtPath("/house"))
        # Create collision for all meshes in the stage
        for prim in tqdm(stage.Traverse(), leave=False):
            if prim.GetTypeName() == "Mesh":
                UsdPhysics.CollisionAPI.Apply(prim)

        stage.GetRootLayer().Export(output_file)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Isaac Simulation Runner")
    # Arguments for the IsaacLab
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save the data from camera at index specified by ``--camera_id``.",
    )
    # IssacSim Environment Initialization
    isaaclab.app.AppLauncher.add_app_launcher_args(parser)
    isaaclab_args, script_args = parser.parse_known_args()
    app_launcher = isaaclab.app.AppLauncher(isaaclab_args)

    # Arguments for the script
    parser.add_argument(
        "--input_dir", default=os.path.join(PROJECT_HOME, os.pardir, "USD")
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(PROJECT_HOME, os.pardir, "objects")
    )
    args = parser.parse_args(script_args)

    main(args.input_dir, args.output_dir)
    app_launcher.app.close()
