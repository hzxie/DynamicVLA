# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2025-04-04 10:36:03
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-05-06 18:57:58
# @Email:  root@haozhexie.com

import argparse
import logging
import os
import sys
import json

import isaaclab.app
from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.dirname(__file__))


def get_object_size(usd_path):
    import pxr
    import omni.usd

    usd_context = omni.usd.get_context()
    usd_context.open_stage(usd_path)
    stage = usd_context.get_stage()
    default_prim = stage.GetDefaultPrim()
    bbox_cache = pxr.UsdGeom.BBoxCache(
        pxr.Usd.TimeCode.Default(), [pxr.UsdGeom.Tokens.default_]
    )
    bbox = bbox_cache.ComputeWorldBound(default_prim).ComputeAlignedBox()
    size = bbox.max - bbox.min
    usd_context.new_stage()
    return size


def main(object_dir):
    categories = [f for f in os.listdir(object_dir)]

    all_bbox_json = {}

    for category in categories :
        all_bbox_json[category] = {}
        objects = [f for f in os.listdir(os.path.join(object_dir, category)) if f.endswith(".usd")]
        
        for object in objects :
            usd_path = os.path.join(object_dir, category, object)
            size = get_object_size(usd_path)
            all_bbox_json[category][object] = size

    with open(os.path.join(object_dir, "object_size_metadata.json"), 'w') as f:
        json.dump(all_bbox_json, f, indent = 4)


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
        "--object_dir", default=os.path.join(PROJECT_HOME, os.pardir, "objects")
    )
    args = parser.parse_args(script_args)

    main(args.object_dir)
    app_launcher.app.close()
