# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2025-04-04 10:36:03
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-04-22 10:36:08
# @Email:  root@haozhexie.com

import argparse
import logging
import os
import sys

import isaaclab.app
from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.dirname(__file__))


def create_object_usd(curr_stage, root_prim_path):
    import isaacsim.core.utils.prims as prim_utils
    from pxr import UsdPhysics

    prim_utils.create_prim(
        root_prim_path,
        prim_type="Xform",
        translation=[0.0, 0.0, 0.0],
        orientation=[1.0, 0.0, 0.0, 0.0],
    )
    curr_stage.SetDefaultPrim(curr_stage.GetPrimAtPath(root_prim_path))
    UsdPhysics.RigidBodyAPI.Apply(curr_stage.GetPrimAtPath(root_prim_path))


def create_asset_prim(_, prim_path, usd_path, scale=(1.0, 1.0, 1.0)):
    import isaacsim.core.utils.prims as prim_utils
    from pxr import Usd

    prim_utils.create_prim(prim_path, prim_type="Mesh", usd_path=usd_path, scale=scale)


def _get_collider_type(category):
    COLLIDERS = {
        "apple": "Sphere",
        "can": "Cylinder",
    }
    assert category in COLLIDERS
    return COLLIDERS[category]


def create_dummy_collider(curr_stage, prim_path, category):
    import isaacsim.core.utils.prims as prim_utils
    from pxr import UsdGeom, UsdPhysics

    prim_utils.create_prim(prim_path, prim_type=_get_collider_type(category))
    UsdPhysics.CollisionAPI.Apply(curr_stage.GetPrimAtPath(prim_path))
    UsdGeom.Imageable(curr_stage.GetPrimAtPath(prim_path)).MakeInvisible()


def reset_object_material(prim):
    from pxr import UsdShade

    mtl = UsdShade.Material(prim)
    surface = mtl.GetSurfaceOutput().GetConnectedSource()[0]
    shader = UsdShade.Shader(surface)
    shader.GetInput("roughness").Set(0.5)
    shader.GetInput("useSpecularWorkflow").Set(0)


def main(input_dir, output_dir):
    import isaacsim.core.utils.stage as stage_utils

    usd_files = [f for f in os.listdir(input_dir) if f.endswith(".usd")]
    for uf in tqdm(usd_files):
        input_file = os.path.join(input_dir, uf)
        output_file = os.path.join(output_dir, uf)
        category = os.path.basename(input_file).split("_")[0]

        stage_utils.create_new_stage()
        stage = stage_utils.get_current_stage()
        create_object_usd(stage, "/Object")
        create_asset_prim(stage, "/Object/geometry", input_file)
        create_dummy_collider(stage, "/Object/collider", category)
        reset_object_material(stage.GetPrimAtPath("/Object/geometry/mtl/material_0"))

        # Remove external references
        stage.Flatten().Export(output_file)


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
