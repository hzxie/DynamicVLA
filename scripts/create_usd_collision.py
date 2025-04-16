# -*- coding: utf-8 -*-
#
# @File:   create_usd_collision.py
# @Author: Haozhe Xie
# @Date:   2025-04-04 10:36:03
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-04-15 14:19:04
# @Email:  root@haozhexie.com

import argparse
import logging
import os
import shutil
import sys

import gymnasium as gym
import isaaclab.app
from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.dirname(__file__))


def set_object_scale(xform, category):
    from pxr import UsdGeom

    # TODO: Assign different scales to different categories
    OBJECT_SCALES = {
        "apple": 0.075,
        "can": 0.13,
    }

    scale_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    else:
        scale_op = xform.AddScaleOp()

    scale_op.Set((OBJECT_SCALES[category],) * 3)


def reset_object_material(prim):
    from pxr import UsdShade

    mtl = UsdShade.Material(prim)
    surface = mtl.GetSurfaceOutput().GetConnectedSource()[0]
    shader = UsdShade.Shader(surface)
    shader.GetInput("roughness").Set(0.5)
    shader.GetInput("useSpecularWorkflow").Set(0)


def main(input_dir, output_dir):
    from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    usd_files = os.listdir(input_dir)
    for uf in tqdm(usd_files):
        input_file = os.path.join(input_dir, uf)
        output_file = os.path.join(output_dir, uf)
        if output_file != input_file:
            shutil.copyfile(input_file, output_file)

        stage = Usd.Stage.Open(output_file)
        prim_names = [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()]
        is_house = "/house" in prim_names
        assert len(prim_names) == 1 or is_house

        if is_house:
            # Set the default prim to the house
            stage.SetDefaultPrim(stage.GetPrimAtPath("/house"))
        else:
            default_prim = stage.GetPrimAtPath("/object")
            stage.SetDefaultPrim(default_prim)
            xform = UsdGeom.Xform(default_prim)
            set_object_scale(xform, os.path.basename(uf).split("_")[0])
            reset_object_material(stage.GetPrimAtPath("/object/mtl/material_0"))


        # Create collision for all meshes in the stage
        for prim in tqdm(stage.Traverse(), leave=False):
            if prim.GetTypeName() == "Mesh":
                collider = UsdPhysics.CollisionAPI.Apply(prim)
                if not is_house:
                    # Create SDF collision for the object
                    mesh_collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
                    mesh_collider.CreateApproximationAttr().Set("sdf")
                    collider.GetCollisionEnabledAttr().Set(True)
                    collision_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                    collision_api.CreateSdfResolutionAttr().Set(1024)
                    # Enable rigid body API for the object
                    UsdPhysics.RigidBodyAPI.Apply(prim)

        stage.GetRootLayer().Save()


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
        "--output_dir", default=os.path.join(PROJECT_HOME, os.pardir, "scenes")
    )
    args = parser.parse_args(script_args)

    main(args.input_dir, args.output_dir)
    app_launcher.app.close()
