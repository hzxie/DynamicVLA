# -*- coding: utf-8 -*-
#
# @File:   create_scene_collision.py
# @Author: Haozhe Xie
# @Date:   2025-04-04 10:36:03
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-03 09:56:20
# @Email:  root@haozhexie.com

import argparse
import json
import logging
import os
import sys
import uuid

import numpy as np
from omni.isaac.kit import SimulationApp
from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.dirname(__file__))


def apply_collision(stage, exclude_prim_keywords=[]):
    from isaaclab.sim.utils import safe_set_attribute_on_usd_schema
    from pxr import PhysxSchema, UsdPhysics

    for prim in tqdm(stage.Traverse(), leave=False):
        if prim.GetTypeName() == "Mesh":
            UsdPhysics.CollisionAPI.Apply(prim)
            usd_collision_api = UsdPhysics.CollisionAPI(prim)

            physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
            if not physx_collision_api:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)

            # set into USD API
            safe_set_attribute_on_usd_schema(
                usd_collision_api,
                "collision_enabled",
                not any(keyword in prim.GetName() for keyword in exclude_prim_keywords),
                camel_case=True,
            )


def regularize_tables(stage):
    from pxr import UsdGeom

    ACCEPTED_ANGLES = [0, 90, 180]
    ANGLE_THRESHOLD = 20
    for prim in tqdm(stage.Traverse(), leave=False):
        if prim.GetName().find("Table") == -1 or prim.GetTypeName() != "Xform":
            continue

        rot_op = [
            op
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
            if op.GetOpName() == "xformOp:rotateXYZ"
        ]
        rot_op = None if len(rot_op) == 0 else rot_op[0]
        if rot_op is not None:
            y_rotation = abs(rot_op.Get()[1]) % 180
            # Check whether the y-axis-rotation is near 0, 90, 180 degrees
            for angle in ACCEPTED_ANGLES:
                diff = abs(y_rotation - angle)
                if diff > 1e-3 and diff < ANGLE_THRESHOLD:
                    rotation = rot_op.Get()
                    rotation[1] = angle
                    logging.debug(
                        "Regularizing table: %s; Old rotation: %s; New rotation: %s",
                        prim.GetPath(),
                        rot_op.Get(),
                        rotation,
                    )
                    rot_op.Set(rotation)
                    break


def remove_table_objects(stage):
    from pxr import Usd, UsdGeom

    DIFF_THRESHOLD = 0.1
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in tqdm(stage.Traverse(), leave=False):
        if prim.GetName().find("Table") == -1:
            continue

        children = prim.GetChildren()
        if len(children) <= 1:
            continue

        bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        for child in children:
            _bbox = bbox_cache.ComputeWorldBound(child).ComputeAlignedBox()
            _bbox_diff = bbox.GetSize() - _bbox.GetSize()
            if (
                np.all(np.array(_bbox_diff) > DIFF_THRESHOLD)
                and np.sum(_bbox_diff) >= DIFF_THRESHOLD
            ):
                assert stage.RemovePrim(child.GetPath())


def get_table_heights(scene_usd_file):
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    HEIGHT_THRESHOLDS = (0.1, 2.0)
    usd_context = omni.usd.get_context()
    # Make current stage as the temporary stage to get the table assets
    usd_context.open_stage(scene_usd_file)
    stage = usd_context.get_stage()
    default_prim = stage.GetDefaultPrim()
    # Set z-axis as the up axis (to run the simulation)
    xform = UsdGeom.Xformable(default_prim)
    xform.AddOrientOp().Set(Gf.Quatf(0.7071068, 0.7071068, 0, 0))

    heights = {}
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in tqdm(stage.Traverse(), leave=False):
        if prim.GetName().find("Table") == -1 or prim.GetTypeName() != "Xform":
            continue

        bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        height = _get_table_height(prim, bbox)
        if height >= HEIGHT_THRESHOLDS[0] and height <= HEIGHT_THRESHOLDS[1]:
            heights[prim.GetPath()] = height
        else:
            logging.warning(
                "Table height could not be determined for: %s", prim.GetPath()
            )

    return heights


def _get_table_height(prim, bbox):
    from isaacsim.core.api.objects.cuboid import DynamicCuboid
    from isaacsim.core.utils.prims import delete_prim
    from omni.isaac.core import SimulationContext

    CUBE_SIZE = 0.05
    sim = SimulationContext()
    bbox_min, bbox_max = bbox.GetMin(), bbox.GetMax()
    x_values = [
        bbox_min[0] + CUBE_SIZE,
        bbox_max[0] - CUBE_SIZE,
        (bbox_min[0] + bbox_max[0]) / 2.0,
    ]
    y_values = [
        bbox_min[1] + CUBE_SIZE,
        bbox_max[1] - CUBE_SIZE,
        (bbox_min[1] + bbox_max[1]) / 2.0,
    ]
    cube_positions = [
        [x, y, bbox_max[2] + CUBE_SIZE] for x in x_values for y in y_values
    ]

    height_values = []
    for cp in cube_positions:
        cube_name = "Cube_%s" % str(uuid.uuid4())[-12:]
        cube = DynamicCuboid(
            prim_path="/World/%s" % cube_name,
            name=cube_name,
            position=cp,
            size=CUBE_SIZE,
        )
        sim.reset()
        # Simulation loop for each cube position
        for _ in range(1000):
            sim.step(render=False)
            cube_vel = np.linalg.norm(cube.get_linear_velocity())
            if cube_vel < 1e-3:
                pos = cube.get_world_pose()[0]
                height_values.append(pos[2] - CUBE_SIZE / 2.0)
                break

        delete_prim("/World/%s" % cube_name)

    logging.debug("Table: %s; Height: %s" % (prim.GetPath(), height_values))
    return np.mean(height_values) if np.std(height_values) < 1e-3 else -1


def save_table_heights(stage, heights):
    from pxr import Sdf

    for prim_path, height in heights.items():
        prim = stage.GetPrimAtPath(prim_path)
        if height != -1:
            prim.CreateAttribute("height", Sdf.ValueTypeNames.Float).Set(height)


def add_table_planes(stage, heights):
    from pxr import Gf, UsdGeom, UsdPhysics

    THICKNESS = 0.01
    for prim_path, height in heights.items():
        prim = stage.GetPrimAtPath(prim_path)
        # NOTE: bbox_cache.ComputeWorldBound may not be correct.
        bbox = _get_bounding_box(prim)
        size = bbox.GetSize()

        plane = UsdGeom.Cube.Define(stage, "%s/TablePlane" % prim_path)
        plane_prim = plane.GetPrim()
        UsdPhysics.CollisionAPI.Apply(plane_prim)
        # UsdPhysics.RigidBodyAPI.Apply(plane_prim)
        # Set the translation and size of the plane
        # NOTE:
        # 1. The default size of the plane is 2.0
        # 2. The y-axis is the up-axis in the USD
        # 3. The translation is the relative position of the table bbox
        plane.AddTranslateOp().Set(Gf.Vec3f(0.0, height - THICKNESS / 2, 0.0))
        plane.AddScaleOp().Set(Gf.Vec3f(size[0] / 2.0, THICKNESS, size[2] / 2.0))
        # Set the transparency of the plane
        UsdGeom.Imageable(plane_prim).MakeInvisible()


def _get_bounding_box(prim):
    from pxr import Usd, UsdGeom, Gf

    DEFAULT_TIME = Usd.TimeCode.Default()
    bbox = Gf.Range3d()
    for child in prim.GetChildren():
        if not child.GetTypeName() == "Mesh":
            continue

        mesh = UsdGeom.Mesh(child)
        for p in mesh.GetPointsAttr().Get(DEFAULT_TIME):
            bbox.UnionWith(Gf.Vec3d(p))
        # Update extent for future use
        mesh.GetExtentAttr().Set([bbox.GetMin(), bbox.GetMax()], DEFAULT_TIME)

    return bbox


def main(input_dir, output_dir, ignore_files=None, range=None):
    from pxr import Usd

    usd_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".usd")])
    if range is not None:
        start, end = range
        logging.info("Processing USD files from %d to %d" % (start, end))
        usd_files = usd_files[start:end]

    if ignore_files is not None:
        with open(ignore_files, "r") as fp:
            ignore_files = json.load(fp)

    for uf in tqdm(usd_files):
        if ignore_files is not None and uf in ignore_files:
            continue

        input_file = os.path.join(input_dir, uf)
        output_file = os.path.join(output_dir, uf)
        stage = Usd.Stage.Open(input_file)
        prim_names = [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()]
        assert "/house" in prim_names

        # Set the default prim to the house
        stage.SetDefaultPrim(stage.GetPrimAtPath("/house"))

        temporary_usd_file = os.path.join(output_dir, "T%s" % uf)
        # Create collision for all meshes in the stage
        apply_collision(stage)
        # Regularize tables
        regularize_tables(stage)
        # Remove objects on tables
        remove_table_objects(stage)
        # Set the height of the table (temporarily set z-axis as the up axis)
        stage.GetRootLayer().Export(temporary_usd_file)
        heights = get_table_heights(temporary_usd_file)

        # Remove the temporary USD file
        stage = Usd.Stage.Open(input_file)
        # The file cannot be removed while it is opened (on Windows)
        os.remove(temporary_usd_file)
        if len(heights) == 0:
            logging.warning("No tables found in the scene: %s", input_file)
            continue

        # Create collision for all meshes (excluding tables) in the stage
        apply_collision(stage, exclude_prim_keywords=["Table"])
        # Regularize tables
        regularize_tables(stage)
        # Remove objects on tables
        remove_table_objects(stage)
        # Save the height values of the tables
        save_table_heights(stage, heights)
        # Add table planes
        add_table_planes(stage, heights)
        # Save the modified stage to the output file
        stage.GetRootLayer().Export(output_file)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    # IssacSim Environment Initialization
    app = SimulationApp({"headless": True})

    # Arguments for the script
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", default=os.path.join(PROJECT_HOME, os.pardir, "USD")
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(PROJECT_HOME, os.pardir, "scenes")
    )
    parser.add_argument("--ignore_files", type=str, default=None)
    parser.add_argument("--range", type=int, nargs=2, default=None)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.ignore_files, args.range)
    app.close()
