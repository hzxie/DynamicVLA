# -*- coding: utf-8 -*-
#
# @File:   scene_cfg.py
# @Author: Haozhe Xie
# @Date:   2025-03-23 12:28:24
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-04-04 13:31:04
# @Email:  root@haozhexie.com

import logging
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import numpy as np
import omni.usd
import pxr
import scipy.spatial.transform
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: placeholder. Can be replaced by calling `set_target_object`
    # more objects can be added to the scene by calling `add_object_to_scene`
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Default house asset (as background): will be populated by agent env cfg
    house: AssetBaseCfg = MISSING

    # Default lightings
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            enable_color_temperature=True,
            color_temperature=6500,
            intensity=350,
        ),
    )
    distant_light: AssetBaseCfg = MISSING

    # Default ground plane assets
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.1]),
        spawn=GroundPlaneCfg(visible=False, color=(0.0, 0.0, 0.0)),
    )


def get_camera_cfg(cam_cfg: dict, cam_extra_cfg: dict) -> SceneCfg:
    for k, v in cam_extra_cfg.items():
        cam_cfg[k] = v

    # Set default values
    if "pos" not in cam_cfg:
        cam_cfg["pos"] = [0, 0, 0]
    if "quat" not in cam_cfg:
        cam_cfg["quat"] = [1, 0, 0, 0]
    if "convention" not in cam_cfg:
        cam_cfg["convention"] = "ros"

    prim_path = cam_cfg["prim_path"] if "prim_path" in cam_cfg else "/Robot/SideCamera"
    camera_cfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}" + prim_path,
        update_period=1 / cam_cfg["fps"],
        height=cam_cfg["height"],
        width=cam_cfg["width"],
        data_types=cam_cfg["data_types"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cam_cfg["focal_length"],
            focus_distance=cam_cfg["focus_distance"],
            horizontal_aperture=cam_cfg["horizontal_aperture"],
            clipping_range=(cam_cfg["clip"]["near"], cam_cfg["clip"]["far"]),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=cam_cfg["pos"], rot=cam_cfg["quat"], convention=cam_cfg["convention"]
        ),
    )
    return camera_cfg


def add_scene_camera(
    scene_cfg: SceneCfg, cam_name: str, cam_cfg: CameraCfg
) -> SceneCfg:
    scene_cfg.__setattr__(cam_name, cam_cfg)
    return scene_cfg


def set_light_asset(
    scene_cfg: SceneCfg,
    position: list = [0.0, 0.0, 0.0],
    temperature: int = 6500,
    intensity: float = 1000,
) -> SceneCfg:
    quat = get_quat_from_look_at(position, [0.0, 0.0, 0.0])

    scene_cfg.distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=quat),
        spawn=sim_utils.DistantLightCfg(
            enable_color_temperature=True,
            color_temperature=temperature,
            intensity=intensity,
        ),
    )
    return scene_cfg


def get_quat_from_look_at(cam_pos, cam_look_at):
    fwd_vec = np.array(
        [
            cam_look_at[0] - cam_pos[0],
            cam_look_at[1] - cam_pos[1],
            cam_look_at[2] - cam_pos[2],
        ]
    )
    fwd_vec /= np.linalg.norm(fwd_vec)
    up_vec = np.array([0, 0, 1])
    right_vec = np.cross(up_vec, fwd_vec)
    right_vec /= np.linalg.norm(right_vec)
    up_vec = np.cross(fwd_vec, right_vec)
    R = np.stack([fwd_vec, right_vec, up_vec], axis=1)
    quat = scipy.spatial.transform.Rotation.from_matrix(R).as_quat()
    return [quat[3], quat[0], quat[1], quat[2]]


def add_object_to_scene(
    scene_cfg: SceneCfg, object_name: str, object_cfg: dict
) -> SceneCfg:
    scene_cfg.__setattr__(object_name, _get_object_cfg(object_cfg))
    return scene_cfg


def set_target_object(scene_cfg: SceneCfg, object_cfg: dict) -> SceneCfg:
    scene_cfg.object = _get_object_cfg(object_cfg)
    return scene_cfg


def _get_object_cfg(obj_cfg: dict) -> RigidObjectCfg | DeformableObjectCfg:
    init_state = RigidObjectCfg.InitialStateCfg(pos=obj_cfg["pos"], rot=[1, 0, 0, 0])
    if "lin_vel" in obj_cfg:
        init_state.lin_vel = obj_cfg["lin_vel"]
    if "ang_vel" in obj_cfg:
        init_state.ang_vel = obj_cfg["ang_vel"]

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=init_state,
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        ),
    )


def set_house_asset(
    scene_cfg: SceneCfg, scene_asset_usd_file: str, scene_offset: list = [0, 0, 0]
) -> SceneCfg:
    scene_cfg.house = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/House",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=scene_offset, rot=[0.7071068, 0.7071068, 0, 0]
        ),
        spawn=UsdFileCfg(usd_path=scene_asset_usd_file),
    )
    return scene_cfg


def get_table_assets(scene_asset_usd_file: str, table_limits: dict) -> list:
    TABLE_ASSET_KEYWORD = "Table"
    TABLE_ASSET_GRP_NAME = "/house/furniture"
    WALL_ASSET_GRP_NAME = "/house/structure"

    usd_context = omni.usd.get_context()
    # Make current stage as the temporary stage to get the table assets
    usd_context.open_stage(scene_asset_usd_file)
    stage = usd_context.get_stage()
    default_prim = stage.GetDefaultPrim()
    # Set z-axis as the up axis
    xform = pxr.UsdGeom.Xformable(default_prim)
    xform.AddOrientOp().Set(pxr.Gf.Quatf(0.7071068, 0.7071068, 0, 0))
    # Get the table assets
    tables = []
    bbox_cache = pxr.UsdGeom.BBoxCache(
        pxr.Usd.TimeCode.Default(), [pxr.UsdGeom.Tokens.default_]
    )
    structure_primtives = stage.GetPrimAtPath(WALL_ASSET_GRP_NAME).GetChildren()
    furniture_primtives = stage.GetPrimAtPath(TABLE_ASSET_GRP_NAME).GetChildren()
    for prim in furniture_primtives:
        # There may be some assets on the table top
        if (
            TABLE_ASSET_KEYWORD in prim.GetName()
            and len(prim.GetChildren()) <= table_limits["max_children"]
        ):
            bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            size = bbox.max - bbox.min
            if (
                sum(size[:2]) >= table_limits["min_wh_sum"]
                and (np.array(size) != 0).all()
            ):
                anchors = _get_table_anchors(bbox, size)
                anchors = _get_uncollided_anchors(
                    anchors, furniture_primtives + structure_primtives, bbox_cache
                )
                # Check whether the table contains at least one anchor point for short
                # and long sides
                long_side_anchors = [a for a in anchors if a["side"] == "long"]
                short_side_anchors = [a for a in anchors if a["side"] == "short"]
                if len(long_side_anchors) > 0 and len(short_side_anchors) > 0:
                    tables.append({"anchors": anchors, "bbox": bbox})

    # Create a new stage for the subsequent simulations
    usd_context.new_stage()
    return tables


def _get_table_anchors(bbox: pxr.Gf.BBox3d, size: pxr.Gf.Vec3d) -> dict:
    # NOTE: The corner points (0-3) and anchor points (A-D) of the table are defined as:
    #   0   A   1
    #   +-------+
    # D |       | B
    #   +-------+
    #   3   C   2
    _get_mid_point = lambda pt1, pt2: [(pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2]

    z = max(bbox.min[2], bbox.max[2])
    corners = [
        (bbox.min[0], bbox.min[1]),
        (bbox.max[0], bbox.min[1]),
        (bbox.max[0], bbox.max[1]),
        (bbox.min[0], bbox.max[1]),
    ]
    anchors = [
        {
            "pos": np.array(_get_mid_point(corners[0], corners[1]) + [z]),
            "quat": np.array([0.7071068, 0, 0, 0.7071068]),
            "side": "long" if size[0] > size[1] else "short",
            "collision": False,
        },
        {
            "pos": np.array(_get_mid_point(corners[1], corners[2]) + [z]),
            "quat": np.array([0, 0, 0, 1]),
            "side": "short" if size[0] > size[1] else "long",
            "collision": False,
        },
        {
            "pos": np.array(_get_mid_point(corners[2], corners[3]) + [z]),
            "quat": np.array([-0.7071068, 0, 0, 0.7071068]),
            "side": "long" if size[0] > size[1] else "short",
            "collision": False,
        },
        {
            "pos": np.array(_get_mid_point(corners[3], corners[0]) + [z]),
            "quat": np.array([1, 0, 0, 0]),
            "side": "short" if size[0] > size[1] else "long",
            "collision": False,
        },
    ]
    return anchors


def _get_uncollided_anchors(
    anchors: list, primtives: list, bbox_cache: pxr.UsdGeom.BBoxCache
) -> list:
    ANCHOR_SIZE = 0.25
    for prim in primtives:
        bbox_cache = pxr.UsdGeom.BBoxCache(
            pxr.Usd.TimeCode.Default(), [pxr.UsdGeom.Tokens.default_]
        )
        bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        for anchor in anchors:
            if anchor["collision"]:
                continue

            anchor_bbox = pxr.Gf.Range3d(
                pxr.Gf.Vec3d(
                    anchor["pos"][0] - ANCHOR_SIZE,
                    anchor["pos"][1] - ANCHOR_SIZE,
                    anchor["pos"][2],
                ),
                pxr.Gf.Vec3d(
                    anchor["pos"][0] + ANCHOR_SIZE,
                    anchor["pos"][1] + ANCHOR_SIZE,
                    anchor["pos"][2] + ANCHOR_SIZE,
                ),
            )
            if _is_bbox3d_intersects(bbox, anchor_bbox):
                anchor["collision"] = True
                logging.debug("Anchor %s collides with %s", anchor, prim.GetName())

    return [a for a in anchors if not a["collision"]]


def _is_bbox3d_intersects(bbox1: pxr.Gf.Range3d, bbox2: pxr.Gf.Range3d) -> bool:
    return (
        bbox1.min[0] < bbox2.max[0]
        and bbox1.max[0] > bbox2.min[0]
        and bbox1.min[1] < bbox2.max[1]
        and bbox1.max[1] > bbox2.min[1]
        and bbox1.min[2] < bbox2.max[2]
        and bbox1.max[2] > bbox2.min[2]
    )
