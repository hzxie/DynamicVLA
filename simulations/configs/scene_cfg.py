# -*- coding: utf-8 -*-
#
# @File:   scene_cfg.py
# @Author: Haozhe Xie
# @Date:   2025-03-23 12:28:24
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-08-20 18:36:35
# @Email:  root@haozhexie.com

import logging
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


def get_camera_cfg(cam_cfg: dict, cam_extra_cfg: dict = {}) -> SceneCfg:
    for k, v in cam_extra_cfg.items():
        cam_cfg[k] = v

    # Set default values
    if "pos" not in cam_cfg:
        cam_cfg["pos"] = [0, 0, 0]
    if "quat" not in cam_cfg:
        cam_cfg["quat"] = [1, 0, 0, 0]
    if "convention" not in cam_cfg:
        cam_cfg["convention"] = "ros"

    camera_cfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}" + cam_cfg["prim_path"],
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
    scene_cfg: SceneCfg,
    object_name: str,
    object_cfg: RigidObjectCfg | DeformableObjectCfg,
) -> SceneCfg:
    scene_cfg.__setattr__(object_name, object_cfg)
    return scene_cfg


def set_target_object(
    scene_cfg: SceneCfg, object_cfg: RigidObjectCfg | DeformableObjectCfg
) -> SceneCfg:
    scene_cfg.object = object_cfg
    return scene_cfg


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


def get_table_assets(scene_asset_usd_file: str, cameras: list[dict]) -> list:
    TABLE_ASSET_KEYWORD = "Table"
    TABLE_ASSET_GRP_NAME = "/house/furniture"
    WALL_ASSET_GRP_NAME = "/house/structure"

    usd_context = omni.usd.get_context()
    # Make current stage as the temporary stage to get the table assets
    usd_context.open_stage(scene_asset_usd_file)
    stage = usd_context.get_stage()
    # Set z-axis as the up axis
    default_prim = stage.GetDefaultPrim()
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
        if TABLE_ASSET_KEYWORD in prim.GetName() and prim.HasAttribute("height"):
            bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            # Check whether the table contains at least one anchor point for long sides
            table_height = prim.GetAttribute("height").Get()
            table_anchors = _get_table_anchors(bbox, bbox.GetSize(), table_height)
            table_anchors = _get_uncollided_anchors(
                prim.GetName(),
                table_anchors,
                cameras,
                [fp for fp in furniture_primtives if fp != prim] + structure_primtives,
                bbox_cache,
            )
            long_side_anchors = [a for a in table_anchors if a["side"] == "long"]
            if len(long_side_anchors) > 0:
                bbox.SetMax(pxr.Gf.Vec3d(bbox.max[0], bbox.max[1], table_height))  # Replace table height values in bbox
                tables.append(
                    {"name": prim.GetName(), "anchors": table_anchors, "bbox": bbox}
                )

    # Create a new stage for the subsequent simulations
    usd_context.new_stage()
    return tables


def _get_table_anchors(bbox: pxr.Gf.BBox3d, size: pxr.Gf.Vec3d, height: float) -> dict:
    # NOTE: The corner points (0-3) and anchor points (A-D) of the table are defined as:
    #   0   A   1
    #   +-------+
    # D |       | B
    #   +-------+
    #   3   C   2
    _get_mid_point = lambda pt1, pt2: [(pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2]

    # z is replaced by the height of the table (generated in create_scene_collision.py)
    # z = max(bbox.min[2], bbox.max[2])
    corners = [
        (bbox.min[0], bbox.min[1]),
        (bbox.max[0], bbox.min[1]),
        (bbox.max[0], bbox.max[1]),
        (bbox.min[0], bbox.max[1]),
    ]
    anchors = [
        {
            "pos": np.array(_get_mid_point(corners[0], corners[1]) + [height]),
            "quat": np.array([0.7071068, 0, 0, 0.7071068]),
            "side": "long" if size[0] > size[1] else "short",
            "collision": False,
        },
        {
            "pos": np.array(_get_mid_point(corners[1], corners[2]) + [height]),
            "quat": np.array([0, 0, 0, 1]),
            "side": "short" if size[0] > size[1] else "long",
            "collision": False,
        },
        {
            "pos": np.array(_get_mid_point(corners[2], corners[3]) + [height]),
            "quat": np.array([-0.7071068, 0, 0, 0.7071068]),
            "side": "long" if size[0] > size[1] else "short",
            "collision": False,
        },
        {
            "pos": np.array(_get_mid_point(corners[3], corners[0]) + [height]),
            "quat": np.array([1, 0, 0, 0]),
            "side": "short" if size[0] > size[1] else "long",
            "collision": False,
        },
    ]
    return anchors


def _get_uncollided_anchors(
    table_name: str,
    table_anchors: list[np.array],
    cameras: list[dict],
    primtives: list,
    bbox_cache: pxr.UsdGeom.BBoxCache,
) -> list:
    for prim in primtives:
        bbox_cache = pxr.UsdGeom.BBoxCache(
            pxr.Usd.TimeCode.Default(), [pxr.UsdGeom.Tokens.default_]
        )
        prim_bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        for ta in table_anchors:
            if ta["collision"]:
                continue

            # Check if the table anchor collides with the primitives
            if _is_anchor_collided(prim_bbox, ta["pos"], 0.25):
                ta["collision"] = True
                logging.debug(
                    "[%s] Anchor %s collides with %s"
                    % (table_name, ta["pos"], prim.GetName())
                )

            # Check if the camera anchor collides with the primitives
            for cam in cameras:
                if _is_anchor_collided(prim_bbox, _get_camera_position(ta, cam), 0.05):
                    ta["collision"] = True
                    logging.debug(
                        "[%s] Camera %s of %s collides with %s"
                        % (table_name, cam["name"], ta["pos"], prim.GetName())
                    )

    return [ta for ta in table_anchors if not ta["collision"]]


def _get_camera_position(anchor, cam):
    # scalar_first is not supported in scipy < 1.12.0 (required by Isaac Lab)
    r = scipy.spatial.transform.Rotation.from_quat(
        [anchor["quat"][1], anchor["quat"][2], anchor["quat"][3], anchor["quat"][0]]
    )
    return r.apply(cam["position"]) + anchor["pos"]


def _is_anchor_collided(
    prim_bbox: pxr.Gf.Range3d, anchor_position: dict, anchor_size: float
) -> bool:
    anchor_bbox = pxr.Gf.Range3d(
        pxr.Gf.Vec3d(
            anchor_position[0] - anchor_size,
            anchor_position[1] - anchor_size,
            anchor_position[2],
        ),
        pxr.Gf.Vec3d(
            anchor_position[0] + anchor_size,
            anchor_position[1] + anchor_size,
            anchor_position[2] + anchor_size,
        ),
    )
    return _is_bbox3d_intersects(prim_bbox, anchor_bbox)


def _is_bbox3d_intersects(bbox1: pxr.Gf.Range3d, bbox2: pxr.Gf.Range3d) -> bool:
    return (
        bbox1.min[0] < bbox2.max[0]
        and bbox1.max[0] > bbox2.min[0]
        and bbox1.min[1] < bbox2.max[1]
        and bbox1.max[1] > bbox2.min[1]
        and bbox1.min[2] < bbox2.max[2]
        and bbox1.max[2] > bbox2.min[2]
    )
