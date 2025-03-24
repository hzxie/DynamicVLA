# -*- coding: utf-8 -*-
#
# @File:   scene_cfg.py
# @Author: Haozhe Xie
# @Date:   2025-03-23 12:28:24
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-03-24 20:02:29
# @Email:  root@haozhexie.com

from dataclasses import MISSING

import isaaclab.sim as sim_utils
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
    object: RigidObjectCfg | DeformableObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0, 0, 1], rot=[1, 0, 0, 0], lin_vel=[0, 0, 0], ang_vel=[0, 0, 0]
        ),
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        ),
    )
    # the unique house asset (as background): will be populated by agent env cfg
    house: AssetBaseCfg = MISSING
    # Ground plane
    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    # Light
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


def add_camera_to_scene(scene_cfg, camera_cfg: dict) -> SceneCfg:
    scene_cfg.__setattr__(
        camera_cfg["name"],
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/cameras/%s" % camera_cfg["name"],
            update_period=camera_cfg["period"],
            height=camera_cfg["height"],
            width=camera_cfg["width"],
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=camera_cfg["focal_length"],
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=camera_cfg["position"], rot=camera_cfg["quat"], convention="ros"
            ),
        ),
    )
    return scene_cfg


def add_object_to_scene(
    scene_cfg: SceneCfg,
    object_name: str,
    object_cfg: RigidObjectCfg | DeformableObjectCfg,
) -> SceneCfg:
    scene_cfg.__setattr__(object_name, object_cfg)
    return scene_cfg


def set_target_object(
    scene_cfg: SceneCfg, target_object_cfg: RigidObjectCfg
) -> SceneCfg:
    scene_cfg.object = target_object_cfg
    return scene_cfg


def set_house_asset(scene_cfg: SceneCfg, scene_asset_usd_file: str) -> SceneCfg:
    # from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    scene_cfg.house = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/House",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=scene_asset_usd_file),
        # spawn=UsdFileCfg(
        #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        # ),
    )
    return scene_cfg
