# -*- coding: utf-8 -*-
#
# @File:   object_cfg.py
# @Author: Haozhe Xie
# @Date:   2025-04-16 14:38:58
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-04-23 18:46:45
# @Email:  root@haozhexie.com

import isaaclab.sim as sim_utils
import numpy as np
import scipy.spatial.transform
from isaaclab.assets import DeformableObjectCfg, RigidObjectCfg
from isaaclab.sim.spawners import SpawnerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


def get_object_cfg(
    obj_cfg: dict, spawner_cfg: SpawnerCfg
) -> RigidObjectCfg | DeformableObjectCfg:
    init_state = RigidObjectCfg.InitialStateCfg(pos=obj_cfg["pos"])
    if "lin_vel" in obj_cfg:
        init_state.lin_vel = obj_cfg["lin_vel"]
    if "ang_vel" in obj_cfg:
        init_state.ang_vel = obj_cfg["ang_vel"]
    if "quat" in obj_cfg:
        init_state.rot = obj_cfg["quat"]

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=init_state,
        spawn=spawner_cfg,
    )


def get_spawner_cfg(file_path: str = None, mass: int = 0.5) -> SpawnerCfg:
    if file_path is not None:
        spawner_cfg = UsdFileCfg(
            usd_path=file_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    else:
        # spawner_cfg = sim_utils.SphereCfg(
        spawner_cfg = sim_utils.CylinderCfg(
            radius=0.03,
            height=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        )
    return spawner_cfg


def get_object_init_quat(init_lin_vel: list[float]) -> list[float]:
    quat = scipy.spatial.transform.Rotation.from_euler(
        "xyz",
        [
            np.pi / 2 * np.random.choice([-1, 1]),
            0,
            -np.arctan2(init_lin_vel[1], init_lin_vel[0]),
        ],
    ).as_quat()
    return quat[[1, 2, 3, 0]]
