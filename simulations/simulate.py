# -*- coding: utf-8 -*-
#
# @File:   simulate.py
# @Author: Haozhe Xie
# @Date:   2025-03-22 20:59:36
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-12-07 14:30:05
# @Email:  root@haozhexie.com

import argparse
import ast
import copy
import importlib
import json
import logging
import os
import random
import sys
import uuid

import cv2
import gymnasium as gym
import h5py
import imageio.v3
import numpy as np
import torch
import yaml
from isaaclab.app import AppLauncher
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

from simulations import helpers


def get_object_metadata(object_dir, target_categories=[]):
    object_metadata = {}
    # Get the sizes of all objects (for grasping)
    object_sizes = _get_object_sizes(args.object_dir, target_categories)
    for k, v in object_sizes.items():
        object_name = os.path.basename(k)
        object_category = os.path.basename(os.path.dirname(k))
        object_metadata[object_name] = {
            "file_path": k,
            "size": v,
            "category": object_category,
            "tags": [object_category],  # Default tag for instruction generation
        }

    # Load additional metadata if available
    metadata_file = os.path.join(object_dir, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        for file_name, _metadata in metadata.items():
            if file_name in object_metadata:
                for k, v in _metadata.items():
                    object_metadata[file_name][k] = v
            else:
                logging.warning("Metadata found for unknown object %s." % file_name)

    return object_metadata


def _get_object_sizes(object_dir, target_categories=None):
    object_sizes = {}
    categories = sorted([f for f in os.listdir(object_dir)])
    for c in categories:
        if target_categories and c not in target_categories:
            continue

        objects = [
            f for f in os.listdir(os.path.join(object_dir, c)) if f.endswith(".usd")
        ]
        for o in objects:
            usd_path = os.path.join(object_dir, c, o)
            object_sizes[usd_path] = _get_object_size(usd_path)

    return object_sizes


def _get_object_size(usd_path):
    import omni.usd
    import pxr

    usd_context = omni.usd.get_context()
    usd_context.open_stage(usd_path)
    stage = usd_context.get_stage()
    default_prim = stage.GetDefaultPrim()
    bbox_cache = pxr.UsdGeom.BBoxCache(
        pxr.Usd.TimeCode.Default(), [pxr.UsdGeom.Tokens.default_]
    )
    bbox = bbox_cache.ComputeWorldBound(default_prim).ComputeAlignedBox()
    usd_context.new_stage()
    return np.array(bbox.GetSize(), dtype=np.float32)


def get_env_cfg(sim_cfg, task, robot, object_metadata, scene_dir):
    import configs.scene_cfg
    import configs.termination_cfg
    import isaaclab_tasks

    gym.register(
        id="Robot-Env-Cfg-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": "configs.env_cfg:EnvCfg",
        },
        disable_env_checker=True,
    )
    env_cfg = isaaclab_tasks.utils.parse_cfg.parse_env_cfg(
        "Robot-Env-Cfg-v0",
        device=sim_cfg["device"],
        num_envs=sim_cfg["num_envs"],
        use_fabric=not sim_cfg["disable_fabric"],
    )

    table = None
    scenes = [f for f in os.listdir(scene_dir) if f.endswith(".usd")]
    while table is None:
        # Dynamically create basic scene from USD files
        scene = random.choice(scenes)
        usd_file = os.path.join(scene_dir, scene)
        logging.info("Loading scene from %s", usd_file)
        env_cfg.scene = configs.scene_cfg.set_house_asset(
            env_cfg.scene, os.path.join(scene_dir, usd_file)
        )
        tables = configs.scene_cfg.get_table_assets(
            usd_file, sim_cfg["scene"]["cameras"]
        )
        if len(tables) < 1:
            scenes.remove(scene)
            logging.info("No table found in %s. Trying another scene." % scene)
        else:
            table = random.choice(tables)
            logging.info("Using table %s." % table["name"])

    # Determine the robot pose
    robot_pose = random.choice([a for a in table["anchors"] if a["side"] == "long"])
    # Set up the robot arm
    env_cfg = configs.env_cfg.set_robot(robot, env_cfg, robot_pose)
    # Set up cameras in the scene
    if sim_cfg["enable_cameras"]:
        env_cfg.scene = _set_up_scene_cameras(env_cfg.scene, sim_cfg, robot)

    # Set the light intensity and color
    light_cfg = _get_light_cfg(sim_cfg["lighting"])
    logging.info(
        "Setting light temperature to %d and intensity to %d"
        % (light_cfg["temperature"], light_cfg["intensity"])
    )
    env_cfg.scene = configs.scene_cfg.set_light_asset(env_cfg.scene, **light_cfg)

    # Determine the poses of objects and containers
    object_states = _get_object_states(
        sim_cfg,
        robot_pose,
        table["bbox"],
        object_metadata,
        sim_cfg["robots"][robot]["max_reach_dist"],
    )
    # Dynamically add objects to the scene
    env_cfg.scene = _set_up_scene_objects(env_cfg.scene, object_states["objects"])
    # Dynamically add containers to the scene
    if "containers" in object_states and object_states["containers"]:
        env_cfg.scene = _set_up_scene_containers(
            env_cfg.scene, object_states["containers"]
        )

    # Determine the objects to be used in the task
    objects = []
    if task == "long_horizon":
        objects = [key for key in vars(env_cfg.scene) if key.startswith("object")]
    else:
        objects = ["object"]

    object_sizes = {
        o: get_object_size(
            os.path.basename(getattr(env_cfg.scene, o).spawn.usd_path),
            object_metadata,
            device=sim_cfg["device"],
        )
        for o in objects
    }
    # Modify task-specific parameters
    env_cfg.episode_length_s = sim_cfg["tasks"][task]["episode_length"]
    terimation_args = {
        "goal_position": torch.tensor(
            sim_cfg["robots"][robot]["final_pose"][:3],
            dtype=torch.float32,
            device=sim_cfg["device"],
        ),
        "objects": objects,
        "object_sizes": object_sizes,
    }
    if hasattr(env_cfg.scene, "container"):
        terimation_args["container_size"] = get_object_size(
            os.path.basename(env_cfg.scene.container.spawn.usd_path),
            object_metadata,
            device=sim_cfg["device"],
        )

    env_cfg.terminations = configs.termination_cfg.get_termination_cfg(
        task, terimation_args
    )
    # Return the unique tags of target object and container (if any)
    object_tags = {
        k: helpers.get_object_tags(
            k,
            v,
            robot_pose,
            ["VELOCITY"] if k == "containers" else None,
            sim_cfg["scene"]["objects"]["tag_thresholds"],
        )
        for k, v in object_states.items()
    }
    if task == "long_horizon":
        object_tags["objects"] = ["entire set of objects"]

    logging.debug("Object tags: %s" % object_tags)
    return env_cfg, object_tags, objects, object_sizes


def _set_up_scene_cameras(scene_cfg, sim_cfg, robot):
    import configs.robot_cfg
    import configs.scene_cfg

    # Set up the wrist camera on the robot arm
    scene_cfg = configs.scene_cfg.add_scene_camera(
        scene_cfg,
        "wrist_cam",
        configs.scene_cfg.get_camera_cfg(
            sim_cfg["camera"].copy(), configs.robot_cfg.get_wrist_camera_cfg(robot)
        ),
    )
    # Set up the cameras according to relative position in the config file
    for cam in sim_cfg["scene"]["cameras"]:
        scene_cfg = configs.scene_cfg.add_scene_camera(
            scene_cfg,
            cam["name"],
            configs.scene_cfg.get_camera_cfg(
                sim_cfg["camera"].copy(),
                get_camera_pose(cam),
            ),
        )
    return scene_cfg


def get_camera_pose(cam):
    # scalar_first is not supported in scipy < 1.12.0 (required by Isaac Lab)
    quat = R.from_euler("XYZ", cam["rotation"], degrees=True).as_quat()

    return {
        "prim_path": cam["prim_path"],
        "pos": cam["position"],
        "quat": [quat[3], quat[0], quat[1], quat[2]],  # wxyz
        "convention": "opengl",
    }


def _get_light_cfg(light_cfg):
    light_position = [
        random.randint(*light_cfg["position"]["x"]),
        random.randint(*light_cfg["position"]["y"]),
        random.randint(*light_cfg["position"]["z"]),
    ]
    light_temperature = random.randint(*light_cfg["temperature"])
    light_intensity = random.randint(*light_cfg["intensity"])
    return {
        "position": light_position,
        "temperature": light_temperature,
        "intensity": light_intensity,
    }


def _get_object_states(
    sim_cfg, robot_pose, table_bbox, object_metadata, robot_reach_dist
):
    object_states = {"objects": [], "containers": []}
    # Generate the poses of objects (The first object is the target object)
    object_cfg = sim_cfg["scene"]["objects"]
    object_categories = object_cfg["categories"]
    if not object_categories:
        object_categories = list(set([v["category"] for v in object_metadata.values()]))

    object_candidates = [
        v for v in object_metadata.values() if v["category"] in object_categories
    ]
    object_range_bbox = _get_object_range_bbox(table_bbox)
    for oi in range(object_cfg["n_objects"]):
        _object = random.choice(object_candidates).copy()
        random_orientation = random.random() < object_cfg.get("prob_rnd_quat", 0.5)
        random_static = (
            random.random() < object_cfg.get("prob_static", 0.5) if oi != 0 else False
        )  # The first object is always dynamic
        random_friction = np.random.uniform(*object_cfg.get("friction", [0, 0]))
        random_perturbation = np.random.uniform(*object_cfg.get("perturbation", [0, 0]))
        _state = _get_object_state(
            _get_object_z(object_range_bbox.max[2], _object["size"]),
            robot_pose["pos"],
            object_range_bbox,
            None if random_static else object_cfg.get("moving_speed", None),
            random_friction,
            random_perturbation,
            random_orientation,
        )
        object_states["objects"].append(
            {
                **_object,
                **_state,
                "mass": object_cfg.get("mass", 0.05),
            }
        )

    # Generate the poses of containers (The first container is the target container)
    container_cfg = sim_cfg["scene"]["containers"]
    container_categories = container_cfg["categories"]
    cntr_range_bbox = _get_object_range_bbox(
        table_bbox, robot_pose["pos"], robot_reach_dist
    )
    if not container_categories:
        container_categories = list(
            set([v["category"] for v in object_metadata.values()])
        )

    container_candidates = [
        v for v in object_metadata.values() if v["category"] in container_categories
    ]
    for _ in range(container_cfg["n_containers"]):
        _state = None
        while _state is None and container_candidates:
            _container = random.choice(container_candidates).copy()
            _state = _get_container_state(
                _container["size"],
                cntr_range_bbox,
                random_orientation,
                object_states,
            )
            if _state is None:
                # The container cannot be placed without occlusion
                container_candidates.remove(_container)

        # Remove from the candidates to avoid duplication
        if _state is not None:
            container_candidates.remove(_container)
            object_states["containers"].append(
                {
                    **_container,
                    **_state,
                    "mass": container_cfg.get("mass", 0.1),
                }
            )

    return object_states


def _get_object_range_bbox(table_bbox, robot_position=None, robot_reach_dist=None):
    from pxr import Gf

    object_range_min_0 = table_bbox.min[0] * 3 / 4 + table_bbox.max[0] / 4
    object_range_max_0 = table_bbox.min[0] / 4 + table_bbox.max[0] * 3 / 4
    object_range_min_1 = table_bbox.min[1] * 3 / 4 + table_bbox.max[1] / 4
    object_range_max_1 = table_bbox.min[1] / 4 + table_bbox.max[1] * 3 / 4
    table_z = table_bbox.max[2]
    object_valid_range = Gf.Range3d(
        Gf.Vec3d(object_range_min_0, object_range_min_1, table_z),
        Gf.Vec3d(object_range_max_0, object_range_max_1, table_z),
    )
    if robot_position is None and robot_reach_dist is None:
        return object_valid_range
    else:
        # Consider whether the object is within the robot reach
        robot_reach_bbox = Gf.Range3d(
            Gf.Vec3d(
                robot_position[0] - robot_reach_dist,
                table_bbox.min[1] - robot_reach_dist,
                table_z,
            ),
            Gf.Vec3d(
                robot_position[0] + robot_reach_dist,
                table_bbox.min[1] + robot_reach_dist,
                table_z,
            ),
        )
        return robot_reach_bbox.IntersectWith(object_valid_range)


def _get_object_z(table_z, object_size=None):
    PADDING = 0.02
    return (
        table_z + np.max(object_size) / 2
        if object_size is not None
        else table_z + PADDING
    )


def _get_object_state(
    object_z,
    robot_position,
    object_range_bbox,
    moving_speed,
    friction,
    perturbation,
    random_orientation,
):
    if moving_speed is None:
        object_state = _get_static_object_state(
            object_range_bbox, object_z, random_orientation
        )
    else:
        object_state = _get_dynamic_object_state(
            object_range_bbox,
            object_z,
            moving_speed,
            friction,
            perturbation,
            robot_position,
        )

    return object_state


def _get_static_object_state(object_range_bbox, object_z, random_orientation):
    import configs.object_cfg

    object_position = np.array(
        [
            random.uniform(object_range_bbox.min[0], object_range_bbox.max[0]),
            random.uniform(object_range_bbox.min[1], object_range_bbox.max[1]),
            object_z,
        ]
    )
    object_quat = np.array([1.0, 0.0, 0.0, 0.0])
    if random_orientation:
        object_quat = configs.object_cfg.get_object_init_quat(
            np.random.uniform(-0.1, 0.1, size=3)
        )

    return {"pos": object_position, "quat": object_quat}


def _get_dynamic_object_state(
    object_range_bbox, object_z, moving_speed, friction, perturbation, robot_position
):
    import configs.object_cfg

    object_position = np.array(
        [
            random.uniform(object_range_bbox.min[0], object_range_bbox.max[0]),
            random.uniform(object_range_bbox.min[1], object_range_bbox.max[1]),
            object_z,
        ]
    )
    assert robot_position is not None
    # Generate a random position between the table center and the robot arm
    tbl_ctr = (object_range_bbox.min + object_range_bbox.max) / 2.0
    random_ratio = random.uniform(-0.5, 0.5)
    random_position = tbl_ctr + random_ratio * (robot_position - tbl_ctr)
    random_position[2] = object_z
    # Determine the linear velocity of the object
    assert moving_speed is not None and len(moving_speed) == 2
    object_direction = random_position - object_position
    object_velocity = (
        object_direction
        / np.linalg.norm(object_direction)
        * random.uniform(*moving_speed)
    )
    object_quat = configs.object_cfg.get_object_init_quat(
        object_velocity, perturbation=perturbation
    )

    return {
        "pos": object_position,
        "quat": object_quat,
        "lin_vel": object_velocity,
        "friction": friction,
    }


def _get_container_state(
    container_size, object_range_bbox, random_orientation, existing_objects
):
    import configs.object_cfg

    N_MAX_TRIES = 100
    n_tries = 0
    container_position = None
    while container_position is None and n_tries < N_MAX_TRIES:
        n_tries += 1
        container_position = np.array(
            [
                random.uniform(object_range_bbox.min[0], object_range_bbox.max[0]),
                random.uniform(object_range_bbox.min[1], object_range_bbox.max[1]),
                _get_object_z(object_range_bbox.max[2], container_size),
            ]
        )
        container_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if random_orientation:
            container_quat = configs.object_cfg.get_object_init_quat(
                np.random.uniform(-0.1, 0.1, size=3), upright=True
            )
        # Check whether the container is occluding with existing objects/containers
        for eo in existing_objects["objects"] + existing_objects["containers"]:
            if _is_bbox_overlap(
                _get_object_bbox(container_position, container_size, container_quat),
                _get_object_bbox(
                    eo["pos"], eo["size"], eo["quat"], eo.get("lin_vel", None)
                ),
            ):
                container_position = None
                break

    if container_position is not None:
        return {"pos": container_position, "quat": container_quat}
    else:
        return None


def _get_object_bbox(position, size, quat, lin_vel=None):
    # TODO: Consider the velocity of the object
    dx, dy, dz = size / 2.0
    corners = np.array(
        [
            [-dx, -dy, -dz],
            [dx, -dy, -dz],
            [dx, dy, -dz],
            [-dx, dy, -dz],
            [-dx, -dy, dz],
            [dx, -dy, dz],
            [dx, dy, dz],
            [-dx, dy, dz],
        ]
    )
    rotated_corners = R.from_quat(quat[[1, 2, 3, 0]]).apply(corners)

    bbox = rotated_corners + position
    return Polygon(bbox[np.argsort(bbox[:, 2])[:4], :2])


def _is_bbox_overlap(bbox1, bbox2):
    return bbox1.intersects(bbox2)


def _set_up_scene_objects(scene_cfg, object_states):
    import configs.object_cfg
    import configs.scene_cfg

    assert object_states, "No object states provided."
    target_object = object_states[0]
    other_objects = object_states[1:]

    # Set the target object (the object to be manipulated)
    logging.info(
        "Using target object: %s" % os.path.basename(target_object["file_path"])
    )
    scene_cfg = configs.scene_cfg.set_target_object(
        scene_cfg,
        configs.object_cfg.get_object_cfg(
            "/Object",
            target_object,
            configs.object_cfg.get_spawner_cfg(
                file_path=target_object["file_path"],
                mass=target_object["mass"],
                friction=target_object.get("friction", 0.0),
                semantic_tags=[("class", "OBJECT_MAIN")],
            ),
        ),
    )
    # Add more objects to the scene
    for i, o in enumerate(other_objects):
        logging.info("Using BG object: %s" % os.path.basename(o["file_path"]))
        scene_cfg = configs.scene_cfg.add_object(
            scene_cfg,
            "object%02d" % (i + 1),
            configs.object_cfg.get_object_cfg(
                "/Object%02d" % (i + 1),
                o,
                configs.object_cfg.get_spawner_cfg(
                    file_path=o["file_path"],
                    mass=o["mass"],
                    friction=o.get("friction", 0.0),
                    semantic_tags=[("class", "OBJECT%02d" % (i + 1))],
                ),
            ),
        )
    return scene_cfg


def _set_up_scene_containers(scene_cfg, container_states):
    import configs.object_cfg
    import configs.scene_cfg

    assert container_states, "No container states provided."
    for i, o in enumerate(container_states):
        logging.info("Using container object: %s" % os.path.basename(o["file_path"]))
        cntr_class = "CONTAINER%02d" % (i + 1) if i != 0 else "CONTAINER_MAIN"
        scene_cfg = configs.scene_cfg.add_object(
            scene_cfg,
            "container%02d" % i if i != 0 else "container",
            configs.object_cfg.get_object_cfg(
                "/Container%02d" % i if i != 0 else "/Container",
                o,
                configs.object_cfg.get_spawner_cfg(
                    file_path=o["file_path"],
                    mass=o["mass"],
                    semantic_tags=[("class", cntr_class)],
                ),
            ),
        )
    return scene_cfg


def get_object_size(object_name, object_metadata, device="cpu"):
    if object_name in object_metadata:
        object_size = object_metadata[object_name]["size"]
    else:
        object_size = [0.05, 0.05, 0.05]
        logging.warning(
            "Object size for %s not found. Using default size %s."
            % (object_name, object_size)
        )

    return _get_tensor(object_size, device=device, unsqueeze=True)


def get_state_machine(task_cfg, robot_cfg, sm_args={}):
    state_machine = _get_class(task_cfg["sm"])
    for k, v in robot_cfg.items():
        sm_args[k] = _get_tensor(v, sm_args.get("device"))

    return state_machine(**sm_args)


def _get_tensor(array, device="cpu", unsqueeze=True):
    if not isinstance(array, list) and not isinstance(array, np.ndarray):
        return array

    tensor = torch.tensor(array, dtype=torch.float32, device=device)
    if unsqueeze:
        tensor = tensor.unsqueeze(0)

    return tensor


def _get_class(class_path):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def set_object_material(target_object, n_envs=1):
    materials = target_object.root_physx_view.get_material_properties()
    materials[..., 0] = 0.9  # Static friction.
    materials[..., 1] = 1.0  # Dynamic friction.
    materials[..., 2] = 0.0  # Restitution
    target_object.root_physx_view.set_material_properties(
        materials, torch.arange(n_envs)
    )


def get_curr_state(
    ee_state,
    robot_joint_pos=None,
    object_state=None,
    object_size=None,
    container_state=None,
    container_size=None,
    env_origins=None,
    robot_quat=None,
    device="cpu",
):
    # _get_merged_object_state = lambda object_state, key: torch.cat(
    #     [getattr(os, key)[i:i+1] for i, os in enumerate(object_state)], dim=0
    # )
    curr_state = {}
    if ee_state is not None:
        curr_state["end_effector"] = {
            "pos": helpers.get_robot_relative_position(
                ee_state.target_pos_w[..., 0, :] - env_origins, robot_quat
            ),
            "quat": _get_robot_relative_quaternion(
                ee_state.target_quat_w[..., 0, :], robot_quat
            ),
        }
    if robot_joint_pos is not None:
        curr_state["joints"] = robot_joint_pos
    if object_state is not None:
        # object_root_pos_w = _get_merged_object_state(object_state, "root_pos_w")
        # object_root_quat_w = _get_merged_object_state(object_state, "root_quat_w")
        # object_root_lin_vel_w = _get_merged_object_state(object_state, "root_lin_vel_w")
        curr_state["object"] = {
            "pos": helpers.get_robot_relative_position(
                object_state.root_pos_w - env_origins, robot_quat
            ),
            "quat": _get_robot_relative_quaternion(object_state.root_quat_w, robot_quat),
            "velocity": helpers.get_robot_relative_position(
                object_state.root_lin_vel_w, robot_quat
            ),
        }
        if object_size is not None:
            curr_state["object"]["size"] = helpers.get_object_relative_bbox(
                object_size, object_state.root_quat_w, robot_quat
            )
    if container_state is not None:
        curr_state["container"] = {
            "pos": helpers.get_robot_relative_position(
                container_state.root_pos_w - env_origins, robot_quat
            ),
            "quat": _get_robot_relative_quaternion(
                container_state.root_quat_w, robot_quat
            ),
        }
    if container_size is not None:
        if "container" not in curr_state:
            curr_state["container"] = {}

        curr_state["container"]["size"] = helpers.get_object_relative_bbox(
            container_size, container_state.root_quat_w, robot_quat
        )
    if device == "cpu":
        for csk, csv in curr_state.items():
            if isinstance(csv, dict):
                for k, v in csv.items():
                    if isinstance(v, torch.Tensor):
                        curr_state[csk][k] = v.cpu().numpy()
            elif isinstance(csv, torch.Tensor):
                curr_state[csk] = csv.cpu().numpy()

    return curr_state


def _get_robot_relative_quaternion(w_quat, robot_quat):
    from isaaclab.utils.math import quat_inv, quat_mul

    return quat_mul(quat_inv(robot_quat), w_quat)


def get_camera_views(sensors, views=["rgb"]):
    # NOTE: import isaaclab.utils does not work
    from isaaclab.utils import convert_dict_to_backend

    cam_views = {}
    for name, sensor in sensors.items():
        if type(sensor).__name__ == "Camera":
            cam_views[name] = convert_dict_to_backend(
                {k: v for k, v in sensor.data.output.items()}, backend="numpy"
            )
            # Make semantic segmentation consistent in all views
            if "semantic_segmentation" in cam_views[name]:
                cam_views[name]["seg"] = _get_semantic_segmentation(
                    cam_views[name]["semantic_segmentation"],
                    [
                        i["semantic_segmentation"]["idToLabels"]
                        for i in sensor.data.info
                    ],
                )
                # Remove the original semantic segmentation (with New Key: "seg")
                del cam_views[name]["semantic_segmentation"]

            cam_views[name] = {k: v for k, v in cam_views[name].items() if k in views}

    return cam_views


def _get_semantic_segmentation(rgba_seg_maps, semantic_tags):
    known_tags = helpers.get_semantic_tags()
    seg_maps = np.zeros_like(rgba_seg_maps[..., :1], dtype=np.uint8)

    # Iterate over each image (since the tags may not be the same for each image)
    for si, st in enumerate(semantic_tags):
        for color, tag in st.items():
            tag_name = tag["class"].upper()
            if tag_name in ["BACKGROUND", "UNLABELLED"]:
                continue
            elif tag_name not in known_tags.keys():
                logging.warning("Unknown semantic tag %s.", tag_name)
                continue

            # Convert the color string to a tuple (Unbelievable string here!)
            mask = np.all(rgba_seg_maps[si] == ast.literal_eval(color), axis=-1)
            seg_maps[si][mask] = known_tags[tag_name]

    return seg_maps


def get_next_object(scene_objects, scene, env_idx=None):
    next_object = []  # next object index for each environment
    n_envs = len(scene_objects)
    for i in range(n_envs):
        if len(scene_objects[i]) == 0 or (env_idx is not None and i != env_idx):
            next_object.append(None)
            continue

        objects = scene_objects[i]
        objects_velocity = torch.cat(
            [scene[o].data.root_lin_vel_w[i : i + 1] for o in objects], dim=0
        )
        speed = torch.norm(objects_velocity, dim=-1)
        fastest_index = torch.argmax(speed, dim=0).item()
        next_object.append(fastest_index)

    return next_object if env_idx is None else next_object[env_idx]


def get_env_states(states, n_envs=1):
    KEYS = {
        "sm_state": ("next_state", "sm_state"),
        "ee_pos": ("curr_state", "end_effector", "pos"),
        "ee_quat": ("curr_state", "end_effector", "quat"),
        "joints": ("curr_state", "joints"),
        "object_pos": ("curr_state", "object", "pos"),
        "object_quat": ("curr_state", "object", "quat"),
        "object_vel": ("curr_state", "object", "velocity"),
        "action": ("next_state", "action"),
    }
    if not states:
        return []

    env_states = [{} for _ in range(n_envs)]
    for es in states:
        # es = {"cam_views": dict, "curr_state": dict, "next_state": dict, "is_done": Tensor}
        for eid in range(n_envs):
            # Predefined keys
            for k, v in KEYS.items():
                value = None
                if v[0] not in es:
                    continue
                # Retrieve the value based on the key path
                if len(v) == 1:
                    value = es[v[0]][eid].cpu().numpy()
                elif len(v) == 2 and v[1] in es[v[0]]:
                    value = es[v[0]][v[1]][eid].cpu().numpy()
                elif len(v) == 3 and v[1] in es[v[0]] and v[2] in es[v[0]][v[1]]:
                    value = es[v[0]][v[1]][v[2]][eid].cpu().numpy()

                if value is None:
                    continue
                if k not in env_states[eid]:
                    env_states[eid][k] = []

                env_states[eid][k].append(value)

            # Camera views
            if "cam_views" in es:
                for cam, imgs in es["cam_views"].items():
                    for k, v in imgs.items():
                        cam_key = "%s_%s" % (cam, k)
                        if cam_key not in env_states[eid]:
                            env_states[eid][cam_key] = []

                        env_states[eid][cam_key].append(v[eid])

    return env_states


def simulate(sim_cfg, task, robot, scene_dir, object_metadata, seed):
    import configs.termination_cfg

    # Create a new environment
    env_cfg, object_tags, objects, object_sizes = get_env_cfg(
        sim_cfg,
        task,
        robot,
        object_metadata,
        scene_dir,
    )
    # Check whether the object tags are empty
    if not object_tags["objects"]:
        raise ValueError("No object tags found. Skipping simulation.")
    if (
        sim_cfg["scene"]["containers"]["n_containers"] > 0
        and not object_tags["containers"]
    ):
        raise ValueError("No container tags found. Skipping simulation.")

    env = gym.make("Robot-Env-Cfg-v0", cfg=env_cfg, seed=seed)
    # Reset environment at start
    env.reset(seed=seed)

    # Determine the object size (without transformation)
    if "container" in env.unwrapped.scene.keys():
        container_data = env.unwrapped.scene["container"].data
        container_size = get_object_size(
            os.path.basename(env_cfg.scene.container.spawn.usd_path),
            object_metadata,
            env.unwrapped.device,
        )
    else:
        container_data, container_size = None, None

    # Enable Path Tracing
    if sim_cfg["enable_cameras"] and sim_cfg["path_tracing"]:
        import omni.replicator.core as rep

        rep.settings.set_render_pathtraced()

    # Initialize the state machine
    assert task in sim_cfg["tasks"], "Unknown task: %s." % task
    assert robot in sim_cfg["robots"], "Unknown robot: %s." % robot
    state_machine = get_state_machine(
        sim_cfg["tasks"][task],
        sim_cfg["robots"][robot],
        {
            "dt": env_cfg.sim.dt * env_cfg.decimation,
            "num_envs": env.unwrapped.num_envs,
            "device": env.unwrapped.device,
        },
    )

    # Set the rotation, height, and physical material of the object
    set_object_material(
        env.unwrapped.scene["object"],
        n_envs=env.unwrapped.num_envs,
    )

    # Simulation loop
    env_states = []
    term_mgr = env.env.termination_manager
    done_term = configs.termination_cfg.get_done_term(term_mgr.active_terms)

    scene_objects = [copy.deepcopy(objects) for _ in range(env.unwrapped.num_envs)]
    curr_object_idx = get_next_object(scene_objects, env.unwrapped.scene)
    while not term_mgr.dones.all():
        # Add an option to disable the state machine to accelerate the simulation
        if sim_cfg["disable_sm"]:
            env.step(torch.from_numpy(env.action_space.sample()))
            continue

        curr_object = [objects[coi] for coi in curr_object_idx]
        # Determine the current object to manipulate
        curr_state = get_curr_state(
            env.unwrapped.scene["ee_frame"].data,
            env.unwrapped.scene.state["articulation"]["robot"]["joint_position"],
            [env.unwrapped.scene[co].data for co in curr_object][0],  # TODO: remove [0]
            [object_sizes[co] for co in curr_object][0],  # TODO: remove [0]
            container_data,
            container_size,
            env.unwrapped.scene["robot"].data.root_pos_w,
            env.unwrapped.scene["robot"].data.root_quat_w,
            env.unwrapped.device,
        )
        if task == "long_horizon":
            object_placed = helpers.is_object_placed(
                curr_state["object"]["pos"],
                curr_state["object"]["size"],
                curr_state["container"]["pos"],
                curr_state["container"]["size"],
            )
            if object_placed.any():
                for env_idx, op in enumerate(object_placed):
                    if not op or len(scene_objects[env_idx]) < 2:
                        continue

                    scene_objects[env_idx].remove(curr_object[env_idx])
                    curr_object_idx[env_idx] = get_next_object(
                        scene_objects, env.unwrapped.scene, env_idx
                    )
                    logging.debug(
                        "[Env%02d] Object %s placed. Next object: %s."
                        % (
                            env_idx,
                            curr_object[env_idx],
                            scene_objects[env_idx][curr_object_idx[env_idx]],
                        )
                    )

        # NOTE: state format in xyz, quat (wxyz), gripper (-1/1)
        next_state = state_machine.compute(curr_state)

        cam_views = get_camera_views(
            env.unwrapped.scene.sensors, ["rgb", "depth", "seg"]
        )
        env.step(next_state["action"])
        env_states.append(
            {
                "cam_views": cam_views,
                "curr_state": curr_state,
                "next_state": next_state,
                "curr_obj_idx": curr_object_idx,
                "is_done": term_mgr.dones,
            }
        )

    env_states = get_env_states(env_states, env.unwrapped.num_envs)
    env.close()
    # Ignore the simulation if the task is not finished
    # If in debug mode, save all simulation data even if the task is not finishedq
    is_done = term_mgr.get_term(done_term)
    return (
        env_cfg,
        object_tags,
        [
            es
            for env_id, es in enumerate(env_states)
            if is_done[env_id].item() or sim_cfg["debug"]
        ],
    )


def is_object_stopped(scene_cfg, object_velocity, n_steps=25):
    init_speed = np.linalg.norm(scene_cfg["object"]["init_state"]["lin_vel"])
    # The object is stopped within n_steps
    for i in range(min(n_steps, len(object_velocity))):
        if init_speed > 1e-2 and np.linalg.norm(object_velocity[i]) < 1e-2:
            return True

    return False


def is_object_direction_changed(scene_cfg, object_velocity, n_steps=25):
    robot_quat = np.array(scene_cfg["robot"]["init_state"]["rot"])[[1, 2, 3, 0]]
    init_velocity = scene_cfg["object"]["init_state"]["lin_vel"]
    init_dir_idx = helpers.get_direction_index(init_velocity, robot_quat)
    # The object velocity is too different from the initial velocity
    for i in range(min(n_steps, len(object_velocity))):
        dir_idx = helpers.get_direction_index(object_velocity[i])
        if dir_idx != init_dir_idx:
            return True

    return False


def get_episode_name(task, robot, seed, scene_cfg):
    n_objects = len(
        [
            v["class_type"]
            for v in scene_cfg.values()
            if isinstance(v, dict)
            and v["class_type"]
            == "isaaclab.assets.rigid_object.rigid_object:RigidObject"
        ]
    )
    object_vel = np.linalg.norm(scene_cfg["object"]["init_state"]["lin_vel"])
    object_type = (
        os.path.basename(scene_cfg["object"]["spawn"]["usd_path"][:-4])
        if "usd_path" in scene_cfg["object"]["spawn"]
        else "cylinder"  # default object type
    )
    random_suffix = str(uuid.uuid4())[-4:]

    # Generate a unique name for the episode
    return "%s_%s_%s%s_O%02d_%08d_%s" % (
        task,
        robot,
        object_type,
        "d" if object_vel > 1e-3 else "s",
        n_objects,
        seed,
        random_suffix,
    )


def get_object_without_numpy(obj):
    if obj is None:
        return obj
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {k: get_object_without_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [get_object_without_numpy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(get_object_without_numpy(item) for item in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.float64)):
        return obj.item()
    else:
        # logging.warning("Unknown data type: %s" % type(obj))
        return str(obj)


def get_frames(
    env_state, state_keys=["sm_state", "ee_pos", "object_pos", "object_vel"]
):
    MAX_DEPTH = 25

    cam_name = None
    cam_frames = {}
    for st_key, frames in env_state.items():
        cam_idx = st_key.find("_cam_")
        if cam_idx == -1:
            continue

        cam_name = st_key[:cam_idx]
        img_name = st_key[cam_idx + 5 :]
        if cam_name not in cam_frames:
            cam_frames[cam_name] = {}
        if img_name not in cam_frames[cam_name]:
            cam_frames[cam_name][img_name] = []

        for frame in frames:
            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=-1)
            elif frame.ndim == 3:
                if frame.shape[-1] == 1:
                    frame = np.repeat(frame, 3, axis=-1)
                elif frame.shape[-1] >= 3:
                    frame = frame[:, :, :3]
            else:
                raise ValueError("Unknown camera data shape: %s" % (frame.shape,))

            # Normalize the depth image to 0-255
            if img_name in ["depth", "distance_to_image_plane", "distance_to_camera"]:
                frame = np.clip(frame, 0, MAX_DEPTH)
                frame = (frame / np.max(frame) * 255).astype(np.uint8)
            if img_name in ["seg", "semantic_segmentation"]:
                # Assign a color to each semantic class
                frame = helpers.get_semantic_map(frame[..., 0])

            cam_frames[cam_name][img_name].append(frame)

    if cam_name is None:
        raise ValueError("No camera frames found in the environment state.")

    n_frames = len(cam_frames[cam_name][img_name])
    frames = [[[] for _ in range(len(cam_frames))] for _ in range(n_frames)]
    for cam_idx, cam_imgs in enumerate(cam_frames.values()):
        for img in cam_imgs.values():
            for frame_idx in range(n_frames):
                frames[frame_idx][cam_idx].append(img[frame_idx])

    for frame_idx in range(n_frames):
        frame = np.concatenate(
            [np.concatenate(r, axis=0) for r in frames[frame_idx]], axis=1
        )
        if state_keys:
            frame = _print_state_on_frame(
                frame,
                {k: env_state[k][frame_idx] for k in state_keys if k in env_state},
            )

        frames[frame_idx] = frame

    return frames


def _print_state_on_frame(frame, state):
    TEXT_MARGIN = 10
    TEXT_SCALE = 0.5
    TEXT_THICKNESS = 1
    TEXT_COLOR = (255, 255, 255)
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

    lines = _get_state_text(state).split("\n")
    _, img_width = frame.shape[:2]
    # Print the text on the image
    y = TEXT_MARGIN
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(
            line, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS
        )
        x = img_width - text_width - TEXT_MARGIN
        frame = cv2.putText(
            np.ascontiguousarray(frame),
            line,
            (x, y + text_height),
            TEXT_FONT,
            TEXT_SCALE,
            TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )
        y += text_height + TEXT_MARGIN

    return frame


def _get_state_text(state):
    text = ""
    for k, v in state.items():
        k = k.replace("_", " ").title()
        if isinstance(v, (int, np.int32, np.int64)) or v.ndim == 0:
            text += "%s: %d\n" % (k, v)
        elif isinstance(v, np.ndarray):
            # Convert all quaternions to Euler angles
            if k.find("Quat") != -1:
                k = k.replace("Quat", "Rot")
                v = R.from_quat(v).as_euler("xyz", degrees=True)

            text += "%s: " % k
            text += " ".join(["%.3f" % i for i in v]) + "\n"
        else:
            raise ValueError("Unknown State Value Type: %s" % (type(v),))

    return text


def dump_video(frames, output_path, fps=24):
    if len(frames) == 0:
        return

    imageio.v3.imwrite(
        str(output_path),
        frames,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    )


def main(args):
    with open(args.sim_cfg_file) as fp:
        sim_cfg = yaml.load(fp, Loader=yaml.FullLoader)

    sim_cfg.update(
        {
            "debug": args.debug,
            "device": args.device,
            "disable_fabric": args.disable_fabric,
            "disable_sm": args.disable_sm,
            "enable_cameras": args.enable_cameras,
            "num_envs": args.num_envs,
            "path_tracing": args.path_tracing,
        }
    )
    # Get metadata for the objects (size, description, orientation)
    object_categories = sim_cfg["scene"]["objects"]["categories"]
    container_categories = sim_cfg["scene"]["containers"].get("categories", [])
    object_metadata = get_object_metadata(
        args.object_dir, object_categories + container_categories
    )
    # Perform simulations in the environment
    n_simulations = 0
    seed = args.seed if args.seed is not None else random.randint(0, 65535)
    while n_simulations < args.n_simulations:
        logging.info("Running simulation with seed: %d" % seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        try:
            env_cfg, object_tags, env_states = simulate(
                sim_cfg,
                args.task,
                args.robot,
                args.scene_dir,
                object_metadata,
                seed,
            )
        except Exception as ex:
            logging.exception(ex)
            seed += 1
            n_simulations += 1
            continue

        # Save the simulation data
        for es in env_states:
            # Check whether the initial state is changed
            # 1) The object is stopped within a few steps due to collision
            # 2) The initial object velocity is changed due to collision
            env_cfg = env_cfg.to_dict()
            if is_object_stopped(env_cfg["scene"], es["object_vel"]):
                continue
            if is_object_direction_changed(env_cfg["scene"], es["object_vel"]):
                # Remove direction tags
                object_tags["objects"] = [
                    t for t in object_tags["objects"] if not t.endswith("direction")
                ]

            episode_name = get_episode_name(
                args.task, args.robot, seed, env_cfg["scene"]
            )
            logging.info(
                "Saving episode %s with %d frames."
                % (episode_name, len(es["sm_state"]))
            )
            if args.save:
                with open(
                    os.path.join(args.output_dir, "%s.json" % episode_name), "w"
                ) as fp:
                    env_cfg["seed"] = seed
                    env_cfg["instruction"] = {"task": args.task, **object_tags}
                    json.dump(get_object_without_numpy(env_cfg), fp, indent=2)

                with h5py.File(
                    os.path.join(args.output_dir, "%s.h5" % episode_name), "w"
                ) as fp:
                    for k, v in es.items():
                        fp.create_dataset(k, data=v, compression="gzip")

            if args.debug and args.enable_cameras:
                dump_video(
                    get_frames(es),
                    os.path.join(args.output_dir, "%s.mp4" % episode_name),
                )
                logging.debug(object_tags)

        # Increment the seed for the next simulation
        seed += 1
        n_simulations += 1


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    SHARED_PARAMETERS = ["num_envs", "save"]

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
    AppLauncher.add_app_launcher_args(parser)
    isaaclab_args, script_args = parser.parse_known_args()

    # Arguments for the script
    parser.add_argument("--robot", default="franka")
    parser.add_argument(
        "--scene_dir", default=os.path.join(PROJECT_HOME, os.pardir, "scenes")
    )
    parser.add_argument(
        "--object_dir", default=os.path.join(PROJECT_HOME, os.pardir, "objects")
    )
    parser.add_argument(
        "-o", "--output_dir", default=os.path.join(PROJECT_HOME, os.pardir, "datasets")
    )
    parser.add_argument("--task", default="pick")
    parser.add_argument(
        "-c",
        "--sim_cfg_file",
        default=os.path.join(PROJECT_HOME, "simulations", "configs", "sim_cfg.yaml"),
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--disable_sm", action="store_true", default=False)
    parser.add_argument("--path_tracing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-n", "--n_simulations", type=int, default=10_000)
    args = parser.parse_args(script_args)
    # Copy the shared parameters from isaaclab_args to args
    for sp in SHARED_PARAMETERS:
        if sp in isaaclab_args:
            setattr(args, sp, getattr(isaaclab_args, sp))

    app_launcher = AppLauncher(isaaclab_args)
    # Pass "enable_cameras" to this script
    # Ref: https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/app/app_launcher.html
    args.enable_cameras = app_launcher._enable_cameras
    if not args.enable_cameras:
        logging.warning(
            "Cameras are disabled. No images will be produced during simulation."
        )
        answer = input("Do you want to continue? (y/N) ").strip().lower()
        if answer != "y":
            exit(0)

    main(args)
    app_launcher.app.close()
