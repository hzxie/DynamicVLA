# -*- coding: utf-8 -*-
#
# @File:   simulate.py
# @Author: Haozhe Xie
# @Date:   2025-03-22 20:59:36
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-06-25 14:55:39
# @Email:  root@haozhexie.com

import argparse
import ast
import json
import logging
import os
import random
import uuid

import cv2
import gymnasium as gym
import h5py
import isaaclab.app
import numpy as np
import scipy.spatial.transform
import torch
import yaml

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


def get_env_cfg(scene_dir, object_dir, container_dir, sim_cfg, robot):
    # The following packages MUST be imported after the simulation app is created
    import configs.env_cfg
    import configs.scene_cfg
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
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )

    table = None
    while table is None:
        # Dynamically create basic scene from USD files
        usd_file = os.path.join(
            scene_dir,
            random.choice([f for f in os.listdir(scene_dir) if f.endswith(".usd")]),
        )
        logging.info("Loading scene from %s", usd_file)
        env_cfg.scene = configs.scene_cfg.set_house_asset(
            env_cfg.scene, os.path.join(scene_dir, usd_file)
        )
        tables = configs.scene_cfg.get_table_assets(usd_file, sim_cfg["scene"]["table"])
        if len(tables) != 0:
            table = random.choice(tables)

    # Determine the robot pose
    robot_pose = random.choice([a for a in table["anchors"] if a["side"] == "long"])
    side_cam_pose = random.choice([a for a in table["anchors"] if a["side"] == "short"])
    # Set up the robot arm
    env_cfg = configs.env_cfg.set_robot(robot, env_cfg, robot_pose)
    # Set up cameras in the scene
    env_cfg.scene = _set_up_scene_cameras(
        env_cfg.scene, sim_cfg, robot, robot_pose, side_cam_pose, table["bbox"]
    )

    # Set the light intensity and color
    light_cfg = _get_light_cfg(sim_cfg["lighting"])
    logging.info(
        "Setting light temperature to %d and intensity to %d"
        % (light_cfg["temperature"], light_cfg["intensity"])
    )
    env_cfg.scene = configs.scene_cfg.set_light_asset(env_cfg.scene, **light_cfg)

    # Dynamically add objects to scene
    env_cfg.scene = _set_up_scene_objects(
        env_cfg.scene,
        sim_cfg,
        robot_pose,
        table["bbox"],
        object_dir,
    )
    # Set up the container
    if os.path.exists(container_dir):
        env_cfg.scene = _set_up_scene_container(
            env_cfg.scene, table["bbox"], container_dir
        )

    return env_cfg


def _set_up_scene_cameras(
    scene_cfg, sim_cfg, robot, robot_pose, side_cam_pose, table_bbox
):
    import configs.robot_cfg
    import configs.scene_cfg

    # Set up the top-view camera
    scene_cfg = configs.scene_cfg.add_scene_camera(
        scene_cfg,
        "top_cam",
        configs.scene_cfg.get_camera_cfg(
            sim_cfg["camera"].copy(),
            _get_top_camera_relative_pose(robot_pose, table_bbox),
        ),
    )
    # Set up the third-view camera
    scene_cfg = configs.scene_cfg.add_scene_camera(
        scene_cfg,
        "side_cam",
        configs.scene_cfg.get_camera_cfg(
            sim_cfg["camera"].copy(),
            _get_side_camera_relative_pose(side_cam_pose, robot_pose, table_bbox),
        ),
    )
    # Set up the gripper camera on the robot arm
    scene_cfg = configs.scene_cfg.add_scene_camera(
        scene_cfg,
        "gripper_cam",
        configs.scene_cfg.get_camera_cfg(
            sim_cfg["camera"].copy(), configs.robot_cfg.get_gripper_camera_cfg(robot)
        ),
    )
    return scene_cfg


def _get_top_camera_relative_pose(robot_pose, table_bbox):
    # import configs.scene_cfg
    robot_quat = robot_pose["quat"]
    inv_r = scipy.spatial.transform.Rotation.from_quat(
        [robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]]
    ).inv()

    # Plan A: Look at the table center
    tbl_center = (table_bbox.min + table_bbox.max) / 2.0
    tbl_center[2] = table_bbox.max[2] + 1
    cx, cy, cz = inv_r.apply(np.array(tbl_center) - robot_pose["pos"])

    return {
        "prim_path": "/Robot/TopCamera",
        "pos": [cx, cy, cz],
        "quat": [0.7071068, 0, 0, -0.7071068],
        "convention": "opengl",
    }
    # Plan B: Look at the robot base
    # dx, dy, _ = inv_r.apply((np.array(tbl_center) - robot_pose["pos"]) * 1.5)
    # dz = table_bbox.max[2] + 0.25
    # cam_quat = configs.scene_cfg.get_quat_from_look_at([dx, dy, dz], [0.0, 0.0, 0.0])
    # return {
    #     "prim_path": "/Robot/TopCamera",
    #     "pos": [dx, dy, dz],
    #     "quat": cam_quat,
    #     "convention": "world",
    # }


def _get_side_camera_relative_pose(cam_pose, robot_pose, table_bbox):
    import configs.scene_cfg

    robot_quat = robot_pose["quat"]
    inv_r = scipy.spatial.transform.Rotation.from_quat(
        [robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]]
    ).inv()
    # Relative position of the camera to the robot
    dx, dy, dz = inv_r.apply(cam_pose["pos"] - robot_pose["pos"])

    # Relative rotation of the camera to the robot
    tbl_center = (table_bbox.min + table_bbox.max) / 2.0
    cx, cy, cz = inv_r.apply(np.array(tbl_center) - robot_pose["pos"])
    # cz = -robot_pose["pos"][2]
    cam_quat = configs.scene_cfg.get_quat_from_look_at([dx, dy, dz], [cx, cy, cz])

    # Determine the height of the camera (1/5 of the longer side of the table)
    tbl_size = table_bbox.max - table_bbox.min
    dz += max(tbl_size[:2]) / 5

    return {
        "prim_path": "/Robot/SideCamera",
        "pos": [dx, dy, dz],  # Move the camera above the table top
        "quat": cam_quat,
        "convention": "world",
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


def _set_up_scene_objects(scene_cfg, sim_cfg, robot_pose, table_bbox, object_dir):
    import configs.scene_cfg

    target_category = random.choice(os.listdir(object_dir))
    target_candidates = [
        f
        for f in os.listdir(os.path.join(object_dir, target_category))
        if f.endswith(".usd")
    ]
    target_object = random.choice(target_candidates)
    logging.info("Using target object: %s" % target_object)
    scene_cfg = configs.scene_cfg.set_target_object(
        scene_cfg,
        _get_object_cfg(
            table_bbox,
            file_path=os.path.join(object_dir, target_category, target_object),
            robot_pos=robot_pose["pos"],
            moving_time=sim_cfg["scene"]["object"]["moving_time"],
            semantic_tags=[("class", "OBJECT_MAIN")],
        ),
    )
    # TODO: Add more objects to the scene
    # scene_cfg = configs.scene_cfg.add_object_to_scene(scene_cfg)
    return scene_cfg


def _get_object_cfg(
    table_bbox,
    file_path=None,
    robot_pos=None,
    static=False,
    moving_time=[1, 2],
    semantic_tags=None,
):
    import configs.object_cfg

    PADDING = 0.02
    object_cfg = {}
    tbl_z = table_bbox.max[2] + PADDING
    if static:
        object_range_min_0 = table_bbox.min[0] * 3 / 4 + table_bbox.max[0] / 4
        object_range_max_0 = table_bbox.min[0] / 4 + table_bbox.max[0] * 3 / 4
        object_range_min_1 = table_bbox.min[1] * 3 / 4 + table_bbox.max[1] / 4
        object_range_max_1 = table_bbox.min[1] / 4 + table_bbox.max[1] * 3 / 4
        object_cfg["pos"] = np.array(
            [
                random.uniform(object_range_min_0, object_range_max_0),
                random.uniform(object_range_min_1, object_range_max_1),
                tbl_z,
            ]
        )
        # 50% chance to set a random orientation. Otherwise, use the default orientation.
        if random.random() < 0.5:
            object_cfg["quat"] = configs.object_cfg.get_object_init_quat(
                np.random.uniform(-0.1, 0.1, size=3)
            )
    else:
        object_cfg["pos"] = np.array(
            [
                random.uniform(
                    table_bbox.min[0] + PADDING, table_bbox.max[0] - PADDING
                ),
                random.uniform(
                    table_bbox.min[1] + PADDING, table_bbox.max[1] - PADDING
                ),
                tbl_z,
            ]
        )
        assert (
            robot_pos is not None
        ), "Robot position must be provided for dynamic objects."
        # Generate a random position between the table center and the robot arm
        tbl_ctr = (table_bbox.min + table_bbox.max) / 2.0
        rnd_rto = random.uniform(-0.5, 0.5)
        rnd_pos = tbl_ctr + rnd_rto * (robot_pos - tbl_ctr)
        rnd_pos[2] = tbl_z
        # Determine the linear velocity of the object
        rnd_tme = random.uniform(*moving_time)
        object_cfg["lin_vel"] = (rnd_pos - object_cfg["pos"]) / rnd_tme
        object_cfg["quat"] = configs.object_cfg.get_object_init_quat(
            object_cfg["lin_vel"]
        )

    return configs.object_cfg.get_object_cfg(
        "/Object",
        object_cfg,
        configs.object_cfg.get_spawner_cfg(
            file_path=file_path, semantic_tags=semantic_tags
        ),
    )


def _set_up_scene_container(scene_cfg, table_bbox, container_dir):
    import configs.object_cfg

    container_candidates = [f for f in os.listdir(container_dir) if f.endswith(".usd")]
    target_container = random.choice(container_candidates)
    container_cfg = _get_container_cfg(table_bbox)

    scene_cfg = configs.scene_cfg.add_object_to_scene(
        scene_cfg,
        "container",
        configs.object_cfg.get_object_cfg(
            "/Container",
            container_cfg,
            configs.object_cfg.get_spawner_cfg(
                file_path=os.path.join(container_dir, target_container),
                semantic_tags=[("class", "CONTAINER")],
            ),
        ),
    )
    return scene_cfg


def _get_container_cfg(table_bbox):
    object_cfg = {}
    tbl_z = table_bbox.max[2] + 0.1

    object_range_min_0 = table_bbox.min[0] * 3 / 4 + table_bbox.max[0] / 4
    object_range_max_0 = table_bbox.min[0] / 4 + table_bbox.max[0] * 3 / 4
    object_range_min_1 = table_bbox.min[1] * 3 / 4 + table_bbox.max[1] / 4
    object_range_max_1 = table_bbox.min[1] / 4 + table_bbox.max[1] * 3 / 4
    object_cfg["pos"] = np.array(
        [
            random.uniform(object_range_min_0, object_range_max_0),
            random.uniform(object_range_min_1, object_range_max_1),
            tbl_z,
        ]
    )
    # TODO: How about quat?

    return object_cfg


def get_state_machine(task, robot, sm_args={}):
    import state_machines.pick_sm
    import state_machines.place_sm

    STATE_MACHINES = {
        "pick": state_machines.pick_sm.PickStateMachine,
        "place": state_machines.place_sm.PlaceStateMachine,
    }
    if task not in STATE_MACHINES:
        raise ValueError("Unknown task: %s." % task)

    # Set the final position and quaternion for different tasks and robots
    final_position = get_final_position(task, robot, sm_args.get("device"))
    final_quat = _get_final_quat(task, robot, sm_args.get("device"))
    reach_dist_thres = _get_reach_dist_threshold(robot)
    if final_position is not None:
        sm_args["final_position"] = final_position
    if final_quat is not None:
        sm_args["final_quat"] = final_quat
    if reach_dist_thres is not None:
        sm_args["reach_dist_thres"] = reach_dist_thres

    return STATE_MACHINES[task](**sm_args)


def _get_object_size(object_path, device="cpu"):
    import omni.usd
    import pxr

    usd_context = omni.usd.get_context()
    stage = usd_context.get_stage()
    prim = stage.GetPrimAtPath(object_path)
    bbox_cache = pxr.UsdGeom.BBoxCache(
        pxr.Usd.TimeCode.Default(), [pxr.UsdGeom.Tokens.default_]
    )
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    size = bbox.max - bbox.min
    return torch.tensor(
        [[size[0], size[1], size[2]]], dtype=torch.float32, device=device
    )


def get_final_position(task, robot, device="cpu"):
    FINAL_POSITIONS = {
        "pick": {
            "franka": [0.3, 0, 0.3],
            "piper": [0.3, 0, 0.3],
        },
        "place": {
            "franka": [0.3, 0, 0.3],
            "piper": [0.3, 0, 0.3],
        },
    }
    if task in FINAL_POSITIONS:
        if FINAL_POSITIONS[task] is None:
            # The final position is not set for the task
            return None
        if robot in FINAL_POSITIONS[task]:
            return torch.tensor(
                [FINAL_POSITIONS[task][robot]], dtype=torch.float32, device=device
            )
    else:
        raise ValueError("Unknown task: %s." % task)


def _get_final_quat(task, robot, device="cpu"):
    FINAL_QUATS = {
        "pick": {
            "franka": [0, 1, 0, 0],
            "piper": [0, 1, 0, 0],
        },
        "place": {
            "franka": [0, 1, 0, 0],
            "piper": [0, 1, 0, 0],
        },
    }
    if task in FINAL_QUATS:
        if FINAL_QUATS[task] is None:
            # The final quaternion is not set for the task
            return None
        if robot in FINAL_QUATS[task]:
            return torch.tensor(
                [FINAL_QUATS[task][robot]], dtype=torch.float32, device=device
            )
    else:
        raise ValueError("Unknown task: %s." % task)


def _get_reach_dist_threshold(robot):
    REACHABLE_RANGE = {
        "franka": 0.75,
        "piper": 0.55,
    }
    if robot in REACHABLE_RANGE:
        return REACHABLE_RANGE[robot]
    else:
        raise ValueError("Unknown robot: %s." % robot)


def get_curr_state(
    ee_state,
    robot_joint_pos=None,
    object_state=None,
    container_state=None,
    env_origins=None,
    robot_quat=None,
    device="cpu",
):
    quat_opengl = robot_quat[:, [1, 2, 3, 0]]  # xyzw
    curr_state = {}
    if ee_state is not None:
        curr_state["end_effector"] = {
            "pos": _get_robot_relative_position(
                ee_state.target_pos_w[..., 0, :] - env_origins, quat_opengl
            ),
            "quat": ee_state.target_quat_w[..., 0, :],
        }
    if robot_joint_pos is not None:
        curr_state["joints"] = robot_joint_pos
    if object_state is not None:
        curr_state["object"] = {
            "pos": _get_robot_relative_position(
                object_state.root_pos_w - env_origins, quat_opengl
            ),
            "quat": object_state.root_quat_w,
            "velocity": _get_robot_relative_position(
                object_state.root_lin_vel_w, quat_opengl
            ),
        }
    if container_state is not None:
        curr_state["container"] = {
            "pos": _get_robot_relative_position(
                container_state.root_pos_w - env_origins, quat_opengl
            ),
            "quat": container_state.root_quat_w,
        }

    if device == "cpu":
        for csk, csv in curr_state.items():
            if isinstance(csv, dict):
                for k, v in csv.items():
                    if isinstance(v, torch.Tensor):
                        curr_state[csk][k] = v.cpu().numpy()
            elif isinstance(csv, torch.Tensor):
                curr_state[csk] = csv.cpu().numpy()

    return curr_state


def _get_robot_relative_position(point, robot_quat):
    # inv_quat = scipy.spatial.transform.Rotation.from_quat(robot_quat).inv()
    # inv_offset = inv_quat.apply(point)

    # pytorch3d/transforms/rotation_conversions.html#quaternion_invert
    inv_quat = robot_quat * torch.tensor([[-1, -1, -1, 1]], device=robot_quat.device)
    # pytorch3d/transforms/rotation_conversions.html#quaternion_apply
    t = 2.0 * torch.cross(inv_quat[..., :3], point, dim=-1)
    inv_offset = (
        point + inv_quat[..., 3:] * t + torch.cross(inv_quat[..., :3], t, dim=-1)
    )
    return inv_offset


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
    KNOWN_TAGS = {"ROBOT": 1, "OBJECT_BG": 2, "OBJECT_MAIN": 3, "CONTAINER": 4}
    seg_maps = np.zeros_like(rgba_seg_maps[..., :1], dtype=np.uint8)

    # Iterate over each image (since the tags may not be the same for each image)
    for si, st in enumerate(semantic_tags):
        for color, tag in st.items():
            tag_name = tag["class"].upper()
            if tag_name not in KNOWN_TAGS.keys():
                # logging.warning("Unknown semantic tag %s.", tag)
                continue

            # Convert the color string to a tuple (Unbelievable string here!)
            mask = np.all(rgba_seg_maps[si] == ast.literal_eval(color), axis=-1)
            seg_maps[si][mask] = KNOWN_TAGS[tag_name]

    return seg_maps


def is_final_position_reached(object_pos, ee_pos, final_pos):
    DIST_THRESHOLD = 0.02

    ee_offset = (ee_pos - final_pos).abs().sum(dim=1)
    obj_offset = (object_pos - final_pos).abs().sum(dim=1)
    return torch.bitwise_and(ee_offset < DIST_THRESHOLD, obj_offset < DIST_THRESHOLD)


def _get_curr_env_states(cam_views, curr_state, next_state, is_done):
    # Reorganize by env_id
    env_states = []
    for env_id in range(is_done.size(0)):
        if is_done[env_id].item():
            env_states.append({})
            continue

        env_states.append(
            {
                "sm_state": next_state["sm_state"][env_id].item(),
                "ee_pos": curr_state["end_effector"]["pos"][env_id].cpu().numpy(),
                "ee_quat": curr_state["end_effector"]["quat"][env_id].cpu().numpy(),
                "joints": curr_state["joints"][env_id].cpu().numpy(),
                "object_pos": curr_state["object"]["pos"][env_id].cpu().numpy(),
                "object_quat": curr_state["object"]["quat"][env_id].cpu().numpy(),
                "object_vel": curr_state["object"]["velocity"][env_id].cpu().numpy(),
                "grasp_pos": next_state["grasp_postion"][env_id].cpu().numpy(),
                "grasp_quat": next_state["grasp_quat"][env_id].cpu().numpy(),
                "action": next_state["action"][env_id].cpu().numpy(),
            }
        )
        # Add camera views to the env_states
        if cam_views is not None:
            for cam, imgs in cam_views.items():
                for k, v in imgs.items():
                    env_states[-1]["%s_%s" % (cam, k)] = v[env_id]

    return env_states


def simulate(sim_cfg, task_cfg, dir_cfg, debug_cfg):
    import configs.robot_cfg
    import omni.replicator.core as rep

    # Create a new environment
    env_cfg = get_env_cfg(
        dir_cfg["scene_dir"],
        dir_cfg["object_dir"],
        dir_cfg["container_dir"],
        sim_cfg,
        task_cfg["robot"],
    )
    env = gym.make("Robot-Env-Cfg-v0", cfg=env_cfg, seed=debug_cfg["seed"])
    # Reset environment at start
    # env.reset()
    env.reset(seed=debug_cfg["seed"])
    random.seed(debug_cfg["seed"])
    np.random.seed(debug_cfg["seed"])
    torch.manual_seed(debug_cfg["seed"])

    # Enable Path Tracing
    if debug_cfg["path_tracing"]:
        rep.settings.set_render_pathtraced()

    # Initialize the state machine
    state_machine = get_state_machine(
        task_cfg["task_name"],
        task_cfg["robot"],
        {
            "dt": env_cfg.sim.dt * env_cfg.decimation,
            "num_envs": env.unwrapped.num_envs,
            "device": env.unwrapped.device,
            "object_size": _get_object_size(
                "/World/envs/env_0/Object", env.unwrapped.device
            ),
            "gripper_length": configs.robot_cfg.get_gripper_length(task_cfg["robot"]),
        },
    )

    # Simulation loop
    env_states = [{} for _ in range(env.unwrapped.num_envs)]
    is_done = torch.zeros(
        env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device
    )
    robot_origin = (
        torch.from_numpy(env_cfg.scene.robot.init_state.pos[None, :])
        .float()
        .to(env.unwrapped.device)
    )
    robot_quat = (
        torch.from_numpy(env_cfg.scene.robot.init_state.rot[None, :])
        .float()
        .to(env.unwrapped.device)
    )

    while not is_done.all():
        # Add an option to disable the state machine to accelerate the simulation
        if debug_cfg["disable_sm"]:
            env.step(torch.from_numpy(env.action_space.sample()))
            continue

        curr_state = get_curr_state(
            env.unwrapped.scene["ee_frame"].data,
            env.unwrapped.scene.state["articulation"]["robot"]["joint_position"],
            env.unwrapped.scene["object"].data,
            (
                env.unwrapped.scene["container"].data
                if "container" in env.unwrapped.scene.keys()
                else None
            ),
            env.unwrapped.scene.env_origins + robot_origin,
            robot_quat,
            env.unwrapped.device,
        )
        next_state = state_machine.compute(
            curr_state
        )  # xyz, quat (wxyz), gripper (-1/1)
        cam_views = get_camera_views(
            env.unwrapped.scene.sensors, ["rgb", "depth", "seg"]
        )
        # Check whether the simulation is finished
        response = env.step(next_state["action"])
        # Ideally, _[-2] indicates the simulation is finished, which does not work.
        is_done = is_final_position_reached(
            curr_state["object"]["pos"],
            curr_state["end_effector"]["pos"],
            state_machine.final_object_pose[:, :3],
        )
        # Omit the sequence if the object is dropped or timeout
        if (
            response[-1]["log"]["Episode_Termination/time_out"]
            or response[-1]["log"]["Episode_Termination/object_dropping"]
        ):
            break
        # Save current env. states (camera views, robot states, and object states)
        _env_states = _get_curr_env_states(cam_views, curr_state, next_state, is_done)
        # Reorganize the env_states by keys
        for eid, es in enumerate(_env_states):
            for k, v in es.items():
                # Omit these keys in the dumped data
                if k in ["grasp_pos", "grasp_quat"]:
                    continue
                if k not in env_states[eid]:
                    env_states[eid][k] = []

                env_states[eid][k].append(v)

    env.close()
    # Ignore the simulation if the task is not finished
    # If in debug mode, save all simulation data even if the task is not finished
    return env_cfg, [
        es
        for env_id, es in enumerate(env_states)
        if is_done[env_id].item() or debug_cfg["debug"]
    ]


def get_episode_name(task, robot, scene_cfg):
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
    random_suffix = str(uuid.uuid4())[-12:]

    # Generate a unique name for the episode
    return "%s_%s_%s%s_O%02d_%s" % (
        task,
        robot,
        object_type,
        "d" if object_vel > 1e-3 else "s",
        n_objects,
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
                raise ValueError(f"Unknown camera data shape: {frame.shape}")

            # Normalize the depth image to 0-255
            if img_name in ["depth", "distance_to_image_plane", "distance_to_camera"]:
                frame = np.clip(frame, 0, MAX_DEPTH)
                frame = (frame / np.max(frame) * 255).astype(np.uint8)
            if img_name in ["seg", "semantic_segmentation"]:
                # Assign a color to each semantic class
                frame = cv2.applyColorMap(frame * 64, cv2.COLORMAP_JET)

            cam_frames[cam_name][img_name].append(frame)

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

        frames[frame_idx] = frame[..., ::-1]

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
        if isinstance(v, (int, np.int32, np.int64)):
            text += "%s: %d\n" % (k, v)
        elif isinstance(v, float):
            text += "%s: %.3f\n" % (k, v)
        elif isinstance(v, np.ndarray):
            # Convert all quaternions to Euler angles
            if k.find("Quat") != -1:
                k = k.replace("Quat", "Rot")
                v = scipy.spatial.transform.Rotation.from_quat(v).as_euler(
                    "xyz", degrees=True
                )

            text += "%s: " % k
            text += " ".join(["%.3f" % i for i in v]) + "\n"
        else:
            raise ValueError(f"Unknown State Value Type: {type(v)}")

    return text


def dump_video(frames, output_path, fps=24):
    if len(frames) == 0:
        return

    width, height = frames[0].shape[1], frames[0].shape[0]
    video = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for frame in frames:
        video.write(frame)

    video.release()


def main(simulation_app, args):
    with open(args.sim_cfg_file) as fp:
        sim_cfg = yaml.load(fp, Loader=yaml.FullLoader)

    # Perform simulations in the environment
    while simulation_app.is_running():
        seed = args.seed if args.seed is not None else random.randint(0, 65535)
        env_cfg, env_states = simulate(
            sim_cfg,
            {
                "task_name": args.task,
                "robot": args.robot,
            },
            {
                "scene_dir": args.scene_dir,
                "object_dir": args.object_dir,
                "container_dir": args.container_dir,
            },
            {
                "debug": args.debug,
                "disable_sm": args.disable_sm,
                "path_tracing": args.path_tracing,
                "seed": seed,
            },
        )
        # Save the simulation data
        for es in env_states:
            episode_name = get_episode_name(
                args.task, args.robot, env_cfg.scene.to_dict()
            )
            if args.save:
                with open(
                    os.path.join(args.output_dir, "%s.json" % episode_name), "w"
                ) as fp:
                    _env_cfg = env_cfg.to_dict()
                    _env_cfg["seed"] = seed
                    _env_cfg["final_position"] = get_final_position(
                        args.task, args.robot
                    ).numpy()
                    json.dump(get_object_without_numpy(_env_cfg), fp, indent=2)

                with h5py.File(
                    os.path.join(args.output_dir, "%s.h5" % episode_name), "w"
                ) as fp:
                    for k, v in es.items():
                        fp.create_dataset(k, data=v, compression="gzip")

            if args.debug:
                dump_video(
                    get_frames(es),
                    os.path.join(args.output_dir, "%s.mp4" % episode_name),
                )


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
    isaaclab.app.AppLauncher.add_app_launcher_args(parser)
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
        "--container_dir", default=os.path.join(PROJECT_HOME, os.pardir, "containers")
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(PROJECT_HOME, os.pardir, "datasets")
    )
    parser.add_argument("--task", default="pick")
    parser.add_argument(
        "--sim_cfg_file",
        default=os.path.join(PROJECT_HOME, "simulations", "configs", "sim_cfg.yaml"),
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--disable_sm", action="store_true", default=False)
    parser.add_argument("--path_tracing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(script_args)
    # Copy the shared parameters from isaaclab_args to args
    for sp in SHARED_PARAMETERS:
        if sp in isaaclab_args:
            setattr(args, sp, getattr(isaaclab_args, sp))

    app_launcher = isaaclab.app.AppLauncher(isaaclab_args)
    main(app_launcher.app, args)
    app_launcher.app.close()
