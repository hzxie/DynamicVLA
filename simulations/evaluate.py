# -*- coding: utf-8 -*-
#
# @File:   evaluate.py
# @Author: Haozhe Xie
# @Date:   2025-05-06 15:21:20
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2026-01-13 09:11:13
# @Email:  root@haozhexie.com

import argparse
import datetime
import json
import logging
import os
import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import yaml
import zmq
from isaaclab.app import AppLauncher

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)
sys.path.append(os.path.join(PROJECT_HOME, "simulations"))

import simulations.simulate as sim
from utils.instruction_generator import InstructionGenerator


def get_zmq_sockets(host, img_port, act_port):
    context = zmq.Context()

    obs_socket = context.socket(zmq.PUB)
    obs_socket.bind("tcp://%s:%d" % (host, img_port))

    act_socket = context.socket(zmq.PULL)
    act_socket.bind("tcp://%s:%d" % (host, act_port))

    return obs_socket, act_socket


def get_test_env(
    cfg,
    num_envs,
    scene_dir,
    object_dir,
    physics_time_step,
    tolerance,
    device,
    disable_fabric,
    path_tracing,
):
    import omni.replicator.core as rep

    # Create the environment
    env_cfg = _get_env_cfg(
        cfg, num_envs, scene_dir, object_dir, tolerance, device, disable_fabric
    )
    env_cfg.dt = physics_time_step
    env_cfg.episode_length_s = cfg["episode_length_s"]
    env = gym.make("Robot-Env-Cfg-v0", cfg=env_cfg, seed=cfg["seed"])
    # Increase the fictional frictions of the object
    sim.set_object_material(
        env.unwrapped.scene["object"],
        n_envs=env.unwrapped.num_envs,
    )

    # Enable Path Tracing
    if path_tracing:
        rep.settings.set_render_pathtraced()

    return env


def _get_env_cfg(
    cfg, num_envs, scene_dir, object_dir, tolerance, device, disable_fabric
):
    import configs.event_cfg
    import configs.robot_cfg
    import configs.scene_cfg
    import isaaclab_tasks
    import omni.usd

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
        device=device,
        num_envs=num_envs,
        use_fabric=not disable_fabric,
    )
    env_cfg.events = configs.event_cfg.get_event_cfg(
        None
        if "perturbation" not in cfg["events"]
        else {
            "force": cfg["events"]["perturbation"]["params"]["force_range"],
            "torque": cfg["events"]["perturbation"]["params"]["torque_range"],
        }
    )
    env_cfg.terminations = _get_terimation_cfg(cfg["terminations"], tolerance, device)

    scene_usd_path = os.path.join(
        scene_dir, os.path.basename(cfg["scene"]["house"]["spawn"]["usd_path"])
    )
    logging.info("Loading scene from %s" % scene_usd_path)
    env_cfg.scene = configs.scene_cfg.set_house_asset(
        env_cfg.scene, os.path.join(scene_dir, scene_usd_path)
    )
    usd_context = omni.usd.get_context()
    usd_context.new_stage()

    # Set up the robot arm
    robot_name = configs.robot_cfg.get_robot_name(
        cfg["scene"]["robot"]["spawn"]["usd_path"]
    )
    env_cfg = configs.env_cfg.set_robot(
        robot_name,
        env_cfg,
        {
            "pos": cfg["scene"]["robot"]["init_state"]["pos"],
            "quat": cfg["scene"]["robot"]["init_state"]["rot"],
        },
    )
    # Set up cameras in the scene
    env_cfg.scene = _set_up_scene_cameras(env_cfg.scene, cfg["scene"])

    # Set the light intensity and color
    assert "distant_light" in cfg["scene"]
    env_cfg.scene = _set_up_scene_distant_light(
        env_cfg.scene, cfg["scene"]["distant_light"]
    )

    # Dynamically add objects / containers to scene
    env_cfg.scene = _set_up_scene_objects(env_cfg.scene, cfg["scene"], object_dir)
    return env_cfg


def _get_terimation_cfg(cfg, tolerance, device):
    import configs.termination_cfg

    if "object_picked" in cfg:
        task = "pick"
        args = cfg["object_picked"]["params"]
    elif "objects_placed" in cfg:
        task = "place"
        args = cfg["objects_placed"]["params"]
    elif "object_placed" in cfg:
        task = "place"
        # Competible with single-object placement (legacy implementation)
        args = cfg["object_placed"]["params"]
        args["objects"] = ["object"]
        args["object_sizes"] = {"object": args["object_size"]}
        del args["object_size"]
    else:
        raise NotImplementedError("Unsupported termination config.")

    args["tolerance"] = tolerance
    for k, v in args.items():
        # Tensorize the arguments
        if isinstance(v, (int, float)) or (
            isinstance(v, list) and all(isinstance(x, str) for x in v)
        ):
            args[k] = v
        elif isinstance(v, list):
            args[k] = torch.tensor(v, dtype=torch.float32, device=device)
        elif isinstance(v, dict):
            args[k] = {
                _k: torch.tensor(_v, dtype=torch.float32, device=device)
                for _k, _v in v.items()
            }
        else:
            raise ValueError("Unsupported termination argument type: %s" % type(v))

    return configs.termination_cfg.get_termination_cfg(task, args)


def _set_up_scene_cameras(scene_cfg, cfg):
    import configs.scene_cfg

    for k, v in cfg.items():
        if (
            not isinstance(v, dict)
            or "class_type" not in v
            or not isinstance(v["class_type"], str)
            or not v["class_type"].startswith("isaaclab.sensors.camera")
        ):
            continue

        # Remove prefix: '/World/envs/env_.*'
        prim_path = v["prim_path"]
        prim_path = prim_path[prim_path.rfind("/Robot") :]

        scene_cfg = configs.scene_cfg.add_scene_camera(
            scene_cfg,
            k,
            configs.scene_cfg.get_camera_cfg(
                {
                    "prim_path": prim_path,
                    "fps": 1 / v["update_period"],
                    "width": v["width"],
                    "height": v["height"],
                    "data_types": v["data_types"],
                    "focal_length": v["spawn"]["focal_length"],
                    "focus_distance": v["spawn"]["focus_distance"],
                    "horizontal_aperture": v["spawn"]["horizontal_aperture"],
                    "clip": {
                        "near": v["spawn"]["clipping_range"][0],
                        "far": v["spawn"]["clipping_range"][1],
                    },
                    "pos": v["offset"]["pos"],
                    "quat": v["offset"]["rot"],
                    "convention": v["offset"]["convention"],
                }
            ),
        )

    return scene_cfg


def _set_up_scene_distant_light(scene_cfg, cfg):
    import configs.scene_cfg

    scene_cfg = configs.scene_cfg.set_light_asset(
        scene_cfg,
        position=cfg["init_state"]["pos"],
        temperature=cfg["spawn"]["color_temperature"],
        intensity=cfg["spawn"]["intensity"],
    )
    return scene_cfg


def _set_up_scene_objects(scene_cfg, cfg, object_dir):
    import configs.object_cfg

    for k, v in cfg.items():
        if not (
            isinstance(v, dict)
            and "class_type" in v
            and v["class_type"]
            == "isaaclab.assets.rigid_object.rigid_object:RigidObject"
        ):
            continue

        usd_file_path = None
        if "usd_path" in v["spawn"]:
            usd_folder = os.path.basename(os.path.dirname(v["spawn"]["usd_path"]))
            usd_file_path = os.path.join(
                object_dir, usd_folder, os.path.basename(v["spawn"]["usd_path"])
            )
            logging.info("Loading object from %s" % usd_file_path)
            assert os.path.exists(usd_file_path)

        # Remove prefix: '/World/envs/env_.*'
        prim_path = v["prim_path"]
        prim_path = prim_path[prim_path.rfind("/") :]
        object_cfg = configs.object_cfg.get_object_cfg(
            prim_path,
            {
                "pos": v["init_state"]["pos"],
                "quat": v["init_state"]["rot"],
                "lin_vel": v["init_state"]["lin_vel"],
                "ang_vel": v["init_state"]["ang_vel"],
            },
            configs.object_cfg.get_spawner_cfg(
                usd_file_path,
                v["spawn"]["mass_props"]["mass"],
                v["spawn"]["rigid_props"]["angular_damping"],
                v["spawn"]["semantic_tags"],
            ),
        )
        if k == "object":
            scene_cfg = configs.scene_cfg.set_target_object(scene_cfg, object_cfg)
        else:
            scene_cfg = configs.scene_cfg.add_object(scene_cfg, k, object_cfg)

    return scene_cfg


def get_latest_action(act_socket):
    action = None
    while True:
        try:
            action = act_socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            break

    return action


def _get_action_tensor(action, num_envs, device):
    if isinstance(action, np.ndarray):
        action = torch.from_numpy(action).to(device)
    elif isinstance(action, torch.tensor):
        action = action.to(device)
    else:
        logging.warning("Unsupported action type: %s" % type(action))
        action = None

    if action.size(0) != num_envs or action.size(1) != 8:
        logging.warning(
            "Received action with shape %s, expected (%d, 8)" % (action.shape, num_envs)
        )
        action = None

    return action


def simulate(env, obs_socket, act_socket, init_poses):
    import configs.robot_cfg
    import configs.termination_cfg

    rcv_action = False
    last_action = None
    sim_results = {"status": -1, "cam_views": [], "ee_path": []}
    # The simulation loop
    term_mgr = env.env.termination_manager
    done_term = configs.termination_cfg.get_done_term(term_mgr.active_terms)
    tick = time.perf_counter()
    step_time = None
    while sim_results["status"] == -1:
        # scene_state = env.unwrapped.scene.state
        cam_view = sim.get_camera_views(env.unwrapped.scene.sensors, ["rgb"])
        curr_state = sim.get_curr_state(
            ee_state=env.unwrapped.scene["ee_frame"].data,
            # robot_joint_pos=scene_state["articulation"]["robot"]["joint_position"],
            object_state=env.unwrapped.scene["object"].data,
            env_origins=env.unwrapped.scene["robot"].data.root_pos_w,
            robot_quat=env.unwrapped.scene["robot"].data.root_quat_w,
            device=env.unwrapped.device,
        )
        sim_results["cam_views"].append(cam_view)
        sim_results["ee_path"].append(curr_state["end_effector"]["pos"].cpu().numpy())
        obs_socket.send_pyobj(
            {
                "dt_scale": (
                    1.0 if step_time is None else max(1.0, step_time / env.env.step_dt)
                ),
                "index": len(sim_results["cam_views"]) - 1,
                "observation.state": {
                    "end_effector": {
                        k: v.cpu().numpy()
                        for k, v in curr_state["end_effector"].items()
                    }
                },
                **{"observation.images.%s" % k: v["rgb"] for k, v in cam_view.items()},
            }
        )

        action = get_latest_action(act_socket)
        if action is not None and "action" in action:
            action = _get_action_tensor(
                action["action"], env.unwrapped.num_envs, env.unwrapped.device
            )
            last_action = action
            rcv_action = True

        # If no action is received, use the previous action to make the
        # simulation continuous
        if last_action is None:
            robot_name = configs.robot_cfg.get_robot_name(
                env.unwrapped.scene["robot"].cfg.spawn.usd_path
            )
            last_action = init_poses[robot_name].repeat(env.unwrapped.num_envs, 1)

        env.step(last_action)
        step_time = time.perf_counter() - tick
        # Make sure each step takes at least step_dt seconds
        if step_time < env.env.step_dt:
            time.sleep(env.env.step_dt - step_time)

        tick = time.perf_counter()
        logging.debug(
            "[Step%03d] Time: %.4fs; Scale: %.2f. Action shape: %s"
            % (
                len(sim_results["cam_views"]) - 1,
                step_time,
                step_time / env.env.step_dt,
                (action.shape if isinstance(action, torch.Tensor) else None),
            )
        )
        if term_mgr.get_term(done_term).all():
            sim_results["status"] = 0
        elif term_mgr.dones.all():
            sim_results["status"] = 1 if rcv_action else 2

    # term_mgr.reset()  # NOT WORKING
    term_mgr = term_mgr.__init__(term_mgr.cfg, env.env)
    return sim_results


def get_frames(cam_views):
    frames = {}
    for cv in cam_views:
        for cam, sensors in cv.items():
            for sensor, view in sensors.items():
                key = "%s_%s" % (cam, sensor)
                if key not in frames:
                    frames[key] = []

                frames[key].append(view.squeeze(0))

    return frames


def get_episode_name(cfg_filename, sim_status):
    seq_name = os.path.splitext(cfg_filename)[0]
    return "%s-%s-%s.mp4" % (
        seq_name,
        datetime.datetime.now().strftime("%m%d-%H%M%S"),
        "SUCCESS" if sim_status == 0 else "FAIL",
    )


def get_sim_results(sim_cfg, env_cfg_file_path, obs_socket, act_socket):
    prev_cfg_path = getattr(get_sim_results, "cfg_path", None)
    if env_cfg_file_path == prev_cfg_path:
        env = getattr(get_sim_results, "env")
        env_cfg = getattr(get_sim_results, "cfg")
    else:
        env = getattr(get_sim_results, "env", None)
        if env is not None:
            env.close()  # Close the previous environment

        logging.info("Recovering test environment from %s" % env_cfg_file_path)
        with open(env_cfg_file_path, "r") as fp:
            env_cfg = json.load(fp)

        env = get_test_env(
            env_cfg,
            sim_cfg["num_envs"],
            sim_cfg["scene_dir"],
            sim_cfg["object_dir"],
            sim_cfg["physics_time_step"],
            sim_cfg["tolerance"],
            sim_cfg["device"],
            sim_cfg["disable_fabric"],
            sim_cfg["path_tracing"],
        )
        setattr(get_sim_results, "cfg_path", env_cfg_file_path)
        setattr(get_sim_results, "cfg", env_cfg)
        setattr(get_sim_results, "env", env)

    # Randomize the task instruction before fixing the random seed
    instruction = InstructionGenerator.generate_instruction(env_cfg["instruction"])
    # Fix random seed for reproducibility
    random.seed(env_cfg["seed"])
    np.random.seed(env_cfg["seed"])
    torch.manual_seed(env_cfg["seed"])
    env.reset(seed=env_cfg["seed"])

    # Send the task instruction at the beginning of the simulation
    obs_socket.send_pyobj({"task": instruction})
    sim_results = simulate(env, obs_socket, act_socket, sim_cfg["init_poses"])
    logging.info("Simulation finished with code: %d" % sim_results["status"])
    # Clear the action socket
    get_latest_action(act_socket)

    return sim_results


def main(simulation_app, args):
    # Load the initial poses of the robots (as the default action)
    with open(args.sim_cfg_file) as fp:
        sim_cfg = yaml.load(fp, Loader=yaml.FullLoader)
        init_poses = {
            # Add a gripper open/close flag at the end of the init pose
            k: torch.tensor(
                [v["init_pose"] + [1]], dtype=torch.float32, device=args.device
            )
            for k, v in sim_cfg["robots"].items()
        }

    logging.info("Starting evaluation server...")
    # Set up Zero MQ context and sockets
    obs_socket, act_socket = get_zmq_sockets(args.host, args.img_port, args.act_port)
    logging.info(
        "ZeroMQs are listening on %s:%d for images and %s:%d for actions"
        % (args.host, args.img_port, args.host, args.act_port)
    )

    # Determine the test suites
    _, ext = os.path.splitext(args.env_cfg)
    assert ext in [".txt", ".json"], "Unsupported Config File: %s" % args.env_cfg

    test_envs = []
    if ext == ".json":
        test_envs.append(args.env_cfg)
    elif ext == ".txt":
        with open(args.env_cfg) as fp:
            test_envs = fp.read().splitlines()

    test_envs = [te for te in test_envs if os.path.exists(te)]
    if not test_envs:
        logging.fatal("No valid test environments found. Exiting.")
        sys.exit(2)

    logging.info("#Test environments: %d." % len(test_envs))
    # Simulation loop
    sim_cfg = {
        "num_envs": args.num_envs,
        "scene_dir": args.scene_dir,
        "object_dir": args.object_dir,
        "physics_time_step": args.physics_time_step,
        "tolerance": args.tolerance,
        "device": args.device,
        "disable_fabric": args.disable_fabric,
        "path_tracing": args.path_tracing,
        "init_poses": init_poses,
    }
    while simulation_app.is_running():
        action = get_latest_action(act_socket)
        # Hand-shake to the VLA client
        if action is None or "vla" not in action:
            logging.debug("No VLA clients connected.")
            time.sleep(10)
            continue

        vla_name = action["vla"]
        output_dir = os.path.join(args.output_dir, vla_name, "%04d" % action["epoch"])
        success_rates = {}

        os.makedirs(output_dir, exist_ok=True)
        logging.info("Evaluation started. Output Dir: %s" % (output_dir))
        for te in test_envs:
            n_tests = 0
            env_name = os.path.basename(te)[:-5]
            success_rates[env_name] = 0
            while n_tests < args.n_tests:
                sim_results = get_sim_results(sim_cfg, te, obs_socket, act_socket)
                # Save the frames if needed
                # NOTE:
                # 1) Reset will occur twice if failed (I dnk why), and the second time
                #    will produce just one frame.
                # 2) sim_status == 2 means no action was received, so we do not save
                #    the frames.
                if len(sim_results["cam_views"]) <= 1:  # Unexpected extra reset
                    continue

                n_tests += 1
                if sim_results["status"] == 0:
                    success_rates[env_name] += 1
                elif sim_results["status"] == 2:
                    logging.info("No action received. VLA client may be disconnected.")
                    break

                episode_name = get_episode_name(env_name, sim_results["status"])
                obs_socket.send_pyobj(
                    {
                        "env_name": env_name,
                        "eps_name": episode_name[:-4],
                        "success": sim_results["status"] == 0,
                        "ee_path": np.array(sim_results["ee_path"]),
                    }
                )
                if args.save:
                    episode_file_path = os.path.join(output_dir, episode_name)
                    logging.info(
                        "Saving videos (%d frames) to %s"
                        % (len(sim_results["cam_views"]), episode_file_path)
                    )
                    sim.dump_video(
                        sim.get_frames(
                            get_frames(sim_results["cam_views"]), state_keys=[]
                        ),
                        episode_file_path,
                    )

                # Wait for VLA client to acknowledge the start of this test
                logging.info("VLA client ACK pending...")
                n_waits = 0
                action = get_latest_action(act_socket)
                while action is None or "ack" not in action:
                    time.sleep(1)
                    action = get_latest_action(act_socket)
                    n_waits += 1
                    if n_waits >= 30:
                        break
                if n_waits >= 30:
                    logging.warning(
                        "VLA client[%s] ACK timeout. Skip this env." % vla_name
                    )
                    break

            # Calcuate the success rate for this env
            success_rates[env_name] /= n_tests

        # Say goodbye to the VLA client
        obs_socket.send_pyobj({"vla": vla_name, "success_rates": success_rates})


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    SHARED_PARAMETERS = ["num_envs", "save"]

    parser = argparse.ArgumentParser(description="Evaluation Server Runner")
    # Arguments for the IsaacLab
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the data from camera at index specified by ``--camera_id``.",
    )
    AppLauncher.add_app_launcher_args(parser)
    isaaclab_args, script_args = parser.parse_known_args()

    # Arguments for the script
    parser.add_argument("--path_tracing", action="store_true")
    parser.add_argument("--physics_time_step", type=float, default=0.04)
    parser.add_argument("--tolerance", type=float, default=0.07)
    parser.add_argument(
        "--scene_dir", default=os.path.join(PROJECT_HOME, os.pardir, "scenes")
    )
    parser.add_argument(
        "--object_dir", default=os.path.join(PROJECT_HOME, os.pardir, "objects")
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_HOME, "runs", "evaluation"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--img_port", default=3186, type=int, help="Port for image stream"
    )
    parser.add_argument(
        "--act_port", default=3188, type=int, help="Port for action stream"
    )
    parser.add_argument(
        "-n",
        "--n_tests",
        type=int,
        default=20,
        help="The number of tests to run",
    )
    parser.add_argument(
        "--sim_cfg_file",
        default=os.path.join(PROJECT_HOME, "simulations", "configs", "sim_cfg.yaml"),
    )
    parser.add_argument("--env_cfg", required=True)
    args = parser.parse_args(script_args)
    # Copy the shared parameters from isaaclab_args to args
    for sp in SHARED_PARAMETERS:
        if sp in isaaclab_args:
            setattr(args, sp, getattr(isaaclab_args, sp))

    app_launcher = AppLauncher(isaaclab_args)
    main(app_launcher.app, args)
    app_launcher.app.close()
