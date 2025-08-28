# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

"""
Usage:
python scripts/teleoperation.py --teleop_device keyboard --enable_cameras
python scripts/teleoperation.py --teleop_device keyboard --enable_cameras --robot piper
python scripts/teleoperation.py --teleop_device spacemouse --enable_cameras
"""

import argparse
import ast
import json
import logging
import os
import sys
import random
import uuid

import cv2
import gymnasium as gym
import h5py
import imageio.v3
import isaaclab.app
import numpy as np
import scipy.spatial.transform
import torch
import yaml

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)
sys.path.append(os.path.join(PROJECT_HOME, "simulations"))

import simulations.simulate as sim


def pre_process_actions(
    teleop_data: tuple[np.ndarray, bool] | list[tuple[np.ndarray, np.ndarray, np.ndarray]], num_envs: int, device: str
) -> torch.Tensor:
    """Convert teleop data to the format expected by the environment action space.

    Args:
        teleop_data: Data from the teleoperation device.
        num_envs: Number of environments.
        device: Device to create tensors on.

    Returns:
        Processed actions as a tensor.
    """
    # compute actions based on environment
    # resolve gripper command
    delta_pose, gripper_command = teleop_data
    # convert to torch
    delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=device).repeat(num_envs, 1)
    gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=device)
    gripper_vel[:] = -1 if gripper_command else 1
    # compute actions
    return torch.concat([delta_pose, gripper_vel], dim=1)


def main(simulation_app, args):
    if "handtracking" in args.teleop_device.lower():
        from isaacsim.xr.openxr import OpenXRSpec

    from isaaclab.devices import OpenXRDevice, Se3Gamepad, Se3Keyboard, Se3SpaceMouse

    if args.enable_pinocchio:
        from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import GR1T2Retargeter
        import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
    from isaaclab.devices.openxr.retargeters.manipulator import GripperRetargeter, Se3AbsRetargeter, Se3RelRetargeter
    from isaaclab.managers import TerminationTermCfg as DoneTerm

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.manager_based.manipulation.lift import mdp
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
    from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

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

    # Calculate the bounding boxes of the target objects
    object_sizes = sim.get_object_sizes(
        args.object_dir, sim_cfg["scene"]["target_object"]["categories"]
    )

    task_cfg = {
        "task_name": args.task,
        "robot": args.robot,
    }
    dir_cfg = {
        "scene_dir": args.scene_dir,
        "object_dir": args.object_dir,
        "container_dir": args.container_dir,
    }

    # Perform simulations in the environment
    seed = args.seed if args.seed is not None else random.randint(0, 65535)

    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = sim.get_env_cfg(
        dir_cfg["scene_dir"],
        dir_cfg["object_dir"],
        dir_cfg["container_dir"],
        sim_cfg,
        object_sizes,
        task_cfg["robot"],
    )
    # modify configuration
    env_cfg.terminations.time_out = None
    env_cfg.actions.arm_action.controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
    env_cfg.actions.arm_action.scale=0.5
    # set the resampling time range to large number to avoid resampling
    env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # add termination condition for reaching the goal otherwise the environment won't reset
    env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    # create environment
    env = gym.make("Robot-Env-Cfg-v0", cfg=env_cfg, seed=seed).unwrapped
    # Reset environment at start
    env.reset(seed=seed)

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance():
        """Reset the environment to its initial state.

        This callback is triggered when the user presses the reset key (typically 'R').
        It's useful when:
        - The robot gets into an undesirable configuration
        - The user wants to start over with the task
        - Objects in the scene need to be reset to their initial positions

        The environment will be reset on the next simulation step.
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    def start_teleoperation():
        """Activate teleoperation control of the robot.

        This callback enables active control of the robot through the input device.
        It's typically triggered by a specific gesture or button press and is used when:
        - Beginning a new teleoperation session
        - Resuming control after temporarily pausing
        - Switching from observation mode to control mode

        While active, all commands from the device will be applied to the robot.
        """
        nonlocal teleoperation_active
        teleoperation_active = True

    def stop_teleoperation():
        """Deactivate teleoperation control of the robot.

        This callback temporarily suspends control of the robot through the input device.
        It's typically triggered by a specific gesture or button press and is used when:
        - Taking a break from controlling the robot
        - Repositioning the input device without moving the robot
        - Pausing to observe the scene without interference

        While inactive, the simulation continues to render but device commands are ignored.
        """
        nonlocal teleoperation_active
        teleoperation_active = False

    # create controller
    if args.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05 * args.sensitivity, rot_sensitivity=0.05 * args.sensitivity
        )
    elif args.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args.sensitivity, rot_sensitivity=0.05 * args.sensitivity
        )
    elif args.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args.sensitivity, rot_sensitivity=0.1 * args.sensitivity
        )
    elif "dualhandtracking_abs" in args.teleop_device.lower() and "GR1T2" in args.task:
        # Create GR1T2 retargeter with desired configuration
        gr1t2_retargeter = GR1T2Retargeter(
            enable_visualization=True,
            num_open_xr_hand_joints=2 * (int(OpenXRSpec.HandJointEXT.XR_HAND_JOINT_LITTLE_TIP_EXT) + 1),
            device=env.unwrapped.device,
            hand_joint_names=env.scene["robot"].data.joint_names[-22:],
        )

        # Create hand tracking device with retargeter
        teleop_interface = OpenXRDevice(
            env_cfg.xr,
            retargeters=[gr1t2_retargeter],
        )
        teleop_interface.add_callback("RESET", reset_recording_instance)
        teleop_interface.add_callback("START", start_teleoperation)
        teleop_interface.add_callback("STOP", stop_teleoperation)

        # Hand tracking needs explicit start gesture to activate
        teleoperation_active = False

    elif "handtracking" in args.teleop_device.lower():
        # Create EE retargeter with desired configuration
        if "_abs" in args.teleop_device.lower():
            retargeter_device = Se3AbsRetargeter(
                bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, zero_out_xy_rotation=True
            )
        else:
            retargeter_device = Se3RelRetargeter(
                bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, zero_out_xy_rotation=True
            )

        grip_retargeter = GripperRetargeter(bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT)

        # Create hand tracking device with retargeter (in a list)
        teleop_interface = OpenXRDevice(
            env_cfg.xr,
            retargeters=[retargeter_device, grip_retargeter],
        )
        teleop_interface.add_callback("RESET", reset_recording_instance)
        teleop_interface.add_callback("START", start_teleoperation)
        teleop_interface.add_callback("STOP", stop_teleoperation)

        # Hand tracking needs explicit start gesture to activate
        teleoperation_active = False
    else:
        raise ValueError(
            f"Invalid device interface '{args.teleop_device}'. Supported: 'keyboard', 'spacemouse', 'gamepad',"
            " 'handtracking', 'handtracking_abs'."
        )

    # add teleoperation key for env reset (for all devices)
    teleop_interface.add_callback("R", reset_recording_instance)
    print(teleop_interface)

    # reset environment
    env.reset()
    teleop_interface.reset()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get device command
            teleop_data = teleop_interface.advance()

            # Only apply teleop commands when active
            if teleoperation_active:
                # compute actions based on environment
                actions = pre_process_actions(teleop_data, env.num_envs, env.device)
                # apply actions
                env.step(actions)
            else:
                env.sim.render()

            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False

    # close the simulator
    env.close()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    SHARED_PARAMETERS = ["num_envs", "save"]

    # add argparse arguments
    parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
    parser.add_argument(
        "--enable_pinocchio",
        action="store_true",
        default=False,
        help="Enable Pinocchio.",
    )
    # append AppLauncher cli args
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

    app_launcher_args = vars(args)

    if args.enable_pinocchio:
        # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
        # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
        # GR1T2 retargeter
        import pinocchio  # noqa: F401
    if "handtracking" in args.teleop_device.lower():
        app_launcher_args["xr"] = True

    # run the main function
    main(app_launcher.app, args)
    app_launcher.app.close()
