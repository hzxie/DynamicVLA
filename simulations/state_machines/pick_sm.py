# -*- coding: utf-8 -*-
#
# @File:   pick_sm.py
# @Author: The Isaac Lab Project Developers
# @Date:   2025-03-22 17:10:52
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-08 15:45:28
# @Email:  root@haozhexie.com

import collections

import torch
import warp as wp

from . import sm_utils


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    INIT = wp.constant(0)
    RESET = wp.constant(1)
    APPROACH_ABOVE_OBJECT = wp.constant(2)
    APPROACH_OBJECT = wp.constant(3)
    GRASP_OBJECT = wp.constant(4)
    LIFT_OBJECT = wp.constant(5)
    TO_TARGET = wp.constant(6)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    INIT = wp.constant(0.48)
    RESET = wp.constant(0.08)
    APPROACH_ABOVE_OBJECT = wp.constant(0.4)
    APPROACH_OBJECT = wp.constant(0.72)
    GRASP_OBJECT = wp.constant(0.72)
    LIFT_OBJECT = wp.constant(0.2)
    TO_TARGET = wp.constant(0.6)


class PickStateMachine:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:
    """

    def __init__(
        self,
        dt: float,
        num_envs: int,
        init_pose: torch.tensor,
        final_pose: torch.tensor,
        max_reach_dist: float,
        grasp_dist_thres: float = 0.01,
        grasp_pose_thres: float = 15.0,
        object_dist_thres: float = 0.1,
        gripper_length: float = 0.3,
        device: torch.device | str = "cpu",
    ):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        POSE_DIM = 7

        # parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        # initialize state machine
        self.init_pose = init_pose[:, [0, 1, 2, 4, 5, 6, 3]].repeat(num_envs, 1)
        self.sm_dt = torch.full((num_envs,), dt, device=device)
        self.sm_state = torch.full((num_envs,), 0, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros((num_envs,), device=device)
        # next gripper state
        self.des_ee_pose = torch.zeros((num_envs, POSE_DIM), device=device)
        self.des_gripper_state = torch.full((num_envs,), 0.0, device=device)
        # if use cache grasp quat
        self.use_cache_grasp_quat = torch.zeros((num_envs,), dtype=torch.bool, device=device)

        # the final end-effector position after lifting (quat in xyzw)
        self.final_eef_pose = final_pose.repeat(num_envs, 1)
        # the reachable range of the robot (too far cannot be reached)
        self.max_reach_dist = max_reach_dist
        # the distance threshold for grasping
        self.grasp_dist_thres = grasp_dist_thres
        # the angle threshold (degree) for grasping
        self.grasp_pose_thres = grasp_pose_thres
        # the distance threshold for the object to be considered grasped
        self.object_dist_thres = object_dist_thres
        # the gripper length
        self.gripper_length = gripper_length

        # approach above object offset
        self.offset = torch.zeros((num_envs, POSE_DIM), device=device)
        self.offset[:, 2] = 0.1  # z offset
        self.offset[:, -1] = 1.0  # warp expects quaternion as (xyzw)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)
        self.init_pose_wp = wp.from_torch(self.init_pose, wp.transform)
        self.use_cache_grasp_quat_wp = wp.from_torch(self.use_cache_grasp_quat, wp.bool)

    def reset_idx(self, env_ids: collections.abc.Sequence[int] = None):
        if env_ids is None:
            env_ids = slice(None)

        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, curr_state: dict) -> torch.Tensor:
        ee_pose = torch.cat(
            [
                curr_state["end_effector"]["pos"],
                curr_state["end_effector"]["quat"][:, [1, 2, 3, 0]],  # xyzw
            ],
            dim=-1,
        )
        # Determine the object position before lifting
        grasp_position = sm_utils.get_grasp_position(
            curr_state["object"]["size"],
            curr_state["object"]["pos"],
            curr_state["object"]["velocity"],
            self.gripper_length,
        )
        grasp_quat = sm_utils.get_grasp_quat(
            curr_state["object"]["size"],
            curr_state["object"]["velocity"],
            self.final_eef_pose,
            self.device,
        )
        pose_angle = sm_utils.get_pose_angle(
            grasp_quat[:, [3, 0, 1, 2]],
            curr_state["end_effector"]["quat"],
        )
        grasp_pose = torch.cat([grasp_position, grasp_quat], dim=-1)
        object_pose = torch.cat([curr_state["object"]["pos"], grasp_quat], dim=-1)

        # Convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        grasp_pose_wp = wp.from_torch(grasp_pose, wp.transform)
        object_pose_wp = wp.from_torch(object_pose, wp.transform)
        object_vel_wp = wp.from_torch(curr_state["object"]["velocity"], wp.vec3)
        final_eef_pose_wp = wp.from_torch(self.final_eef_pose, wp.transform)
        pose_angle_wp = wp.from_torch(pose_angle, wp.float32)

        # Run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                grasp_pose_wp,
                object_pose_wp,
                object_vel_wp,
                ee_pose_wp,
                self.init_pose_wp,
                final_eef_pose_wp,
                pose_angle_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.use_cache_grasp_quat_wp, 
                self.max_reach_dist,
                self.grasp_dist_thres,
                self.grasp_pose_thres,
                self.object_dist_thres,
            ],
            device=self.device,
        )
        # Convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]

        # Convert to torch (xyz, quat, grabber_state)
        action = torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)
        return {
            "action": action,
            "sm_state": self.sm_state.clone(),
            "grasp_position": grasp_position,
            "grasp_quat": grasp_quat,
        }


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    grasp_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    object_vel: wp.array(dtype=wp.vec3),
    ee_pose: wp.array(dtype=wp.transform),
    init_pose: wp.array(dtype=wp.transform),
    final_eef_pose: wp.array(dtype=wp.transform),
    pose_angle: wp.array(dtype=float),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    use_cache_grasp_quat: wp.array(dtype=wp.bool),
    max_reach_dist: float,  # the object is reachable
    grasp_dist_threshold: float,  # the object is graspable
    grasp_pose_threshold: float,  # the ee pose is aligned with object
    object_dist_threshold: float,  # the object to be considered grasped
):
    debug = False
    tid = wp.tid()
    state = sm_state[tid]

    # decide next state
    if state == PickSmState.INIT:
        gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = init_pose[tid]
        dist_eef_obj = sm_utils.get_length(
            wp.transform_get_translation(grasp_pose[tid])
        )
        if sm_wait_time[tid] >= PickSmWaitTime.INIT and dist_eef_obj < max_reach_dist:
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
        if debug:
            print("INIT")
    elif state == PickSmState.RESET:
        gripper_state[tid] = GripperState.OPEN
        dist_eef_obj = sm_utils.get_length(
            wp.transform_get_translation(grasp_pose[tid])
        )
        if dist_eef_obj > max_reach_dist:
            des_ee_pose[tid] = final_eef_pose[tid]
        else:
            des_ee_pose[tid] = wp.transform_multiply(offset[tid], grasp_pose[tid])
            if sm_wait_time[tid] >= PickSmWaitTime.RESET:
                sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
                sm_wait_time[tid] = 0.0
        if debug:
            print("RESET")
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], grasp_pose[tid])
        thres_object_ee = wp.vec3(
            grasp_dist_threshold,
            grasp_dist_threshold,
            offset[tid][2] + grasp_dist_threshold,
        )
        offset_object_ee = sm_utils.get_offset(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(object_pose[tid]),
        )
        dist_eef_obj = sm_utils.get_length(
            wp.transform_get_translation(grasp_pose[tid])
        )
        if dist_eef_obj >= max_reach_dist:
            sm_state[tid] = PickSmState.RESET
            sm_wait_time[tid] = 0.0
        elif (
            sm_utils.is_offset_below_threshold(
                offset_object_ee, thres_object_ee, object_vel[tid]
            )
            and pose_angle[tid] < grasp_pose_threshold
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_ABOVE_OBJECT:
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
        if debug:
            wp.printf("APPROACH_ABOVE_OBJECT: pose_angle: %.4f\n", pose_angle[tid])
    elif state == PickSmState.APPROACH_OBJECT:
        gripper_state[tid] = GripperState.OPEN
        if not use_cache_grasp_quat[tid] :
            des_ee_pose[tid] = grasp_pose[tid]
            use_cache_grasp_quat[tid] = True
        else :
            des_ee_pose[tid] = wp.transformation(
                wp.transform_get_translation(grasp_pose[tid]),
                wp.transform_get_rotation(des_ee_pose[tid])
            )
        
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            sm_state[tid] = PickSmState.GRASP_OBJECT
            use_cache_grasp_quat[tid] = False
            sm_wait_time[tid] = 0.0
        if debug:
            print("APPROACH_OBJECT")
    elif state == PickSmState.GRASP_OBJECT:
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
        if debug:
            print("GRASP_OBJECT")
    elif state == PickSmState.LIFT_OBJECT:
        gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        dist_ee_object = sm_utils.get_length(
            wp.transform_get_translation(ee_pose[tid])
            - wp.transform_get_translation(object_pose[tid])
        )
        if dist_ee_object > object_dist_threshold:
            sm_state[tid] = PickSmState.RESET
            sm_wait_time[tid] = 0.0
        elif sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
            sm_state[tid] = PickSmState.TO_TARGET
            sm_wait_time[tid] = 0.0
        if debug:
            print("LIFT_OBJECT")
    elif state == PickSmState.TO_TARGET:
        des_ee_pose[tid] = final_eef_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        dist_ee_object = sm_utils.get_length(
            wp.transform_get_translation(ee_pose[tid])
            - wp.transform_get_translation(object_pose[tid])
        )
        if dist_ee_object > object_dist_threshold:
            sm_state[tid] = PickSmState.RESET
            sm_wait_time[tid] = 0.0
        if debug:
            wp.printf(
                "TO_TARGET: obj: [%.4f, %.4f, %.4f] ee: [%.4f, %.4f, %.4f], dist: %.4f\n",
                object_pose[tid][0],
                object_pose[tid][1],
                object_pose[tid][2],
                ee_pose[tid][0],
                ee_pose[tid][1],
                ee_pose[tid][2],
                dist_ee_object,
            )

    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]
