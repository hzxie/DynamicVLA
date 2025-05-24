# -*- coding: utf-8 -*-
#
# @File:   place_sm.py
# @Author: The Isaac Lab Project Developers
# @Date:   2025-03-22 17:10:52
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-05-22 11:10:20
# @Email:  root@haozhexie.com

import collections

import numpy as np
import torch
import warp as wp

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PlaceSmState:
    """States for the place state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    TO_TARGET = wp.constant(5)
    PLACE = wp.constant(6)


class PlaceSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.1)
    APPROACH_ABOVE_OBJECT = wp.constant(0.4)
    APPROACH_OBJECT = wp.constant(0.2)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(0.2)
    TO_TARGET = wp.constant(0.6)
    PLACE = wp.constant(0.3)


class PlaceStateMachine:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the position.
    3. APPROACH_OBJECT: The robot moves to the position.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to a height.
    6. TO_TARGET: The robot moves the object to the desired pose. 
    7. PLACE: The robot place the object. This is the final state.
    """

    def __init__(
        self,
        dt: float,
        num_envs: int,
        object_size: torch.tensor,
        final_position: torch.tensor,
        final_quat: torch.tensor,
        reach_dist_thres: float,
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
        self.object_size = object_size
        # initialize state machine
        self.sm_dt = torch.full((num_envs,), dt, device=device)
        self.sm_state = torch.full((num_envs,), 0, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros((num_envs,), device=device)

        # desired grasp state
        self.grasp_position = torch.zeros((num_envs, 3), device=device)
        self.standard_orientation = torch.zeros((num_envs, 4), device=device)
        self.standard_orientation[:, 1] = 1.0
        # next gripper state
        self.des_ee_pose = torch.zeros((num_envs, POSE_DIM), device=device)
        self.des_gripper_state = torch.full((num_envs,), 0.0, device=device)

        # the final object position after lifting (quat in xyzw)
        self.final_object_pose = torch.cat([final_position, final_quat], dim=1)

        # the reachable range of the robot (too near or too far cannot be reached)
        self.reach_dist_thres = reach_dist_thres
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
        self.offset[:, 2] = 0.1  # offset height
        self.offset[:, -1] = 1.0  # warp expects quaternion as (xyzw)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: collections.abc.Sequence[int] = None):
        if env_ids is None:
            env_ids = slice(None)

        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def _get_grasp_position(
        self,
        object_size: torch.Tensor,
        object_position: torch.Tensor,
    ) -> torch.Tensor:
        OBJECT_HEIGHT_THRES = 0.006
        grasp_position = object_position
        object_height = object_size[:, 2]
        grasp_position_z = object_position[:, 2]

        # Plan A: Grasp as low as you can
        grasp_position_z = object_height - self.gripper_length
        
        # # Plan B: Try to grasp the center of the object
        # grasp_position_z = torch.where(object_height > OBJECT_HEIGHT_THRES * 2, grasp_position_z - OBJECT_HEIGHT_THRES, grasp_position_z)
        # grasp_position_z = torch.where(object_height > self.gripper_length * 2, object_height - self.gripper_length, grasp_position_z)
        
        # ensure grasping height higher than table
        grasp_position[:, 2] = torch.where(grasp_position_z < OBJECT_HEIGHT_THRES, OBJECT_HEIGHT_THRES, grasp_position_z)
        return grasp_position

    def compute(self, curr_state: dict) -> torch.Tensor:
        ee_pose = torch.cat(
            [
                curr_state["end_effector"]["pos"],
                curr_state["end_effector"]["quat"][:, [1, 2, 3, 0]],  # xyzw
            ],
            dim=-1,
        )

        # Determine the object position before lifting
        self.grasp_position = self._get_grasp_position(
            self.object_size,
            curr_state["object"]["pos"],
        )

        grasp_pose = torch.cat([self.grasp_position, self.standard_orientation], dim=-1)
        object_pose = torch.cat([curr_state["object"]["pos"], self.standard_orientation], dim=-1)

        # Convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        grasp_pose_wp = wp.from_torch(grasp_pose, wp.transform)
        object_pose_wp = wp.from_torch(object_pose, wp.transform)
        final_object_pose_wp = wp.from_torch(self.final_object_pose, wp.transform)

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
                ee_pose_wp,
                final_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.reach_dist_thres,
                self.grasp_dist_thres,
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
            "sm_state": self.sm_state,
            "grasp_postion": self.grasp_position,
            "grasp_quat": self.standard_orientation,
        }


@wp.func
def get_distance(current_pos: wp.vec3, desired_pos: wp.vec3) -> float:
    return wp.length(current_pos - desired_pos)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    grasp_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    ee_pose: wp.array(dtype=wp.transform),
    final_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    reach_dist_threshold: float,  # the object is reachable
    grasp_dist_threshold: float,  # the object is graspable
    object_dist_threshold: float,  # the object to be considered grasped
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PlaceSmState.REST:
        # print("REST")
        gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = final_object_pose[tid]
        dist = get_distance(
            wp.vec3(0.0, 0.0, 0.0),
            wp.transform_get_translation(grasp_pose[tid]),
        )
        # wait for a while
        if sm_wait_time[tid] >= PlaceSmWaitTime.REST and dist < reach_dist_threshold:
            # move to next state and reset wait time
            sm_state[tid] = PlaceSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PlaceSmState.APPROACH_ABOVE_OBJECT:
        # print("APPROACH_ABOVE_OBJECT")
        gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], grasp_pose[tid])
        grasp_dist = get_distance(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
        )
        if grasp_dist < grasp_dist_threshold :
            # wait for a while
            if sm_wait_time[tid] >= PlaceSmWaitTime.APPROACH_ABOVE_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PlaceSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PlaceSmState.APPROACH_OBJECT:
        gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = grasp_pose[tid]
        dist = get_distance(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
        )
        if dist < grasp_dist_threshold:
            if sm_wait_time[tid] >= PlaceSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PlaceSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PlaceSmState.GRASP_OBJECT:
        gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = grasp_pose[tid]
        # wait for a while
        if sm_wait_time[tid] >= PlaceSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PlaceSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PlaceSmState.LIFT_OBJECT:
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] < PlaceSmWaitTime.LIFT_OBJECT:
            # lift the object away from the desktop
            des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        else:
            # move to next state and reset wait time
            sm_state[tid] = PlaceSmState.TO_TARGET
            sm_wait_time[tid] = 0.0
    elif state == PlaceSmState.TO_TARGET:
        des_ee_pose[tid] = final_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE

        dist = get_distance(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(object_pose[tid]),
        )
        # check if the object is grasped
        if dist > object_dist_threshold:
            sm_state[tid] = PlaceSmState.REST
            sm_wait_time[tid] = 0.0

    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]
