# -*- coding: utf-8 -*-
#
# @File:   lift_object.py
# @Author: The Isaac Lab Project Developers
# @Date:   2025-03-22 17:10:52
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-04-07 15:38:06
# @Email:  root@haozhexie.com

import collections
import os
import sys
import numpy as np
import torch

import warp as wp

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.2)
    APPROACH_OBJECT = wp.constant(0.1)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(0.6)


class PickStateMachine:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the position.
    3. APPROACH_OBJECT: The robot moves to the position.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    # TODO: Add a state to lift the gripper vertically as a intermediate state to prevent it from sweeping the table
    """

    def __init__(
        self,
        dt: float,
        num_envs: int,
        device: torch.device | str = "cpu",
        position_threshold=0.01,
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
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((num_envs,), dt, device=device)
        self.sm_state = torch.full((num_envs,), 0, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros((num_envs,), device=device)

        # grasp state
        self.grasp_wait_time = torch.zeros((num_envs,), device=device)
        self.des_pose_changed = torch.zeros(
            (num_envs,), dtype=torch.bool, device=device
        )

        # desired state
        self.des_ee_pose = torch.zeros((num_envs, POSE_DIM), device=device)
        self.des_gripper_state = torch.full((num_envs,), 0.0, device=device)
        ## the final object position after lifting
        self.des_object_pose = torch.zeros((num_envs, POSE_DIM), device=device)
        self.des_object_pose[:, 2] = 0.5  # lift height
        self.des_object_pose[:, 4] = 1.0  # set quaternion (xyzw) as [1, 0, 0, 0]

        # approach above object offset
        self.offset = torch.zeros((num_envs, POSE_DIM), device=device)
        self.offset[:, 2] = 0.15  # approach height (height before lifting)
        self.offset[:, -1] = 1.0  # warp expects quaternion as (xyzw)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.grasp_wait_time_wp = wp.from_torch(self.grasp_wait_time, wp.float32)
        self.des_pose_changed_wp = wp.from_torch(self.des_pose_changed, wp.bool)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: collections.abc.Sequence[int] = None):
        if env_ids is None:
            env_ids = slice(None)

        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def get_grasp_positions(
        self, object_pos: torch.Tensor, object_velocity: torch.Tensor
    ) -> torch.Tensor:
        pass

    def get_grasp_waiting_time(self, object_pos: torch.Tensor) -> torch.Tensor:
        pass

    def get_ee_relative_position(
        self, point: torch.Tensor, ee_quat: torch.Tensor
    ) -> torch.Tensor:
        # inv_quat = scipy.spatial.transform.Rotation.from_quat(ee_quat).inv()
        # inv_offset = inv_quat.apply(point)

        # pytorch3d/transforms/rotation_conversions.html#quaternion_invert
        inv_quat = ee_quat * torch.tensor([[-1, -1, -1, 1]], device=ee_quat.device)
        # pytorch3d/transforms/rotation_conversions.html#quaternion_apply
        t = 2.0 * torch.cross(inv_quat[..., :3], point, dim=-1)
        inv_offset = (
            point + inv_quat[..., 3:] * t + torch.cross(inv_quat[..., :3], t, dim=-1)
        )
        return inv_offset

    def compute(self, curr_state: dict, robot_quat: torch.Tensor) -> torch.Tensor:
        robot_quat = robot_quat[:, [1, 2, 3, 0]]  # xyzw
        object_pos = self.get_ee_relative_position(
            curr_state["object"]["pos"], robot_quat
        )

        # Concatenate the position and quaternion
        ee_pos = curr_state["end_effector"]["pos"]
        ee_quat = curr_state["end_effector"]["quat"][:, [1, 2, 3, 0]]  # xyzw
        ee_pose = torch.cat([ee_pos, ee_quat], dim=-1)

        # TODO: Determine the object position before lifting
        object_quat = self.des_object_pose[:, 3:7]  # quaternion (xyzw): [1, 0, 0, 0]
        object_pose = torch.cat([object_pos, object_quat], dim=-1)

        # Convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(self.des_object_pose, wp.transform)

        # Run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                self.grasp_wait_time_wp,
                self.des_pose_changed_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )
        # Convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]

        # Convert to torch (xyz, quat, grabber_state)
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


@wp.func
def is_within_distance(
    current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float
) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    grasp_wait_time: wp.array(dtype=float),
    des_pose_changed: wp.array(dtype=bool),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if is_within_distance(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        if des_pose_changed[tid]:
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
            des_ee_pose[tid] = object_pose[tid]

        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if is_within_distance(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if grasp_wait_time[tid] < 0.1:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if is_within_distance(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]
