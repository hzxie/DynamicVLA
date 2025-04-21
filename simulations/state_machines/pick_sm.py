# -*- coding: utf-8 -*-
#
# @File:   lift_object.py
# @Author: The Isaac Lab Project Developers
# @Date:   2025-03-22 17:10:52
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-04-11 19:23:15
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

    REST = wp.constant(0.1)
    APPROACH_ABOVE_OBJECT = wp.constant(0.2)
    APPROACH_OBJECT = wp.constant(0.5)
    GRASP_OBJECT = wp.constant(0.2)
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
    """

    def __init__(
        self,
        dt: float,
        num_envs: int,
        device: torch.device | str = "cpu",
        dist_threshold=0.005,
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
        self.dist_threshold = dist_threshold
        # initialize state machine
        self.sm_dt = torch.full((num_envs,), dt, device=device)
        self.sm_state = torch.full((num_envs,), 0, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros((num_envs,), device=device)

        # desired grasp state
        self.grasp_position = torch.zeros((num_envs, 3), device=device)
        self.grasp_wait_time = torch.ones((num_envs,), device=device) * -1
        self.grasp_pose_changed = torch.zeros(
            (num_envs,), dtype=torch.bool, device=device
        )
        # next gripper state
        self.des_ee_pose = torch.zeros((num_envs, POSE_DIM), device=device)
        self.des_gripper_state = torch.full((num_envs,), 0.0, device=device)

        # the final object position after lifting
        self.final_object_pose = torch.zeros((num_envs, POSE_DIM), device=device)
        self.final_object_pose[:, [0, 2]] = 0.3  # set lift position as [0.3, 0, 0.3]
        self.final_object_pose[:, 4] = 1.0  # set quaternion (xyzw) as [0, 1, 0, 0]

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
        self.grasp_wait_time[env_ids] = -1.0

    def _get_grasp_position(
        self,
        object_position: torch.Tensor,
        object_velocity: torch.Tensor,
        prev_grasp_position: torch.Tensor,
        prev_waiting_time: torch.Tensor,
    ) -> torch.Tensor:
        
        # Initialization: Constant waiting time
        prediction_time = 0.25
        curr_waiting_time = torch.full((object_velocity.size(0), ), prediction_time, device=self.device)
        # Initialization: Estimate grasp position according to the velocity
        grasp_position = object_position + object_velocity * curr_waiting_time[:, None]

        return grasp_position

    def _get_grasp_quat(self, object_velocity: torch.Tensor) -> torch.Tensor:
        # NOTE: Rotation around the z-axis (0, 0, 1)
        #       Quat = [
        #         x * sin(theta / 2),
        #         y * sin(theta / 2),
        #         z * sin(theta / 2),
        #         w * cos(theta / 2)
        #       ]

        # Determine the grasp quaternion according to the velocity
        gsp_theta = np.pi / 2 - torch.arctan2(object_velocity[:, 1], object_velocity[:, 0])
        gsp_quat = torch.stack(
            [
                torch.zeros_like(gsp_theta),
                torch.zeros_like(gsp_theta),
                torch.sin(gsp_theta / 2),
                torch.cos(gsp_theta / 2),
            ],
            dim=1,
        )  # xyzw

        # Consider the basic rotation of the gripper
        return self._quat_multiply(gsp_quat, self.final_object_pose[:, 3:7])

    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiplication."""
        x1, y1, z1, w1 = q1.unbind(1)
        x2, y2, z2, w2 = q2.unbind(1)
        return torch.stack(
            [
                w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
                w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
                w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
                w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
            ],
            dim=1,
        )

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
            curr_state["object"]["pos"],
            curr_state["object"]["velocity"],
            self.grasp_position,
            self.grasp_wait_time,
        )
        
        grasp_quat = self._get_grasp_quat(curr_state["object"]["velocity"])
        grasp_pose = torch.cat([self.grasp_position, grasp_quat], dim=-1)
        object_pose = torch.cat([curr_state["object"]["pos"], grasp_quat], dim=-1)

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
                self.dist_threshold,
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
            "grasp_quat": grasp_quat,
            "grasp_wait_time": self.grasp_wait_time,
        }


@wp.func
def get_distance(current_pos: wp.vec3, desired_pos: wp.vec3) -> float:
    return wp.length(current_pos - desired_pos)

@wp.func
def get_height_distance(current_pos: wp.vec3, desired_pos: wp.vec3) -> float:
    return wp.abs(current_pos[2] - desired_pos[2])


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
    dist_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        print("REST")
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], grasp_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        dist = get_height_distance(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(object_pose[tid]),
        )
        wp.printf("ABOVE_OBJECT. Dist=%f\n", dist)
        if dist < dist_threshold + 0.1:
            print("ABOVE_OBJECT DIST!!")
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_ABOVE_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = grasp_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        dist = get_height_distance(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(object_pose[tid]),
        )
        wp.printf("APPROACH_OBJECT. WTime %f, Dist=%f\n", dist)
        if dist < dist_threshold:
            print("APPROACH_OBJECT DIST!!")
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        print("GRASP_OBJECT")
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = final_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        dist = get_distance(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
        )

        print("LIFT_OBJECT")
        if dist < dist_threshold:
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0

    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]
