# -*- coding: utf-8 -*-
#
# @File:   place_sm.py
# @Author: The Isaac Lab Project Developers
# @Date:   2025-03-22 17:10:52
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-02 15:21:56
# @Email:  root@haozhexie.com

import collections

import numpy as np
import torch
import warp as wp
from . import sm_utils


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PlaceSmState:
    """States for the place state machine."""

    INIT = wp.constant(0)
    RESET = wp.constant(1)
    APPROACH_ABOVE_OBJECT = wp.constant(2)
    APPROACH_OBJECT = wp.constant(3)
    GRASP_OBJECT = wp.constant(4)
    LIFT_OBJECT = wp.constant(5)
    APPROACH_ABOVE_TARGET = wp.constant(6)
    APPROACH_TARGET = wp.constant(7)
    PLACE_OBJECT = wp.constant(8)
    TO_TARGET = wp.constant(9)


class PlaceSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    INIT = wp.constant(0.28)
    RESET = wp.constant(0.08)
    APPROACH_ABOVE_OBJECT = wp.constant(0.4)
    APPROACH_OBJECT = wp.constant(0.72)
    GRASP_OBJECT = wp.constant(0.72)
    LIFT_OBJECT = wp.constant(0.2)
    APPROACH_ABOVE_TARGET = wp.constant(0.4)
    APPROACH_TARGET = wp.constant(0.4)
    PLACE_OBJECT = wp.constant(0.4)
    TO_TARGET = wp.constant(0.6)


class PlaceStateMachine:
    """A simple state machine in a robot's task space to place and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper.
    """

    def __init__(
        self,
        dt: float,
        num_envs: int,
        init_pose: torch.tensor,
        final_pose: torch.tensor,
        max_reach_dist: float,
        grasp_dist_thres: float = 0.01,
        place_dist_thres: float = 0.1,
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

        # the final end-effector position after placing (quat in xyzw)
        self.final_eef_pose = final_pose.repeat(num_envs, 1)
        # the reachable range of the robot (too far cannot be reached)
        self.max_reach_dist = max_reach_dist
        # the distance threshold for grasping
        self.grasp_dist_thres = grasp_dist_thres
        # the distance threshold for placing
        self.place_dist_thres = place_dist_thres
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

    def reset_idx(self, env_ids: collections.abc.Sequence[int] = None):
        if env_ids is None:
            env_ids = slice(None)

        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def _get_place_pose(
        self,
        container_position: torch.Tensor,
    ) -> torch.Tensor:
        # Simple implementation
        place_position = container_position.clone()
        place_position[:, 2] += 0.10
        place_quat = self.final_eef_pose[:, 3:7]

        return torch.cat([place_position, place_quat], dim=-1)

    def _is_object_placed(
        self,
        object_projected_size: torch.Tensor,
        object_position: torch.Tensor,
        container_projected_size: torch.Tensor,
        container_position: torch.Tensor,
    ) -> torch.Tensor:
        object_relative_size = object_projected_size / 2
        object_negz_mask = (object_relative_size[:, 2] > 0).unsqueeze(1)
        object_negz_size = torch.where(
            object_negz_mask, -object_relative_size, object_relative_size
        )
        lowest_point = object_position + object_negz_size.sum(dim=0)

        containier_relative_size = container_projected_size / 2
        object_container_rela = lowest_point - container_position
        containier_axis_lengths = torch.norm(containier_relative_size, dim=1)
        containier_axis_dirs = (
            containier_relative_size / containier_axis_lengths[:, None]
        )
        object_container_projections = torch.matmul(
            containier_axis_dirs, object_container_rela[0]
        )

        return torch.all(
            torch.abs(object_container_projections) <= containier_axis_lengths
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
        place_pose = self._get_place_pose(
            curr_state["container"]["pos"],
        )
        is_object_placed = self._is_object_placed(
            curr_state["object"]["size"],
            curr_state["object"]["pos"],
            curr_state["container"]["size"],
            curr_state["container"]["pos"],
        ).item()
        grasp_pose = torch.cat([grasp_position, grasp_quat], dim=-1)
        object_pose = torch.cat([curr_state["object"]["pos"], grasp_quat], dim=-1)

        # Convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        grasp_pose_wp = wp.from_torch(grasp_pose, wp.transform)
        object_pose_wp = wp.from_torch(object_pose, wp.transform)
        object_vel_wp = wp.from_torch(curr_state["object"]["velocity"], wp.vec3)
        place_pose_wp = wp.from_torch(place_pose, wp.transform)
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
                place_pose_wp,
                final_eef_pose_wp,
                pose_angle_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.max_reach_dist,
                self.grasp_dist_thres,
                self.grasp_pose_thres,
                self.object_dist_thres,
                is_object_placed,
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
    place_pose: wp.array(dtype=wp.transform),
    final_eef_pose: wp.array(dtype=wp.transform),
    pose_angle: wp.array(dtype=float),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    max_reach_dist: float,  # the object is reachable
    grasp_dist_threshold: float,  # the object is graspable
    grasp_pose_threshold: float,  # the ee pose is aligned with object
    object_dist_threshold: float,  # the object to be considered grasped
    is_object_placed: bool,  # the object is placed
):
    debug = True
    tid = wp.tid()
    state = sm_state[tid]
    # Thresholds for checking offsets
    thres_object_ee = wp.vec3(
        grasp_dist_threshold,
        grasp_dist_threshold,
        offset[tid][2] + grasp_dist_threshold,
    )
    thres_target_ee = wp.vec3(
        grasp_dist_threshold,
        grasp_dist_threshold,
        offset[tid][2] + grasp_dist_threshold,
    )
    # State machine
    if state == PlaceSmState.INIT:
        gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = init_pose[tid]
        dist_eef_obj = sm_utils.get_length(
            wp.transform_get_translation(grasp_pose[tid])
        )
        if sm_wait_time[tid] >= PlaceSmWaitTime.INIT and dist_eef_obj < max_reach_dist:
            sm_state[tid] = PlaceSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
        if debug:
            print("INIT")
    elif state == PlaceSmState.RESET:
        gripper_state[tid] = GripperState.OPEN
        dist_eef_obj = sm_utils.get_length(
            wp.transform_get_translation(grasp_pose[tid])
        )
        if dist_eef_obj > max_reach_dist:
            des_ee_pose[tid] = final_eef_pose[tid]
        else:
            des_ee_pose[tid] = wp.transform_multiply(offset[tid], grasp_pose[tid])
            if sm_wait_time[tid] >= PlaceSmWaitTime.RESET:
                sm_state[tid] = PlaceSmState.APPROACH_ABOVE_OBJECT
                sm_wait_time[tid] = 0.0
        if debug:
            print("RESET")
    elif state == PlaceSmState.APPROACH_ABOVE_OBJECT:
        gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], grasp_pose[tid])
        offset_object_ee = sm_utils.get_offset(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(object_pose[tid]),
        )
        dist_eef_obj = sm_utils.get_length(
            wp.transform_get_translation(grasp_pose[tid])
        )
        if dist_eef_obj >= max_reach_dist:
            sm_state[tid] = PlaceSmState.RESET
            sm_wait_time[tid] = 0.0
        elif (
            sm_utils.is_offset_below_threshold(
                offset_object_ee, thres_object_ee, object_vel[tid]
            )
            and pose_angle[tid] < grasp_pose_threshold
        ):
            if sm_wait_time[tid] >= PlaceSmWaitTime.APPROACH_ABOVE_OBJECT:
                sm_state[tid] = PlaceSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
        if debug:
            wp.printf("APPROACH_ABOVE_OBJECT: pose_angle: %.4f\n", pose_angle[tid])
    elif state == PlaceSmState.APPROACH_OBJECT:
        gripper_state[tid] = GripperState.OPEN
        des_ee_pose[tid] = grasp_pose[tid]
        if sm_wait_time[tid] >= PlaceSmWaitTime.APPROACH_OBJECT:
            sm_state[tid] = PlaceSmState.GRASP_OBJECT
            sm_wait_time[tid] = 0.0
        if debug:
            print("APPROACH_OBJECT")
    elif state == PlaceSmState.GRASP_OBJECT:
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PlaceSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PlaceSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
        if debug:
            print("GRASP_OBJECT")
    elif state == PlaceSmState.LIFT_OBJECT:
        gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        dist_ee_object = sm_utils.get_length(
            wp.transform_get_translation(ee_pose[tid])
            - wp.transform_get_translation(object_pose[tid])
        )
        if dist_ee_object > object_dist_threshold:
            sm_state[tid] = PlaceSmState.RESET
            sm_wait_time[tid] = 0.0
        elif sm_wait_time[tid] >= PlaceSmWaitTime.LIFT_OBJECT:
            sm_state[tid] = PlaceSmState.APPROACH_ABOVE_TARGET
            sm_wait_time[tid] = 0.0
        if debug:
            print("LIFT_OBJECT")
    elif state == PlaceSmState.APPROACH_ABOVE_TARGET:
        gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], place_pose[tid])
        dist_ee_object = sm_utils.get_length(
            wp.transform_get_translation(ee_pose[tid])
            - wp.transform_get_translation(object_pose[tid])
        )
        offset_ee_target = sm_utils.get_offset(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(place_pose[tid]),
        )
        if dist_ee_object > object_dist_threshold:
            sm_state[tid] = PlaceSmState.RESET
            sm_wait_time[tid] = 0.0
        elif sm_utils.is_offset_below_threshold(
            offset_ee_target, thres_target_ee, object_vel[tid]
        ):
            if sm_wait_time[tid] >= PlaceSmWaitTime.APPROACH_ABOVE_TARGET:
                sm_state[tid] = PlaceSmState.PLACE_OBJECT
                sm_wait_time[tid] = 0.0
        if debug:
            wp.printf(
                "APPROACH_ABOVE_TARGET: ee: [%.4f, %.4f, %.4f] tgt: [%.4f, %.4f, %.4f], ",
                ee_pose[tid][0],
                ee_pose[tid][1],
                ee_pose[tid][2],
                place_pose[tid][0],
                place_pose[tid][1],
                place_pose[tid][2],
            )
            wp.printf(
                "offset: [%.4f, %.4f, %.4f], threshold: [%.4f, %.4f, %.4f]\n",
                offset_ee_target[0],
                offset_ee_target[1],
                offset_ee_target[2],
                thres_target_ee[0],
                thres_target_ee[1],
                thres_target_ee[2],
            )
    elif state == PlaceSmState.APPROACH_TARGET:
        des_ee_pose[tid] = place_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PlaceSmWaitTime.APPROACH_TARGET:
            sm_state[tid] = PlaceSmState.PLACE_OBJECT
            sm_wait_time[tid] = 0.0
        if debug:
            print("APPROACH_TARGET")
    elif state == PlaceSmState.PLACE_OBJECT:
        des_ee_pose[tid] = place_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PlaceSmWaitTime.PLACE_OBJECT:
            sm_state[tid] = PlaceSmState.TO_TARGET
            sm_wait_time[tid] = 0.0
        if debug:
            print("PLACE_OBJECT")
    elif state == PlaceSmState.TO_TARGET:
        des_ee_pose[tid] = final_eef_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        dist_object_target = sm_utils.get_length(
            wp.transform_get_translation(object_pose[tid])
            - wp.transform_get_translation(place_pose[tid])
        )
        if not is_object_placed:
            sm_state[tid] = PlaceSmState.RESET
            sm_wait_time[tid] = 0.0
        if debug:
            wp.printf(
                "TO_TARGET: ee: [%.4f, %.4f, %.4f] tgt: [%.4f, %.4f, %.4f], dist: %.4f\n",
                object_pose[tid][0],
                object_pose[tid][1],
                object_pose[tid][2],
                place_pose[tid][0],
                place_pose[tid][1],
                place_pose[tid][2],
                dist_object_target,
            )

    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]
