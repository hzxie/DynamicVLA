# -*- coding: utf-8 -*-
#
# @File:   env_cfg.py
# @Author: Haozhe Xie
# @Date:   2025-03-22 21:04:28
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-04-30 16:31:23
# @Email:  root@haozhexie.com

from dataclasses import MISSING

import configs.robot_cfg
from configs.scene_cfg import SceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift import mdp


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        object_position = ObservationTermCfg(
            func=mdp.object_position_in_robot_root_frame
        )
        target_object_position = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewardTermCfg(
        func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0
    )
    lifting_object = RewardTermCfg(
        func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0
    )
    object_goal_tracking = RewardTermCfg(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewardTermCfg(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )
    # action penalty
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    object_dropping = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.05,
            "asset_cfg": SceneEntityCfg("object"),
        },
        time_out=True,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurriculumTermCfg(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )
    joint_vel = CurriculumTermCfg(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=25)
    # Robot settings
    actions: configs.robot_cfg.ActionCfg = MISSING
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # number of simulation steps per environment step
        self.decimation = 2
        # the length of the episode in seconds
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


def set_robot(robot: str, env_cfg: EnvCfg, robot_pose: list) -> EnvCfg:
    assert robot in ["franka", "piper"], "Unknown robot: %s" % robot
    env_cfg.commands.object_pose.body_name = configs.robot_cfg.get_body_name(robot)
    env_cfg.scene.ee_frame = configs.robot_cfg.get_ee_frame_cfg(robot)
    env_cfg.scene.robot = configs.robot_cfg.get_robot_cfg(robot)
    env_cfg.scene.robot.init_state.pos = robot_pose["pos"]
    env_cfg.scene.robot.init_state.rot = robot_pose["quat"]
    env_cfg.actions = configs.robot_cfg.get_robot_action_cfg(robot)
    return env_cfg
