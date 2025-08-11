# -*- coding: utf-8 -*-
#
# @File:   helpers.py
# @Author: Haozhe Xie
# @Date:   2025-06-14 15:17:59
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-08-11 10:22:56
# @Email:  root@haozhexie.com

import json
import logging
import os
import pathlib

import av
import easydict
import numpy as np
import scipy.spatial.transform
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from PIL import Image


def get_n_parameters(model: PreTrainedPolicy, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_formatted_big_number(num: int, precision: int = 0) -> str:
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num


def get_rotation_vector(quat, format="quat", scalar_first=True):
    if format == "quat":
        return quat.astype(np.float32)
    elif format == "euler":
        return _get_euler_angle_from_quaternion(quat, scalar_first).astype(np.float32)
    elif format == "rotvec":
        # return (
        #     scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=scalar_first)
        #     .as_rotvec()
        #     .astype(np.float32)
        # )
        # This implementation is aligned with LIBERO dataset
        return _get_axis_angle_from_quaternion(quat, scalar_first).astype(np.float32)
    else:
        raise ValueError(
            "Unsupported format: %s. Use 'quat', 'euler', or 'rotvec'." % format
        )


def _get_euler_angle_from_quaternion(quat, scalar_first=True):
    euler_angles = (
        scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=scalar_first)
        .as_euler("xyz", degrees=False)
        .astype(np.float32)
    )
    # Make euler angles in the range [0, 2 * pi) for rX and rZ -> Make the values continuous
    euler_angles[..., [0, 2]] = np.mod(euler_angles[..., [0, 2]], 2 * np.pi)

    return euler_angles


def _get_axis_angle_from_quaternion(quat, scalar_first=True):
    # Ref: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    # assert quat.ndim == 2 and quat.shape[1] == 4, quat.shape
    if scalar_first:
        quat = quat[..., [1, 2, 3, 0]]  # wxyz to xyzw

    # Clamp w (quat[:, 3]) to [-1.0, 1.0]
    w = np.clip(quat[..., 3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)

    # Angle part in radians
    angles = 2.0 * np.arccos(w)
    # Avoid division by zero
    zero_mask = den < 1e-8

    # Normalize axis and multiply by angle
    axis = np.zeros_like(quat[..., :3])
    axis[~zero_mask] = quat[~zero_mask, :3] / den[~zero_mask, np.newaxis]
    return axis * angles[..., np.newaxis]


def get_quaternion(rotation, format="rotvec", scalar_first=True):
    if format == "rotvec":
        return (
            scipy.spatial.transform.Rotation.from_rotvec(rotation)
            .as_quat(scalar_first=scalar_first)
            .astype(np.float32)
        )
    elif format == "euler":
        # Align with the convention in _get_euler_angle_from_quaternion (inverse)
        rotation[..., [0, 2]] = (rotation[..., [0, 2]] + np.pi) % (2 * np.pi) - np.pi
        return (
            scipy.spatial.transform.Rotation.from_euler("xyz", rotation, degrees=False)
            .as_quat(scalar_first=scalar_first)
            .astype(np.float32)
        )
    elif format == "quat":
        return rotation.astype(np.float32)
    else:
        raise ValueError(
            "Unsupported format: %s. Use 'rotvec', 'euler', or 'quat'." % format
        )


def get_delta_timestamps(
    policy_cfg: str,
    dataset_cfg: dict[str, list[float]] | None = None,
) -> dict[str, list] | None:
    # Ref: lerobot.datasets.factorfvy.resolve_delta_timestamps
    policy_cfg = get_policy_cfg(policy_cfg)
    delta_timestamps = {}
    if policy_cfg.reward_delta_indices is not None:
        delta_timestamps["reward"] = policy_cfg.reward_delta_indices
    if policy_cfg.action_delta_indices is not None:
        delta_timestamps["action"] = policy_cfg.action_delta_indices
    if policy_cfg.observation_delta_indices is not None:
        delta_timestamps["observation"] = policy_cfg.observation_delta_indices

    # Overwrite with dataset configuration if provided
    if dataset_cfg is not None:
        delta_timestamps = {**delta_timestamps, **dataset_cfg}

    return delta_timestamps if len(delta_timestamps) > 0 else None


def get_policy(
    policy_cfg: dict,
    dataset_metadata: LeRobotDatasetMetadata,
    img_size: tuple[int, int] | None = None,
    required_features: list[str] | None = None,
) -> PreTrainedPolicy:
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {
        key: ft
        for key, ft in features.items()
        if ft.type is FeatureType.ACTION and key in required_features
    }
    input_features = {
        key: ft
        for key, ft in features.items()
        if key not in output_features and key in required_features
    }
    cfg = get_policy_cfg(
        policy_cfg,
        input_features=input_features,
        output_features=output_features,
        img_size=img_size,
    )

    policy_class = get_policy_class(policy_cfg.TYPE)
    return policy_class(
        cfg, dataset_stats=fix_0std_dataset_stats(dataset_metadata.stats)
    )


def fix_0std_dataset_stats(
    dataset_stats: dict[str, dict[str, np.array]],
) -> dict[str, dict[str, np.array]]:
    for k, v in dataset_stats.items():
        for mean, (idx, std) in zip(v["mean"], enumerate(v["std"])):
            if abs(mean) < 1e-6 and abs(std) < 1e-6:
                logging.warning(
                    f"Dataset stats for {k} has zero mean and std. "
                    f"Setting std to 1.0 for index {idx}."
                )
                v["std"][idx] = 1.0

    return dataset_stats


def get_policy_class(policy_name: str) -> type[PreTrainedPolicy]:
    policy_classes = {
        "diffusion": DiffusionPolicy,
        "pi0": PI0Policy,
        "pi0fast": PI0FASTPolicy,
        "smolvla": SmolVLAPolicy,
    }
    if policy_name in policy_classes:
        return policy_classes[policy_name]
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def get_policy_features(features: dict[str, dict]) -> dict[str, FeatureType]:
    # Ref: lerobot.configs.types
    FEATURE_TYPES = {
        "STATE": FeatureType.STATE,
        "VISUAL": FeatureType.VISUAL,
        "ENV": FeatureType.ENV,
        "ACTION": FeatureType.ACTION,
        "REWARD": FeatureType.REWARD,
    }
    return {
        key: PolicyFeature(type=FEATURE_TYPES.get(ft["type"]), shape=ft.get("shape"))
        for key, ft in features.items()
    }


def get_policy_cfg(
    policy_cfg: dict = {},
    input_features: dict = {},
    output_features: dict = {},
    img_size: tuple[int, int] | None = None,
    cfg_file: pathlib.Path | str | None = None,
) -> PreTrainedConfig:
    if cfg_file is not None and os.path.exists(cfg_file):
        with open(cfg_file, "r") as f:
            policy_cfg = json.load(f)

        policy_cfg = easydict.EasyDict(policy_cfg)
        policy_cfg.TYPE = policy_cfg.type
        input_features = get_policy_features(policy_cfg.get("input_features"))
        output_features = get_policy_features(policy_cfg.get("output_features"))
        logging.info(
            f"Loaded policy configuration from {cfg_file} with input features:"
            f"{input_features} and output features: {output_features}"
        )

    if img_size is not None:
        for feature in input_features.values():
            if feature.type == FeatureType.VISUAL:
                feature.shape = (feature.shape[0], img_size[0], img_size[1])
        for feature in output_features.values():
            if feature.type == FeatureType.VISUAL:
                feature.shape = (feature.shape[0], img_size[0], img_size[1])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_class = None
    if policy_cfg.TYPE == "diffusion":
        cfg_class = DiffusionConfig
    elif policy_cfg.TYPE == "pi0":
        cfg_class = PI0Config
    elif policy_cfg.TYPE == "pi0fast":
        cfg_class = PI0FASTConfig
    elif policy_cfg.TYPE == "smolvla":
        cfg_class = SmolVLAConfig
    else:
        raise ValueError(f"Unknown policy: {policy_cfg.TYPE}")

    cfg = cfg_class(
        input_features=input_features,
        output_features=output_features,
        device=device,
    )
    for k, v in policy_cfg.items():
        attr_key = k.lower()
        if k in ["type", "device", "input_features", "output_features"]:
            continue
        if hasattr(cfg, attr_key) and v is not None:
            attr_value = getattr(cfg, attr_key)
            if attr_value != v:
                logging.warning(f"Overriding {k} in policy config: {attr_value} -> {v}")
                setattr(cfg, attr_key, v)

    return cfg
