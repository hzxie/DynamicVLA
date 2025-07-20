# -*- coding: utf-8 -*-
#
# @File:   helpers.py
# @Author: Haozhe Xie
# @Date:   2025-06-14 15:17:59
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-07-20 10:41:16
# @Email:  root@haozhexie.com

import json
import logging
import os
import pathlib

import av
import numpy as np
import scipy.spatial.transform
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
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
    assert quat.ndim == 2 and quat.shape[1] == 4
    if scalar_first:
        quat = quat[:, [1, 2, 3, 0]]  # wxyz to xyzw

    # Clamp w (quat[:, 3]) to [-1.0, 1.0]
    w = np.clip(quat[:, 3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)

    # Angle part in radians
    angles = 2.0 * np.arccos(w)
    # Avoid division by zero
    zero_mask = den < 1e-8

    # Normalize axis and multiply by angle
    axis = np.zeros_like(quat[:, :3])
    axis[~zero_mask] = quat[~zero_mask, :3] / den[~zero_mask, np.newaxis]
    return axis * angles[:, np.newaxis]


def get_quaternion(rotation_vector, format="rotvec", scalar_first=True):
    if format == "rotvec":
        return (
            scipy.spatial.transform.Rotation.from_rotvec(rotation_vector)
            .as_quat(scalar_first=scalar_first)
            .astype(np.float32)
        )
    elif format == "euler":
        return (
            scipy.spatial.transform.Rotation.from_euler(
                "xyz", rotation_vector, degrees=False
            )
            .as_quat(scalar_first=scalar_first)
            .astype(np.float32)
        )
    elif format == "quat":
        return rotation_vector.astype(np.float32)
    else:
        raise ValueError(
            "Unsupported format: %s. Use 'rotvec', 'euler', or 'quat'." % format
        )


def get_delta_timestamps(
    policy_name: str,
    chunk_size: int | None = None,
    dataset_cfg: dict[str, list[float]] | None = None,
) -> dict[str, list] | None:
    # Ref: lerobot.common.datasets.factorfvy.resolve_delta_timestamps
    policy_cfg = get_policy_cfg(policy_name, chunk_size=chunk_size)
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
    policy_name: str,
    dataset_metadata: LeRobotDatasetMetadata,
    img_size: tuple[int, int] | None = None,
    chunk_size: int | None = None,
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
    policy_cfg = get_policy_cfg(
        policy_name,
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        img_size=img_size,
    )

    policy_class = get_policy_class(policy_name)
    return policy_class(
        policy_cfg, dataset_stats=fix_0std_dataset_stats(dataset_metadata.stats)
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
    policy_name: str,
    input_features: dict = {},
    output_features: dict = {},
    img_size: tuple[int, int] | None = None,
    chunk_size: int | None = None,
    cfg_file: pathlib.Path | str | None = None,
) -> PreTrainedConfig:
    if cfg_file is not None and os.path.exists(cfg_file):
        with open(cfg_file, "r") as f:
            cfg_data = json.load(f)

        input_features = get_policy_features(cfg_data.get("input_features"))
        output_features = get_policy_features(cfg_data.get("output_features"))
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
    if policy_name == "diffusion":
        policy_cfg = DiffusionConfig(
            input_features=input_features,
            output_features=output_features,
            device=device,
        )
    elif policy_name == "pi0":
        policy_cfg = PI0Config(
            input_features=input_features,
            output_features=output_features,
            device=device,
        )
    elif policy_name == "pi0fast":
        policy_cfg = PI0FASTConfig(
            input_features=input_features,
            output_features=output_features,
            device=device,
        )
    elif policy_name == "smolvla":
        policy_cfg = SmolVLAConfig(
            input_features=input_features,
            output_features=output_features,
            device=device,
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    if chunk_size is not None:
        logging.info(
            "Setting chunk size to %d for policy %s." % (chunk_size, policy_name)
        )
        policy_cfg.chunk_size = chunk_size

    return policy_cfg


def dump_video(frames, output_path, fps=24):
    if len(frames) == 0:
        return

    # Ref: lerobot.common.datasets.video_utils.encode_video_frames
    with av.open(str(output_path), "w") as output:
        output_stream = output.add_stream(
            "libsvtav1", fps, options={"g": "2", "crf": "30"}
        )
        output_stream.pix_fmt = "yuv420p"
        output_stream.width = frames[0].shape[1]
        output_stream.height = frames[0].shape[0]
        # Loop through input frames and encode them
        for frame in frames:
            input_frame = av.VideoFrame.from_image(Image.fromarray(frame))
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)
        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)
