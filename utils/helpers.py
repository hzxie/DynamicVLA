# -*- coding: utf-8 -*-
#
# @File:   helpers.py
# @Author: Haozhe Xie
# @Date:   2025-06-14 15:17:59
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-06-19 14:34:38
# @Email:  root@haozhexie.com

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType


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


def get_delta_timestamps(
    policy_name: str, dataset_cfg: dict[str, list[float]] | None = None
) -> dict[str, list] | None:
    # Ref: lerobot.common.datasets.factory.resolve_delta_timestamps
    policy_cfg = get_policy_cfg(policy_name)
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
    )

    if policy_name == "diffusion":
        policy = DiffusionPolicy(policy_cfg, dataset_stats=dataset_metadata.stats)
    elif policy_name == "pi0":
        policy = PI0Policy(policy_cfg, dataset_stats=dataset_metadata.stats)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    return policy


def get_policy_cfg(
    policy_name: str, input_features: dict = {}, output_features: dict = {}
) -> PreTrainedConfig:
    if policy_name == "diffusion":
        policy_cfg = DiffusionConfig(
            input_features=input_features, output_features=output_features
        )
    elif policy_name == "pi0":
        policy_cfg = PI0Config(
            input_features=input_features, output_features=output_features
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    return policy_cfg
