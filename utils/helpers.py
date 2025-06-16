# -*- coding: utf-8 -*-
#
# @File:   helpers.py
# @Author: Haozhe Xie
# @Date:   2025-06-14 15:17:59
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-06-16 15:59:48
# @Email:  root@haozhexie.com

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType


def get_n_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_formatted_big_number(num, precision=0):
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num


def get_policy(dataset):
    dataset_metadata = LeRobotDatasetMetadata(dataset)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }
    # TODO: Handle different policies
    policy_cfg = DiffusionConfig(
        input_features=input_features, output_features=output_features
    )
    policy = DiffusionPolicy(policy_cfg, dataset_stats=dataset_metadata.stats)

    return policy
