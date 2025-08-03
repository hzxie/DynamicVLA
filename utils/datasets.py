# -*- coding: utf-8 -*-
#
# @File:   datasets.py
# @Author: Haozhe Xie
# @Date:   2025-06-17 16:10:33
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-08-02 08:55:12
# @Email:  root@haozhexie.com

import logging
import pathlib
import typing

import lerobot.constants
import lerobot.datasets.compute_stats
import lerobot.datasets.lerobot_dataset
import lerobot.datasets.utils
import numpy as np
import pyarrow.parquet as pq
import torch
import torchcodec.decoders
import torchvision.io
import torchvision.transforms.v2
import torchvision.transforms.v2.functional as F
from tqdm import tqdm


def get_dataset(
    dataset_name: str,
    split: str,
    pin_memory: bool,
    delta_action: bool,
    required_features: list[str] | None = None,
    image_transforms: typing.Callable | None = None,
    delta_timestamps: dict[str, list[float]] | None = None,
) -> torch.utils.data.Dataset:
    if dataset_name.startswith("lerobot/"):
        return LeRobotDataset(
            dataset_name[8:],  # Remove 'lerobot/' prefix
            split=split,
            pin_memory=pin_memory,
            delta_action=delta_action,
            required_features=required_features,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


class ImageTransforms:
    def __init__(self, img_size=None, cfg={}):
        self.img_size = img_size
        self.brightness = cfg.get("BRIGHTNESS")
        self.contrast = cfg.get("CONTRAST")
        self.saturation = cfg.get("SATURATION")
        self.hue = cfg.get("HUE")
        self.sharpness = cfg.get("SHARPNESS")

    def __call__(self, item, camera_keys: list[str]) -> dict:
        transform_args = torchvision.transforms.v2.ColorJitter.get_params(
            brightness=self.brightness if self.brightness else None,
            contrast=self.contrast if self.contrast else None,
            saturation=self.saturation if self.saturation else None,
            hue=self.hue if self.hue else None,
        )
        for k, v in item.items():
            if k not in camera_keys:
                continue

            item[k] = self._apply_transforms(v, transform_args)

        return item

    def _apply_transforms(self, image, transform_args):
        if self.img_size is not None:
            image = F.resize(image, self.img_size)

        for tr_idx in transform_args[0]:
            if tr_idx == 0 and transform_args[1] is not None:
                image = F.adjust_brightness(image, transform_args[1])
            elif tr_idx == 1 and transform_args[2] is not None:
                image = F.adjust_contrast(image, transform_args[2])
            elif tr_idx == 2 and transform_args[3] is not None:
                image = F.adjust_saturation(image, transform_args[3])
            elif tr_idx == 3 and transform_args[4] is not None:
                image = F.adjust_hue(image, transform_args[4])

        return image


class LeRobotDataset(torch.utils.data.Dataset):
    LEROBOT_VERSION = "v2.1"
    """
    A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...
    """

    def __init__(
        self,
        repo_id: str,
        root: str | pathlib.Path | None = None,
        split: str = "train",
        pin_memory: bool = False,
        delta_action: bool = True,
        required_features: list[str] | None = None,
        episodes: list[int] | None = None,
        image_transforms: typing.Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
    ):
        super().__init__()
        self._memcached = {}
        self.repo_id = repo_id
        self.root = (
            pathlib.Path(root) if root else lerobot.constants.HF_LEROBOT_HOME / repo_id
        )
        self.delta_action = delta_action
        self.required_features = required_features
        self.episodes = episodes
        self.image_transforms = image_transforms
        self.tolerance_s = tolerance_s
        self.delta_indices = None

        # Load metadata
        self.meta = lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata(
            self.repo_id, self.root
        )
        if self.episodes is None:
            self.episodes = [ep_idx for ep_idx in range(self.meta.total_episodes)]
            if split == "train":
                self.episodes = [e for e in self.episodes if e % 1000 != 0]
            elif split == "test":
                self.episodes = [e for e in self.episodes if e % 1000 == 0]
            else:
                raise ValueError(f"Unknown split {split}.")

        # Loading episode to RAM if pin_memory is True
        if pin_memory:
            logging.info(
                f"Loading {len(self.episodes)} episodes from the dataset {self.repo_id}"
                " into memory."
            )
            self._memcached = {
                ep_idx: self._get_episode(ep_idx) for ep_idx in tqdm(self.episodes)
            }

        # Compute dataset stats
        self.stats = lerobot.datasets.compute_stats.aggregate_stats(
            [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
        )
        # Load data indices
        self.episode_data_index = lerobot.datasets.utils.get_episode_data_index(
            self.meta.episodes, self.episodes
        )
        self.episode_data_index["length"] = (
            self.episode_data_index["to"] - self.episode_data_index["from"]
        )
        # Setup delta indices
        if delta_timestamps is not None:
            # Resolve delta timestamps for specific features
            ds_delta_timestamps = {}
            for key in self.meta.features:
                if key not in self.required_features:
                    continue
                # Ref: lerobot.datasets.factory.resolve_delta_timestamps
                if key == "next.reward" and "reward" in delta_timestamps:
                    ds_delta_timestamps[key] = [
                        i / self.meta.fps for i in delta_timestamps["reward"]
                    ]
                if key == "action" and "action" in delta_timestamps:
                    ds_delta_timestamps[key] = [
                        i / self.meta.fps for i in delta_timestamps["action"]
                    ]
                if key.startswith("observation.") and "observation" in delta_timestamps:
                    ds_delta_timestamps[key] = [
                        i / self.meta.fps for i in delta_timestamps["observation"]
                    ]

            self.delta_indices = lerobot.datasets.utils.get_delta_indices(
                ds_delta_timestamps, self.meta.fps
            )

    def __repr__(self) -> str:
        feature_keys = list(self.meta.features.keys())
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.meta.total_episodes}',\n"
            f"    Number of selected samples: '{self.meta.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    def __len__(self) -> int:
        return torch.sum(self.episode_data_index["length"]).item()

    def _get_episode(self, ep_idx: int) -> dict[str, torch.Tensor | bytes]:
        episode = {}
        # Load episode data
        data_frame = pq.read_table(
            self.root / self.meta.get_data_file_path(ep_idx)
        ).to_pandas()
        for col in data_frame.columns:
            if col in self.meta.image_keys:
                episode[col] = data_frame[col].values
            else:
                episode[col] = torch.tensor(np.stack(data_frame[col].values))

        # Load video data
        for video_key in self.meta.video_keys:
            episode[video_key] = self._get_video_bytes(ep_idx, video_key)

        return episode

    def _get_video_bytes(self, ep_idx: int, video_key: str) -> torch.Tensor:
        video_path = self.root / self.meta.get_video_file_path(ep_idx, video_key)
        with open(video_path, "rb") as f:
            return f.read()

    def __getitem__(self, idx) -> dict:
        ep_idx = self._get_episode_index(
            idx, self.episode_data_index["from"], self.episode_data_index["to"]
        )
        frame_idx = idx - self.episode_data_index["from"][ep_idx].item()

        g_ep_idx = self.episodes[ep_idx]
        episode = (
            self._memcached[g_ep_idx]
            if g_ep_idx in self._memcached
            else self._get_episode(g_ep_idx)
        )
        item = {
            k: v[frame_idx] for k, v in episode.items() if k in self.required_features
        }

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(
                frame_idx,
                self.episode_data_index["length"][ep_idx].item(),
                self.delta_indices,
            )
            query_result = self._query_episode(episode, query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.image_keys) > 0:
            for key in self.meta.image_keys:
                item[key] = (
                    torchvision.io.decode_image(
                        torch.frombuffer(
                            episode[key][frame_idx]["bytes"], dtype=torch.uint8
                        )
                    ).float()
                    / 255.0
                )

        if len(self.meta.video_keys) > 0:
            current_ts = episode["timestamp"][frame_idx].item()
            query_timestamps = self._get_query_timestamps(
                episode["timestamp"], current_ts, query_indices
            )
            video_frames = self._query_videos(episode, query_timestamps)
            item = {**item, **video_frames}

        if self.image_transforms is not None:
            item = self.image_transforms(item, self.meta.camera_keys)

        if self.delta_action:
            act_dim = item["action"].shape[-1] - 1
            # Only delta the XYZ and rotation components of the action
            item["action"][..., :act_dim] -= item["observation.state"][..., :act_dim]

        # Add task as a string
        task_idx = episode["task_index"][frame_idx].item()
        item["task"] = self.meta.tasks[task_idx]

        return item

    def _get_episode_index(
        self,
        frame_idx: int,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
    ) -> int:
        """
        Get the episode index for a given sample index.
        """
        ep_idx = torch.searchsorted(start_indices, frame_idx, right=True) - 1
        if ep_idx < 0 or frame_idx >= end_indices[ep_idx]:
            raise IndexError(f"Frame Index {frame_idx} is out of bounds.")

        return ep_idx.item()

    def _get_query_indices(
        self,
        frame_idx: int,
        ep_length: int,
        delta_indices: dict[str, list[int]],
    ) -> tuple[dict[str, list[int | bool]]]:
        # Ref: lerobot.datasets.lerobot_dataset._get_query_indices
        query_indices = {
            key: [max(0, min(ep_length - 1, frame_idx + delta)) for delta in delta_idx]
            for key, delta_idx in delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [
                    (frame_idx + delta < 0) | (frame_idx + delta >= ep_length)
                    for delta in delta_idx
                ]
            )
            for key, delta_idx in delta_indices.items()
        }
        return query_indices, padding

    def _query_episode(
        self,
        episode: dict[str, torch.Tensor | bytes],
        query_indices: dict[str, list[int]],
    ) -> dict:
        return {
            key: episode[key][q_idx]
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _get_query_timestamps(
        self,
        episode_ts: torch.Tensor,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        # Ref: lerobot.datasets.lerobot_dataset._get_query_timestamps
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                query_timestamps[key] = episode_ts[query_indices[key]].tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_videos(
        self,
        episode: dict[str, torch.Tensor | bytes],
        query_timestamps: dict[str, list[float]],
    ) -> dict[str, torch.Tensor]:
        # Ref: lerobot.datasets.lerobot_dataset._query_videos
        videos = {}
        for video_key, query_ts in query_timestamps.items():
            video_decoder = torchcodec.decoders.VideoDecoder(
                episode[video_key],
                seek_mode="approximate",
            )
            videos[video_key] = self._get_video_frames(video_decoder, query_ts).squeeze(
                0
            )

        return videos

    def _get_video_frames(
        self, video_decoder: torchcodec.decoders.VideoDecoder, timestamps: list[float]
    ) -> torch.Tensor:
        # Ref: lerobot.datasets.video_utils.decode_video_frames_torchcodec
        loaded_frames = []
        loaded_ts = []
        # convert timestamps to frame indices
        frame_indices = [round(ts * self.meta.fps) for ts in timestamps]
        # retrieve frames based on indices
        frames_batch = video_decoder.get_frames_at(indices=frame_indices)
        for frame, pts in zip(
            frames_batch.data, frames_batch.pts_seconds, strict=False
        ):
            loaded_frames.append(frame)
            loaded_ts.append(pts.item())

        query_ts = torch.tensor(timestamps)
        loaded_ts = torch.tensor(loaded_ts)

        # compute distances between each query timestamp and loaded timestamps
        dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
        min_, argmin_ = dist.min(1)
        assert (min_ < self.tolerance_s).all()

        # get closest frames to the query timestamps
        closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
        # closest_ts = loaded_ts[argmin_]

        # convert to float32 in [0,1] range (channel first)
        closest_frames = closest_frames.type(torch.float32) / 255.0
        assert len(timestamps) == len(closest_frames)
        return closest_frames
