# -*- coding: utf-8 -*-
#
# @File:   create_lerobot_dataset.py
# @Author: Haozhe Xie
# @Date:   2025-05-30 10:43:57
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-08 18:37:45
# @Email:  root@haozhexie.com

import argparse
import logging
import math
import multiprocessing
import os
import pathlib
import shutil
import sys

import av
import h5py
import lerobot.constants
import lerobot.datasets.lerobot_dataset
import lerobot.datasets.utils
import numpy as np
import pandas as pd
import pyarrow
import torchcodec.decoders
from lerobot.datasets import compute_stats
from PIL import Image
from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import utils.helpers
import utils.instruction_generator


def get_episode_metadata(episode_path):
    episode_dir = os.path.dirname(episode_path)
    episode_file = os.path.basename(episode_path)
    episode_name = os.path.splitext(episode_file)[0]

    tokens = episode_file.split("_")
    env_cfg = lerobot.datasets.utils.load_json(
        os.path.join(episode_dir, "%s.json" % episode_name)
    )

    return {
        "robot_type": tokens[1],
        "fps": round(1 / env_cfg["sim"]["dt"]),
        "cameras": _get_cameras(env_cfg["scene"]),
    }


def _get_cameras(scene_cfg):
    cameras = []
    for k, v in scene_cfg.items():
        if (
            not isinstance(v, dict)
            or "class_type" not in v
            or not isinstance(v["class_type"], str)
            or not v["class_type"].startswith("isaaclab.sensors.camera")
        ):
            continue

        cameras.append(
            {
                "name": k,
                "width": v["width"],
                "height": v["height"],
                "offset": v["offset"],
                "data_types": v["data_types"],
                "focal": v["spawn"]["focal_length"],
            }
        )

    return cameras


def convert_lerobot_episodes(
    episode_path, episode_metadata, rot_fmt, fps, remove_source
):
    frames = get_episode_frames(episode_path, episode_metadata, rot_fmt)
    n_frames = len(frames)
    assert n_frames > 0, "No frames found in episode %s" % episode_path

    parquet_path = episode_path.replace(".h5", ".parquet")
    if not os.path.exists(parquet_path):
        _convert_episode_parquet(frames, fps, parquet_path)

    stat_path = episode_path.replace(".h5", ".stats")
    if not os.path.exists(stat_path):
        _convert_episode_stats(frames, stat_path)

    _convert_episode_videos(frames, fps, episode_path)

    if remove_source:
        os.remove(episode_path)


def _convert_episode_parquet(frames, fps, parquet_path):
    keys = frames[0].keys()
    n_frames = len(frames)

    df = {
        k: [frame[k] for frame in frames]
        for k in keys
        if not k.startswith("observation.image")
    }
    df["timestamp"] = [i / fps for i in range(n_frames)]
    df["frame_index"] = list(range(n_frames))
    table = pyarrow.Table.from_pandas(pd.DataFrame(df))
    pyarrow.parquet.write_table(table, parquet_path)


def _convert_episode_stats(frames, stats_path):
    keys = frames[0].keys()
    ep_stats = {}
    # Ref: lerobot.datasets.compute_stats.compute_episode_stats
    for key in keys:
        _frames = np.array([f[key] for f in frames])
        _axes_to_reduce = (0,)
        _keepdims = _frames.ndim == 1
        if key.startswith("observation.image"):
            # Shape: NHWC -> NCHW (Align with PyTorch DataLoader)
            _frames = np.transpose(_frames, (0, 3, 1, 2))
            _axes_to_reduce = (0, 2, 3)
            _keepdims = True

        ep_stats[key] = compute_stats.get_feature_stats(
            _frames, _axes_to_reduce, keepdims=_keepdims
        )
        if key.startswith("observation.image"):
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
                for k, v in ep_stats[key].items()
            }

    lerobot.datasets.utils.write_json(
        lerobot.datasets.utils.serialize_dict(ep_stats),
        pathlib.Path(stats_path),
    )


def _convert_episode_videos(frames, fps, episode_path):
    keys = frames[0].keys()
    n_frames = len(frames)

    for k in keys:
        if not k.startswith("observation.image"):
            continue

        video_path = episode_path.replace(".h5", ".%s.mp4" % k.split(".")[-1])
        if os.path.exists(video_path):
            continue

        while not is_video_valid(video_path, n_frames):
            dump_video([frame[k] for frame in frames], video_path, fps)


def get_episode_frames(episode_path, episode_metadata, rot_fmt="quat"):
    with h5py.File(episode_path, "r") as f:
        env_states = {k: f[k][()] for k in f.keys()}

    frames = []
    for i in range(len(env_states["action"])):
        _frame = {
            "action": np.concatenate(
                [
                    env_states["action"][i][:3],
                    utils.helpers.get_rotation_vector(
                        env_states["action"][i][3:-1], rot_fmt
                    ),
                    env_states["action"][i][-1:],
                ],
                axis=-1,
            ),
            "observation.state": np.concatenate(
                [
                    env_states["ee_pos"][i],
                    utils.helpers.get_rotation_vector(
                        env_states["ee_quat"][i], rot_fmt
                    ),
                ],
                axis=-1,
            ),
            "observation.environment_state": np.concatenate(
                [
                    env_states["object_pos"][i],
                    utils.helpers.get_rotation_vector(
                        env_states["object_quat"][i], rot_fmt
                    ),
                    env_states["object_vel"][i],
                ],
                axis=-1,
            ),
        }
        for cam in episode_metadata["cameras"]:
            # TODO: Only RGB is supported in LeRobotDataset (wait for upstream support)
            key = "%s_rgb" % cam["name"]
            assert key in env_states
            _frame["observation.images.%s" % cam["name"]] = env_states[key][i]

        frames.append(_frame)

    return frames


def dump_video(frames, output_path, fps):
    if len(frames) == 0:
        return

    # Ref: lerobot.datasets.video_utils.encode_video_frames
    with av.open(str(output_path), "w") as output:
        output_stream = output.add_stream(
            "libsvtav1", fps, options={"g": "2", "crf": "30"}
        )
        output_stream.pix_fmt = "yuv420p"
        output_stream.width = frames[0].shape[1]
        output_stream.height = frames[0].shape[0]
        for frame in frames:
            input_frame = av.VideoFrame.from_image(Image.fromarray(frame))
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        packet = output_stream.encode()
        if packet:
            output.mux(packet)


def is_video_valid(video_path, video_length):
    try:
        video_decoder = torchcodec.decoders.VideoDecoder(
            video_path,
            seek_mode="approximate",
        )
        video_decoder.get_frames_in_range(0, video_length)
    except Exception as ex:
        return False

    return True


def get_features(metadata, rot_fmt="quat"):
    action_fields = _get_fields(["ee_pos_", "ee_rot_", "gripper"], rot_fmt)
    state_fields = _get_fields(["ee_pos_", "ee_rot_"], rot_fmt)
    env_state_fields = _get_fields(
        ["object_pos_", "object_rot_", "object_vel_"], rot_fmt
    )
    features = {
        "action": {
            "dtype": "float32",
            "shape": (len(action_fields),),
            "names": action_fields,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_fields),),
            "names": state_fields,
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (len(env_state_fields),),
            "names": env_state_fields,
        },
    }
    for c in metadata["cameras"]:
        # TODO: Only RGB is supported in LeRobotDataset (wait for upstream support)
        for m in ["rgb"]:  # c["data_types"]
            # Shorten the name for semantic segmentation
            m = "seg" if m == "semantic_segmentation" else m
            features["observation.images.%s" % c["name"]] = {
                "dtype": "video",
                "names": ["height", "width", "channel"],
                "shape": (c["height"], c["width"], 3 if m == "rgb" else 1),
                "video_info": {
                    "has_audio": False,
                    "video.codec": "av1",
                    "video.fps": metadata["fps"],
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": m == "depth",
                },
            }

    return features


def _get_fields(prefixes, rot_fmt="quat"):
    fields = []
    for p in prefixes:
        if p.find("_pos_") != -1:
            fields.append(p + "x")
            fields.append(p + "y")
            fields.append(p + "z")
        elif p.find("_rot_") != -1 and rot_fmt == "quat":
            fields.append(p + "qw")
            fields.append(p + "qx")
            fields.append(p + "qy")
            fields.append(p + "qz")
        elif p.find("_rot_") != -1 and rot_fmt in ["euler", "rotvec"]:
            fields.append(p + "x")
            fields.append(p + "y")
            fields.append(p + "z")
        elif p.find("_vel_") != -1:
            fields.append(p + "x")
            fields.append(p + "y")
            fields.append(p + "z")
        else:
            fields.append(p)

    return fields


def get_task(episode_file_name, dataset_tasks):
    task_prompt = utils.instruction_generator.InstructionGenerator.generate_instruction(
        filename=episode_file_name
    )
    if task_prompt in dataset_tasks:
        task_index = dataset_tasks.index(task_prompt)
    else:
        task_index = len(dataset_tasks)
        dataset_tasks.append(task_prompt)

    return task_index, task_prompt


def save_lerobot_episodes(
    input_dir, output_dir, dataset_info, episode_name, task_index
):
    episode_index = dataset_info["total_episodes"]

    episode_chunk = episode_index // lerobot.datasets.utils.DEFAULT_CHUNK_SIZE
    # Videos
    _save_episode_videos(
        input_dir, output_dir, dataset_info, episode_name, episode_chunk, episode_index
    )
    # Parquet
    n_frames = _save_episode_parquet(
        input_dir,
        output_dir,
        dataset_info,
        episode_name,
        episode_chunk,
        episode_index,
        task_index,
    )
    # Metadata
    episode_stats = _get_episode_metadata(
        input_dir, episode_name, episode_index, task_index, n_frames, dataset_info
    )

    return episode_stats, n_frames


def _save_episode_videos(
    input_dir, output_dir, dataset_info, episode_name, episode_chunk, episode_index
):
    for key, value in dataset_info["features"].items():
        dtype = value["dtype"]
        if dtype != "video":
            continue

        cam_name = key.split(".")[-1]
        episode_video_path = os.path.join(
            output_dir,
            dataset_info["video_path"].format(
                episode_chunk=episode_chunk, episode_index=episode_index, video_key=key
            ),
        )
        os.makedirs(os.path.dirname(episode_video_path), exist_ok=True)
        shutil.copy(
            os.path.join(input_dir, episode_name + ".%s.mp4" % cam_name),
            episode_video_path,
        )


def _save_episode_parquet(
    input_dir,
    output_dir,
    dataset_info,
    episode_name,
    episode_chunk,
    episode_index,
    task_index,
):
    episode_data_path = os.path.join(
        output_dir,
        dataset_info["data_path"].format(
            episode_chunk=episode_chunk, episode_index=episode_index
        ),
    )
    os.makedirs(os.path.dirname(episode_data_path), exist_ok=True)
    df = pyarrow.parquet.read_table(
        os.path.join(input_dir, episode_name + ".parquet")
    ).to_pandas()
    df["episode_index"] = episode_index
    df["index"] = df["frame_index"] + dataset_info["total_frames"]
    df["task_index"] = task_index
    pyarrow.parquet.write_table(pyarrow.Table.from_pandas(df), episode_data_path)
    return len(df)


def _get_episode_metadata(
    input_dir, episode_name, episode_index, task_index, n_frames, dataset_info
):
    episode_stats = lerobot.datasets.utils.load_json(
        os.path.join(input_dir, episode_name + ".stats")
    )
    episode_stats["episode_index"] = _get_dataset_stats(
        episode_index, episode_index, n_frames
    )
    episode_stats["index"] = _get_dataset_stats(
        dataset_info["total_frames"],
        dataset_info["total_frames"] + n_frames,
        n_frames,
    )
    episode_stats["task_index"] = _get_dataset_stats(task_index, task_index, n_frames)

    return episode_stats


def _get_dataset_stats(min_value, max_value, count):
    return {
        "min": [min_value],
        "max": [max_value],
        "mean": [(min_value + max_value) / 2],
        "std": [1],  # Nobody cares about std in this case
        "count": [count],
    }


def get_dataset_info(dataset_info, episode_metadata, n_tasks):
    dataset_info["total_tasks"] = n_tasks
    dataset_info["total_videos"] = dataset_info["total_episodes"] * len(
        episode_metadata["cameras"]
    )
    dataset_info["total_chunks"] = math.ceil(
        dataset_info["total_episodes"] / dataset_info["chunks_size"]
    )
    dataset_info["splits"] = {"train": "0:%d" % dataset_info["total_episodes"]}

    for feature, values in dataset_info["features"].items():
        if values["dtype"] == "video":
            dataset_info["features"][feature]["info"] = {
                k: v
                for k, v in values["video_info"].items()
                if k not in ["video.is_depth_map"]
            }
            dataset_info["features"][feature]["info"].update(
                {
                    "video.height": values["shape"][0],
                    "video.width": values["shape"][1],
                    "video.channel": values["shape"][2],
                }
            )

    EXTRA_FEATURES = [
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ]
    for feature in EXTRA_FEATURES:
        if feature[0] not in dataset_info["features"]:
            dataset_info["features"][feature[0]] = {
                "dtype": feature[1],
                "shape": (1,),
                "names": None,
            }

    return dataset_info


def main(repo_id, input_dir, rot_fmt, remove_source, push_to_hub):
    output_dir = lerobot.constants.HF_LEROBOT_HOME / repo_id
    # Listing all episodes in the input directory
    episodes = sorted([f[:-3] for f in os.listdir(input_dir) if f.endswith(".h5")])
    if not episodes:
        logging.error("No episodes found in the input directory: %s" % input_dir)
        sys.exit(2)

    if os.path.exists(output_dir):
        answer = (
            input(
                "Output directory %s already exists. Do you want to overwrite? (y/N) "
                % output_dir
            )
            .strip()
            .lower()
        )
        if answer in ["y", "yes"]:
            shutil.rmtree(output_dir)
        else:
            sys.exit(0)

    # Creating the dataset in LeRobot format
    episode_metadata = get_episode_metadata(os.path.join(input_dir, "%s.h5" % episodes[0]))
    logging.info("Episode metadata: %s" % episode_metadata)

    # Converting the dataset (.h5) to LeRobot format
    logging.info("Converting episodes to LeRobot format...")
    # convert_lerobot_episodes(
    #     os.path.join(input_dir, episodes[0]),
    #     episode_metadata,
    #     rot_fmt,
    #     episode_metadata["fps"],
    #     remove_source,
    # )
    n_processes = max(1, multiprocessing.cpu_count() - 4)
    with multiprocessing.Pool(processes=n_processes) as pool:
        pool.starmap(
            convert_lerobot_episodes,
            [
                (
                    os.path.join(input_dir, "%s.h5" % e),
                    episode_metadata,
                    rot_fmt,
                    episode_metadata["fps"],
                    remove_source,
                )
                for e in episodes
            ],
        )

    logging.info("Creating LeRobot dataset...")
    dataset_info = lerobot.datasets.utils.create_empty_dataset_info(
        lerobot.datasets.lerobot_dataset.CODEBASE_VERSION,
        fps=episode_metadata["fps"],
        features=get_features(episode_metadata, rot_fmt=rot_fmt),
        use_videos=True,
        robot_type=episode_metadata["robot_type"],
    )
    dataset_tasks = []
    episode_length = []
    os.makedirs(os.path.join(output_dir, "meta"))

    episodes = sorted([f for f in os.listdir(input_dir) if f.endswith(".stats")])
    for episode in tqdm(episodes):
        episode = episode[:-6]  # Remove .stats extension
        if episode.find(episode_metadata["robot_type"]) == -1:
            logging.warning(
                "Skipping episode %s as it does not match the robot type %s"
                % (episode, episode_metadata["robot_type"])
            )
            continue

        task_index, task_prompt = get_task(episode, dataset_tasks)
        episode_stat, length = save_lerobot_episodes(
            input_dir, output_dir, dataset_info, episode, task_index
        )
        episode_length.append(length)

        # Manually save the metadata
        episode_index = dataset_info["total_episodes"]
        lerobot.datasets.utils.append_jsonlines(
            {
                "episode_index": episode_index,
                "filename": episode,
                "cameras": episode_metadata["cameras"],
            },
            pathlib.Path(os.path.join(output_dir, "meta", "camera.jsonl")),
        )
        lerobot.datasets.utils.append_jsonlines(
            {"episode_index": episode_index, "tasks": task_prompt, "length": length},
            pathlib.Path(os.path.join(output_dir, "meta", "episodes.jsonl")),
        )
        lerobot.datasets.utils.append_jsonlines(
            {"episode_index": episode_index, "stats": episode_stat},
            pathlib.Path(os.path.join(output_dir, "meta", "episodes_stats.jsonl")),
        )

        dataset_info["total_frames"] += length
        dataset_info["total_episodes"] += 1

    # Save dataset info and tasks
    lerobot.datasets.utils.write_json(
        get_dataset_info(dataset_info, episode_metadata, len(dataset_tasks)),
        pathlib.Path(os.path.join(output_dir, "meta", "info.json")),
    )
    for tid, task in enumerate(dataset_tasks):
        lerobot.datasets.utils.append_jsonlines(
            {"task_index": tid, "task": task},
            pathlib.Path(os.path.join(output_dir, "meta", "tasks.jsonl")),
        )

    if push_to_hub:
        lerobot_dataset = lerobot.datasets.lerobot_dataset.LeRobotDataset(
            repo_id=repo_id,
            root=output_dir,
        )
        lerobot_dataset.push_to_hub(
            tags=["LeRobot", episode_metadata["robot_type"], "synthetic", "dynamic"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Create LeRobot dataset")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="hzxie/dynamic-objects",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--rotation",
        type=str,
        default="euler",
    )
    parser.add_argument(
        "--remove-source",
        action="store_true",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    args = parser.parse_args()

    main(
        args.repo_id,
        args.dataset_dir,
        args.rotation,
        args.remove_source,
        args.push_to_hub,
    )
