# -*- coding: utf-8 -*-
#
# @File:   create_lerobot_dataset.py
# @Author: Haozhe Xie
# @Date:   2025-05-30 10:43:57
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-06-03 14:22:24
# @Email:  root@haozhexie.com
#
# Ref: https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py

import argparse
import json
import logging
import os
import pathlib
import shutil
import sys

import h5py
import lerobot.common.datasets.utils
import numpy as np
from huggingface_hub.constants import HF_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import utils.instruction_generator


def _get_cameras(scene_cfg):
    cameras = []
    for k, v in scene_cfg.items():
        if not k.endswith("_cam"):
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


def get_episode_metadata(episode_path):
    episode_dir = os.path.dirname(episode_path)
    episode_file = os.path.basename(episode_path)
    episode_name = os.path.splitext(episode_file)[0]

    tokens = episode_file.split("_")
    with open(os.path.join(episode_dir, "%s.json" % episode_name)) as f:
        env_cfg = json.load(f)

    return {
        "robot_type": tokens[1],
        "fps": round(1 / env_cfg["sim"]["dt"]),
        "cameras": _get_cameras(env_cfg["scene"]),
    }


def create_lerobot_dataset(repo_id, metadata):
    features = {
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "ee_pos_x",
                "ee_pos_y",
                "ee_pos_z",
                "ee_quat_w",
                "ee_quat_x",
                "ee_quat_y",
                "ee_quat_z",
                "gripper",
            ],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "ee_pos_x",
                "ee_pos_y",
                "ee_pos_z",
                "ee_quat_w",
                "ee_quat_x",
                "ee_quat_y",
                "ee_quat_z",
            ],
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (10,),
            "names": [
                "object_pos_x",
                "object_pos_y",
                "object_pos_z",
                "object_quat_qw",
                "object_quat_qx",
                "object_quat_qy",
                "object_quat_qz",
                "object_vel_x",
                "object_vel_y",
                "object_vel_z",
            ],
        },
    }
    for c in metadata["cameras"]:
        # for m in c["data_types"]:
        # TODO: Only RGB is supported in LeRobotDataset (wait for upstream support)
        for m in ["rgb"]:
            # Shorten the name for semantic segmentation
            m = "seg" if m == "semantic_segmentation" else m
            features["observation.images.%s_%s" % (c["name"], m)] = {
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

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=metadata["robot_type"],
        fps=metadata["fps"],
        features=features,
        image_writer_threads=32,
        image_writer_processes=8,
    )


def _get_task_instruction(episode_file_name):
    return utils.instruction_generator.InstructionGenerator.generate_instruction(
        filename=episode_file_name
    )


def get_episode_frames(episode_path):
    with h5py.File(episode_path, "r") as f:
        env_states = {k: f[k][()] for k in f.keys()}

    frames = []
    for i in range(len(env_states["action"])):
        _frame = {
            "action": env_states["action"][i],
            "task": _get_task_instruction(os.path.basename(episode_path)),
            "observation.state": np.concatenate(
                [
                    env_states["ee_pos"][i],
                    env_states["ee_quat"][i],
                ],
                axis=-1,
            ),
            "observation.environment_state": np.concatenate(
                [
                    env_states["object_pos"][i],
                    env_states["object_quat"][i],
                    env_states["object_vel"][i],
                ],
                axis=-1,
            ),
        }
        # Add camera frames
        for k, v in env_states.items():
            # if k.find("_cam_") == -1:
            # TODO: Only RGB is supported in LeRobotDataset (wait for upstream support)
            if k.find("_cam_rgb") == -1:
                continue
            _frame["observation.images.%s" % k] = v[i]

        frames.append(_frame)

    return frames


def main(input_dir, push_to_hub):
    REPO_ID = "hzxie/dynamic_objects"
    OUTPUT_DIR = os.path.join(HF_HOME, "lerobot", REPO_ID)

    logging.info("Creating LeRobot dataset from %s to %s" % (input_dir, OUTPUT_DIR))
    if os.path.exists(OUTPUT_DIR):
        logging.warning(
            "Output directory %s already exists. It will be overwritten." % OUTPUT_DIR
        )
        shutil.rmtree(OUTPUT_DIR)

    episodes = [f for f in os.listdir(input_dir) if f.endswith(".h5")]
    if not episodes:
        logging.error("No episodes found in the input directory: %s" % input_dir)
        sys.exit(2)

    dataset_metadata = get_episode_metadata(os.path.join(input_dir, episodes[0]))
    logging.info("Episode metadata: %s" % dataset_metadata)
    lerobot_dataset = create_lerobot_dataset(REPO_ID, dataset_metadata)
    logging.info("Dataset Overview: %s" % lerobot_dataset)

    for i, e in enumerate(tqdm(episodes)):
        if e.find(dataset_metadata["robot_type"]) == -1:
            continue

        try:
            _metadata = get_episode_metadata(os.path.join(input_dir, e))
            _frames = get_episode_frames(os.path.join(input_dir, e))
            for f in _frames:
                lerobot_dataset.add_frame(f)
        except Exception as ex:
            logging.exception(ex)
            continue

        lerobot_dataset.save_episode()
        # Manually save the camera parameters
        lerobot.common.datasets.utils.append_jsonlines(
            {"episode_idx": i, "filename": e, "cameras": _metadata["cameras"]},
            pathlib.Path(os.path.join(OUTPUT_DIR, "meta", "camera.jsonl")),
        )

    if push_to_hub:
        lerobot_dataset.push_to_hub(
            tags=["LeRobot", dataset_metadata["robot_type"], "synthetic", "dynamic"],
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
        "--dataset_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    args = parser.parse_args()

    main(args.dataset_dir, args.push_to_hub)
