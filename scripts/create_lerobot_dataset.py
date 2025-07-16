# -*- coding: utf-8 -*-
#
# @File:   create_lerobot_dataset.py
# @Author: Haozhe Xie
# @Date:   2025-05-30 10:43:57
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-07-15 11:05:56
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
import lerobot.common.constants
import lerobot.common.datasets.lerobot_dataset
import lerobot.common.datasets.utils
import numpy as np
import torchcodec.decoders
from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import utils.helpers
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


def create_lerobot_dataset(repo_id, metadata, rot_fmt="quat"):
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

    return lerobot.common.datasets.lerobot_dataset.LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=metadata["robot_type"],
        fps=metadata["fps"],
        features=features,
        image_writer_threads=32,
        image_writer_processes=32,
    )


def get_task_instruction(episode_file_name):
    return utils.instruction_generator.InstructionGenerator.generate_instruction(
        filename=episode_file_name
    )


def _get_delta_action(curr_action, curr_state):
    return np.concatenate(
        [
            curr_action[:6] - curr_state[:6],
            curr_action[6:],
        ]
    ).astype(np.float32)


def get_episode_frames(episode_path, rot_fmt="quat"):
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

        for k, v in env_states.items():
            # TODO: Only RGB is supported in LeRobotDataset (wait for upstream support)
            if not k.endswith("_cam_rgb"):
                continue

            cam_name = k[:-4]  # Remove the "_rgb" suffix
            _frame["observation.images.%s" % cam_name] = v[i]

        frames.append(_frame)

    return frames


def is_video_valid(video_path, video_length):
    try:
        video_decoder = torchcodec.decoders.VideoDecoder(
            video_path,
            seek_mode="approximate",
        )
        video_decoder.get_frames_in_range(0, video_length - 1)
    except Exception as ex:
        return False

    return True


def main(repo_id, input_dir, rot_fmt, push_to_hub):
    output_dir = lerobot.common.constants.HF_LEROBOT_HOME / repo_id
    # Listing all episodes in the input directory
    episodes = sorted([f for f in os.listdir(input_dir) if f.endswith(".h5")])[:5000]
    if not episodes:
        logging.error("No episodes found in the input directory: %s" % input_dir)
        sys.exit(2)

    # Check whether the output directory exists
    overwrite = True
    existing_episodes = []
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
            overwrite = False
            existing_episodes = [
                ep["filename"]
                for ep in lerobot.common.datasets.utils.load_jsonlines(
                    pathlib.Path(os.path.join(output_dir, "meta", "camera.jsonl"))
                )
            ]

    # Creating the dataset in LeRobot format
    episode_metadata = get_episode_metadata(os.path.join(input_dir, episodes[0]))
    logging.info("Episode metadata: %s" % episode_metadata)
    if overwrite:
        logging.info("Creating LeRobot dataset from %s to %s" % (input_dir, output_dir))
        lerobot_dataset = create_lerobot_dataset(repo_id, episode_metadata, rot_fmt)
        logging.info("Dataset Overview: %s" % lerobot_dataset)
    else:
        lerobot_dataset = lerobot.common.datasets.lerobot_dataset.LeRobotDataset(
            repo_id=repo_id,
            root=output_dir,
        )

    ep_idx = len(existing_episodes)
    for e in tqdm(episodes):
        if e.find(episode_metadata["robot_type"]) == -1:
            logging.warning(
                "Skipping episode %s as it does not match the robot type %s"
                % (e, episode_metadata["robot_type"])
            )
            continue
        if e in existing_episodes:
            logging.warning(
                "Skipping episode %s as it already exists in the dataset" % e
            )
            continue

        try:
            _metadata = get_episode_metadata(os.path.join(input_dir, e))
            _frames = get_episode_frames(os.path.join(input_dir, e), rot_fmt)
            _task = get_task_instruction(os.path.basename(e))
            for f in _frames:
                lerobot_dataset.add_frame(f, _task)
        except Exception as ex:
            logging.exception(ex)
            continue

        lerobot_dataset.save_episode()

        # Manually save the camera parameters
        lerobot.common.datasets.utils.append_jsonlines(
            {"episode_idx": ep_idx, "filename": e, "cameras": _metadata["cameras"]},
            pathlib.Path(os.path.join(output_dir, "meta", "camera.jsonl")),
        )
        ep_idx += 1

    if push_to_hub:
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
        default="quat",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    args = parser.parse_args()

    main(args.repo_id, args.dataset_dir, args.rotation, args.push_to_hub)
