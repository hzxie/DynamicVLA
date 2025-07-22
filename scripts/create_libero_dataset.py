# -*- coding: utf-8 -*-
#
# @File:   create_libero_dataset.py
# @Author: Physical Intelligence Team
# @Date:   2025-07-11 14:06:55
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-07-23 06:44:51
# @Email:  root@haozhexie.com

"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
python scripts/create_libero_dataset.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
python scripts/create_libero_dataset.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import os
import shutil

import lerobot.constants
import lerobot.datasets.lerobot_dataset
import tensorflow_datasets as tfds
import tyro
from tqdm import tqdm

# Name of the output dataset, also used for the Hugging Face Hub
REPO_NAME = "hzxie/libero"
# For simplicity we will combine multiple Libero datasets into one training dataset
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]


def main(data_dir: str, *, push_to_hub: bool = False):
    output_path = lerobot.constants.HF_LEROBOT_HOME / REPO_NAME
    # Clean up any existing dataset in the output directory
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = lerobot.datasets.lerobot_dataset.LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "observation.images.base": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],
            },
        },
        image_writer_threads=32,
        image_writer_processes=32,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in tqdm(raw_dataset):
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "observation.images.base": step["observation"]["image"],
                        "observation.images.wrist": step["observation"]["wrist_image"],
                        "observation.state": step["observation"]["state"],
                        "action": step["action"],
                    },
                    task=step["language_instruction"].decode(),
                )
            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
