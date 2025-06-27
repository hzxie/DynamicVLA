# -*- coding: utf-8 -*-
#
# @File:   fix_lerobot_videos.py
# @Author: Haozhe Xie
# @Date:   2025-06-25 18:53:11
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-06-27 14:42:49
# @Email:  root@haozhexie.com

import argparse
import logging
import os
import pathlib

import av
import h5py
import huggingface_hub.constants
import lerobot.common.datasets.lerobot_dataset
import lerobot.common.datasets.utils
import torchcodec.decoders
from PIL import Image
from tqdm import tqdm


def fix_lerobot_video(episode_index, video_path, video_length, h5_dir, lerobot_dir):
    episode_info = lerobot.common.datasets.utils.load_json(
        pathlib.Path(os.path.join(lerobot_dir, "meta", "info.json"))
    )
    episode_meta = lerobot.common.datasets.utils.load_jsonlines(
        pathlib.Path(os.path.join(lerobot_dir, "meta", "camera.jsonl"))
    )
    episode_meta = next(em for em in episode_meta if em["episode_idx"] == episode_index)
    with h5py.File(os.path.join(h5_dir, episode_meta["filename"]), "r") as f:
        env_states = {k: f[k][()] for k in f.keys()}

    CAM_KEY = "side_cam_rgb"
    assert env_states[CAM_KEY].shape[0] == video_length
    # Ref: lerobot.common.datasets.video_utils.encode_video_frames
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(
            "libsvtav1", episode_info["fps"], options={"g": "2", "crf": "30"}
        )
        output_stream.pix_fmt = "yuv420p"
        output_stream.width = env_states[CAM_KEY].shape[2]
        output_stream.height = env_states[CAM_KEY].shape[1]
        # Loop through input frames and encode them
        for image in env_states[CAM_KEY]:
            input_frame = av.VideoFrame.from_image(Image.fromarray(image))
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)
        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)


def main(repo_id, h5_dir):
    output_dir = os.path.join(huggingface_hub.constants.HF_HOME, "lerobot", repo_id)
    lerobot_dataset = lerobot.common.datasets.lerobot_dataset.LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
    )

    video_features = [
        k for k, v in lerobot_dataset.features.items() if v["dtype"] == "video"
    ]
    video_chunks = sorted(os.listdir(os.path.join(output_dir, "videos")))
    for vc in video_chunks:
        for vf in video_features:
            videos = sorted(os.listdir(os.path.join(output_dir, "videos", vc, vf)))
            for v in tqdm(
                videos, desc="Checking videos in %s/%s" % (vc, vf), leave=False
            ):
                episode_index = int(v.rsplit("_")[-1][:-4])
                if episode_index not in [1523, 3979, 6586, 6669, 9085, 9627]:
                    continue

                video_path = os.path.join(output_dir, "videos", vc, vf, v)
                video_length = lerobot_dataset.meta.episodes[episode_index]["length"]
                try:
                    video_decoder = torchcodec.decoders.VideoDecoder(
                        video_path,
                        seek_mode="approximate",
                    )
                    video_decoder.get_frames_in_range(0, video_length - 1)
                except Exception as e:
                    logging.warning(
                        "Failed to decode video %s in chunk %s/%s: %s",
                        v,
                        vc,
                        vf,
                        str(e),
                    )
                    fix_lerobot_video(
                        episode_index, video_path, video_length, h5_dir, output_dir
                    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=str,
        default="hzxie/dynamic_objects",
    )
    parser.add_argument(
        "--h5_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    main(args.repo, args.h5_dir)
