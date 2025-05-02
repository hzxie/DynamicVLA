# -*- coding: utf-8 -*-
#
# @File:   helpers.py
# @Author: Haozhe Xie
# @Date:   2025-05-02 19:01:00
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-05-02 21:03:18
# @Email:  root@haozhexie.com

import cv2
import numpy as np
import scipy.spatial.transform


def get_curr_frame(env_state, state_keys=[], env_id=0):
    MAX_DEPTH = 25

    frames = []
    cameras = set()
    env_state = env_state[env_id]
    for k, v in env_state.items():
        cam_idx = k.find("_cam_")
        if cam_idx == -1:
            continue

        cameras.add(k[:cam_idx])
        img_name = k[cam_idx + 5 :]
        if v.ndim == 2:
            frame = np.repeat(v[:, :, None], 3, axis=-1)
        elif v.shape[-1] == 1:
            frame = np.repeat(v[:, :, :], 3, axis=-1)
        elif v.shape[-1] >= 3:
            frame = v[:, :, :3]
        else:
            raise ValueError(f"Unknown camera data shape: {v.shape}")

        # Normalize the depth image to 0-255
        if img_name in ["depth", "distance_to_image_plane", "distance_to_camera"]:
            frame = np.clip(frame, 0, MAX_DEPTH)
            frame = (frame / np.max(frame) * 255).astype(np.uint8)
        if img_name in ["seg", "semantic_segmentation"]:
            # Assign a color to each semantic class
            frame = cv2.applyColorMap(frame * 64, cv2.COLORMAP_JET)

        frames.append(frame[..., ::-1])

    n_cameras, n_images = len(cameras), len(frames) // len(cameras)
    frames = np.vstack(
        [np.hstack(frames[i * n_images : (i + 1) * n_images]) for i in range(n_cameras)]
    )
    # if state_keys:
    #     frames = _print_state_on_frame(
    #         frames, {k: v for k, v in env_state.items() if k in state_keys}
    #     )

    return frames


def _print_state_on_frame(frame, state):
    TEXT_MARGIN = 10
    TEXT_SCALE = 0.5
    TEXT_THICKNESS = 1
    TEXT_COLOR = (255, 255, 255)
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

    lines = _get_state_text(state).split("\n")
    img_height, img_width = frame.shape[:2]
    # Print the text on the image
    y = TEXT_MARGIN
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(
            line, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS
        )
        x = img_width - text_width - TEXT_MARGIN
        frame = cv2.putText(
            np.ascontiguousarray(frame),
            line,
            (x, y + text_height),
            TEXT_FONT,
            TEXT_SCALE,
            TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )
        y += text_height + TEXT_MARGIN

    return frame


def _get_state_text(state):
    text = ""
    for k, v in state.items():
        k = k.replace("_", " ").title()
        if type(v) == int:
            text += "%s: %d\n" % (k, v)
        elif type(v) == float:
            text += "%s: %.3f\n" % (k, v)
        elif type(v) == np.ndarray:
            # Convert all quaternions to Euler angles
            if k.find("Quat") != -1:
                k = k.replace("Quat", "Rot")
                v = scipy.spatial.transform.Rotation.from_quat(v).as_euler(
                    "xyz", degrees=True
                )

            text += "%s: " % k
            text += " ".join(["%.3f" % i for i in v]) + "\n"
        else:
            raise ValueError(f"Unknown State Value Type: {type(v)}")

    return text
