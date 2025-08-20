# -*- coding: utf-8 -*-
#
# @File:   update_usd_tex_loc.py
# @Author: Haozhe Xie
# @Date:   2025-05-05 10:12:39
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-08-20 06:50:20
# @Email:  root@haozhexie.com


import argparse
import logging
import os
import shutil
import sys

from omni.isaac.kit import SimulationApp
from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.dirname(__file__))


def update_texture_location(usd_file_path, old_texture_dir, new_texture_dir):
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(usd_file_path)
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue

        shader = UsdShade.Shader(prim)
        # Check if it's a texture node
        if shader.GetIdAttr().Get() == "UsdUVTexture":
            file_input = shader.GetInput("file")
            old_texture_path = file_input.Get().path
            assert old_texture_path.startswith(
                old_texture_dir
            ), "Texture path: %s does not start with %s" % (
                old_texture_path,
                old_texture_dir,
            )
            new_texture_path = old_texture_path.replace(
                old_texture_dir, new_texture_dir
            )
            # For 3D-FUTURE
            # new_texture_path = new_texture_path.replace("/texture.png", ".png")
            # For Objaverse
            # new_texture_path = (
            #     "./texture/%s.jpg" % os.path.basename(usd_file_path).split(".")[0]
            # )
            logging.debug(
                "Updated texture path from %s to %s"
                % (old_texture_path, new_texture_path)
            )
            shutil.copyfile(
                old_texture_path,
                os.path.join(os.path.dirname(usd_file_path), new_texture_path),
            )
            file_input.Set(new_texture_path)

    stage.GetRootLayer().Save()


def main(usd_dir, old_texture_dir, new_texture_dir):
    usd_files = [f for f in os.listdir(usd_dir) if f.endswith(".usd")]
    for uf in tqdm(usd_files):
        update_texture_location(
            os.path.join(usd_dir, uf), old_texture_dir, new_texture_dir
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    # IssacSim Environment Initialization
    app = SimulationApp({"headless": True})

    # Arguments for the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd_dir", required=True)
    parser.add_argument("--old_texture_dir", required=True)
    parser.add_argument("--new_texture_dir", default="./texture")
    args = parser.parse_args()

    main(args.usd_dir, args.old_texture_dir, args.new_texture_dir)
    app.close()
