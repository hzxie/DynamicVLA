# -*- coding: utf-8 -*-
#
# @File:   create_usd_objects.py
# @Author: Haozhe Xie
# @Date:   2025-04-12 13:42:34
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-05-14 20:12:18
# @Email:  root@haozhexie.com

import argparse
import logging
import os
import sys

from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import utils.maya_controller


def main(maya_ctl, input_dir, output_dir, categories):
    # Load the object directories
    category_dirs = [
        os.path.join(input_dir, c)
        for c in os.listdir(input_dir)
        if c in categories and os.path.isdir(os.path.join(input_dir, c))
    ]
    object_dirs = [
        os.path.join(cd, od)
        for cd in category_dirs
        for od in os.listdir(cd)
        if os.path.isdir(os.path.join(cd, od))
    ]

    for od in tqdm(object_dirs):
        output_file_path = os.path.join(output_dir, "%s.usd" % os.path.basename(od))
        # if os.path.exists(output_file_path):
        #     continue

        # Initialize a new scene
        maya_ctl.set_new_scene()
        # Load the object
        input_file_path = os.path.join(od, "visual", "model_normalized_0.obj")
        maya_ctl.send_python_command(
            "cmds.file('%s', i=True, type='OBJ')" % input_file_path.replace("\\", "/")
        )
        maya_ctl.send_python_command("cmds.rename('Mesh', 'mesh')")
        maya_ctl.send_python_command("cmds.group(em=True, name='object')")
        maya_ctl.send_python_command("cmds.parent('mesh', 'object')")
        # Smooth the object
        # maya_ctl.send_python_command("cmds.polySmooth('geometry', mth=0, dv=2)")
        # Output the object to USD
        maya_ctl.send_python_command(
            "cmds.file('%s', force=True, options='exportUVs=1', type='USD Export', exportAll=True)"
            % output_file_path.replace("\\", "/")
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.DEBUG,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", default=os.path.join(PROJECT_HOME, os.pardir, "Objaverse")
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(PROJECT_HOME, os.pardir, "USD")
    )
    parser.add_argument(
        "--categories", default=["apple", "avocado"], nargs="+", type=str
    )
    parser.add_argument("--port", default=12345, type=int)
    args = parser.parse_args()

    logging.info(
        "Run the following command in Maya's Script Editor to start the server."
    )
    logging.info(
        "  import maya.cmds as cmds; cmds.commandPort(n='localhost:%d')" % args.port
    )
    answer = input("Is your Maya server running? (Y/n) ").strip().lower()
    if answer not in ["y", "yes", ""]:
        logging.error("Please start the Maya server first.")
        sys.exit(1)

    # Connect to Maya server
    maya_ctl = utils.maya_controller.MayaController(port=args.port)
    maya_ctl.send_python_command("from maya.api import OpenMaya")
    logging.info("Connected to Maya Server.")
    # Load the Maya USD plugin
    maya_ctl.send_python_command("cmds.loadPlugin('mayaUsdPlugin')")
    if not maya_ctl.send_python_command(
        "cmds.pluginInfo('mayaUsdPlugin', query=True, loaded=True)"
    ).startswith("1"):
        logging.error("Failed to load the Maya USD plugin.")
        sys.exit(1)

    main(maya_ctl, args.input_dir, args.output_dir, args.categories)
