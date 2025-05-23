# -*- coding: utf-8 -*-
#
# @File:   fix_error_scenes.py
# @Author: Haozhe Xie
# @Date:   2025-05-22 14:40:51
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-05-23 20:14:28
# @Email:  root@haozhexie.com

import argparse
import logging
import numpy as np
import os
import sys

import isaaclab.app
from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.dirname(__file__))


def contains_orphan_objects(stage):
    orphan_objects = []
    for prim in stage.GetPseudoRoot().GetChildren():
        if prim.GetPath().pathString.startswith("/house"):
            continue

        orphan_objects.append(prim)

    return orphan_objects


def get_new_prim_path(prim):
    from pxr import Sdf

    prim_type = prim.GetTypeName()
    if prim_type.endswith("Light"):
        return Sdf.Path("/house/lights/%s" % prim.GetName())
    else:
        return Sdf.Path("/house/furniture/%s" % prim.GetName())


def _repath_properties(layer, old_path, new_path):
    # Ref: https://gist.github.com/BigRoy/44250f5d9fdba79d127ce96e88bcc197
    from pxr import Sdf

    LIST_ATTRS = [
        "addedItems",
        "appendedItems",
        "deletedItems",
        "explicitItems",
        "orderedItems",
        "prependedItems",
    ]
    old_path_str = str(old_path)
    peformed_repath = False

    def replace_in_list(spec_list):
        """Replace paths in SdfTargetProxy or SdfConnectionsProxy"""
        for attr in LIST_ATTRS:
            entries = getattr(spec_list, attr)
            for i, entry in enumerate(entries):
                entry_str = str(entry)
                if entry == old_path or entry_str.startswith(old_path_str + "/"):
                    # Repath
                    entries[i] = Sdf.Path(
                        str(new_path) + entry_str[len(old_path_str) :]
                    )
                    peformed_repath = True

    def repath(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.RelationshipSpec):
            replace_in_list(spec.targetPathList)
        if isinstance(spec, Sdf.AttributeSpec):
            replace_in_list(spec.connectionPathList)

    # Repath any relationship pointing to this src prim path
    layer.Traverse("/", repath)
    return peformed_repath


def rename_prim(layer, src_path, dst_path):
    from pxr import Sdf

    reparent_edit = Sdf.NamespaceEdit.ReparentAndRename(
        src_path, dst_path.GetParentPath(), dst_path.name, -1
    )
    edit = Sdf.BatchNamespaceEdit()
    edit.Add(reparent_edit)
    if not layer.Apply(edit) and layer.GetPrimAtPath(src_path):
        return False

    _repath_properties(layer, src_path, dst_path)
    return True


def _get_duplicated_prim(stage, prim):
    for _prim in stage.Traverse():
        if _prim.GetName() == prim.GetName():
            return _prim

    return None


def contains_floating_objects(stage):
    # Represended by 71f2ab3f-b9de-4d34-a58c-f4e48e70ccd7.usd
    floating_objects = []
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if not prim_path.startswith("/house/lights/"):
            continue

        # Detect floating objects (that are not lights)
        if not prim_path.startswith("/house/lights/Lighting"):
            dup_prim = _get_duplicated_prim(stage, prim)
            floating_objects.append(prim)
            if dup_prim is not None:
                floating_objects.append(dup_prim)

    return floating_objects


def contains_invisible_prims(stage):
    # Represented by 3ed8823a-0603-49ed-9a51-302cf15adf53.usd
    from pxr import UsdGeom

    invisible_prims = []
    for prim in stage.Traverse():
        # Remain ceiling primitives invisible
        if prim.GetPath().pathString.startswith("/house/ceiling"):
            continue

        imageable = UsdGeom.Imageable(prim)
        visibility = imageable.GetVisibilityAttr().Get()
        if visibility == UsdGeom.Tokens.invisible:
            invisible_prims.append(prim)

    return invisible_prims


def main(scene_dir):
    from pxr import Sdf, Usd, UsdGeom, UsdUtils

    usd_files = [f for f in os.listdir(scene_dir) if f.endswith(".usd")]
    for uf in tqdm(usd_files):
        usd_file = os.path.join(scene_dir, uf)
        modified = False
        stage = Usd.Stage.Open(usd_file, load=Usd.Stage.LoadAll)
        prim_names = [str(p.GetPath()) for p in stage.TraverseAll()]
        assert "/house" in prim_names

        orphan_objects = contains_orphan_objects(stage)
        if orphan_objects:
            modified = True
            layers = stage.GetLayerStack()
            layer = next(
                l for l in layers if l.identifier == stage.GetRootLayer().identifier
            )
            for oo in orphan_objects:
                src_path = oo.GetPath()
                dst_path = get_new_prim_path(oo)
                if not rename_prim(layer, src_path, dst_path):
                    logging.warning("Failed to rename %s to %s" % (src_path, dst_path))

        floating_objects = contains_floating_objects(stage)
        if floating_objects:
            modified = True
            logging.info("Floating objects[%s] found in %s" % (floating_objects, uf))
            for fo in floating_objects:
                if not stage.RemovePrim(fo.GetPath()):
                    logging.warning(
                        "Failed to remove floating object %s" % (fo.GetPath())
                    )

        invisible_prims = contains_invisible_prims(stage)
        if invisible_prims:
            modified = True
            logging.info("Invisible primitives[%s] found in %s" % (invisible_prims, uf))
            for ip in invisible_prims:
                UsdGeom.Imageable(ip).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)

        if modified:
            stage.GetRootLayer().Save()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Isaac Simulation Runner")
    # Arguments for the IsaacLab
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save the data from camera at index specified by ``--camera_id``.",
    )
    # IssacSim Environment Initialization
    isaaclab.app.AppLauncher.add_app_launcher_args(parser)
    isaaclab_args, script_args = parser.parse_known_args()
    app_launcher = isaaclab.app.AppLauncher(isaaclab_args)

    # Arguments for the script
    parser.add_argument(
        "--scene_dir", default=os.path.join(PROJECT_HOME, os.pardir, "scenes")
    )
    args = parser.parse_args(script_args)

    main(args.scene_dir)
    app_launcher.app.close()
