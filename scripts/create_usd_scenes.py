# -*- coding: utf-8 -*-
#
# @File:   create_usd_scenes.py
# @Author: Haozhe Xie
# @Date:   2025-03-20 14:41:09
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-03-31 14:31:09
# @Email:  root@haozhexie.com
#
# References:
# - arnold-benchmark/Usdify/front.py

import argparse
import json
import logging
import math
import os
import sys

import numpy as np
from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import helpers.maya


def _get_furniture_category(model):
    super_category = model["super-category"]
    category = model["category"]
    if super_category.find("/") == -1:
        return super_category
    else:
        return category.split("/")[0].strip()


def _get_instances(definition):
    instances = {}
    for r_id, room in enumerate(definition["scene"]["room"]):
        for c in room["children"]:
            # Example: furniture/102
            inst_id = c["instanceid"]
            if inst_id in instances:
                # logging.warning("Duplicate instance ID: %s", inst_id)
                continue

            instances[inst_id] = {
                "room_id": r_id,
                # Example: [0.0, 0.0, 0.0]
                "position": c["pos"],
                # Example: [0.0, 0.0, 0.0, 1.0]
                "rotation": c["rot"],
                # Example: [1.33, 1.0, 1.0]
                "scale": c["scale"],
                # Example: 32210/model
                "ref": c["ref"],
            }
    return instances


def _get_meshes(definition):
    meshes = {}
    for mesh in definition["mesh"]:
        # Example: 440721587242490579/0
        mesh_id = mesh["uid"]
        if mesh_id in meshes:
            # logging.warning("Duplicate mesh ID: %s", mesh_id)
            continue

        meshes[mesh_id] = {
            # Example: sge/a2d21b8d-50d4-408a-9982-5036982b8c6d/4850
            "material_id": mesh["material"],
            # Example: WallInner
            "type": mesh["type"],
            # Example of the following attributes: 1D array [1, 2, ...]
            "vertices": np.array(mesh["xyz"], dtype=np.float32).reshape(-1, 3),
            "faces": np.array(mesh["faces"], dtype=np.int32).reshape(-1, 3),
            "normal": np.array(mesh["normal"], dtype=np.float32).reshape(-1, 3),
            "uv": np.array(mesh["uv"], dtype=np.float32).reshape(-1, 2),
        }

    return meshes


def _get_material_color_value(color):
    if not color:  # Fix color values equals to []
        r, g, b, a = (255,) * 4
    else:
        r, g, b, a = color if len(color) == 4 else [*color, 255]
    return a << 24 | r << 16 | g << 8 | b


def _get_material_color_array(color):
    _color_array = [
        (color & 0x00FF0000) >> 16,
        (color & 0x0000FF00) >> 8,
        (color & 0x000000FF) >> 0,
        (color & 0xFF000000) >> 24,
    ]
    return tuple(map(lambda e: e / 255, _color_array))


def _get_material_texture(material):
    if "texture" not in material:
        return ""

    texture = material["texture"]
    return texture["value"] if isinstance(texture, dict) else texture


def _get_material_color_mode(material):
    if not bool(material.get("texture")):
        return "color"
    if "colorMode" in material:
        return material["colorMode"]
    if bool(material.get("useColor")):
        return "color"

    return "texture"


def _get_material_uv_transform(material):
    return (
        np.array(material["UVTransform"]).reshape(3, 3)
        if "UVTransform" in material
        else np.eye(3)
    )


def _get_materials(definition):
    materials = {}
    for material in definition["material"]:
        # Example: 440721587242490579/044072
        material_id = material["uid"]
        if material_id in materials:
            # logging.warning("Duplicate material ID: %s", material_id)
            continue

        materials[material_id] = {
            # Example: [255, 255, 255, 255]
            "color": _get_material_color_value(material["color"]),
            # Example: ""
            "texture": _get_material_texture(material),
            # Example: ""
            "color_mode": _get_material_color_mode(material),
            # Example: 1D array [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            "uv_transform": _get_material_uv_transform(material),
        }

    return materials


def _get_furnitures(definition, furniture_categories):
    furnitures = {}
    for furniture in definition["furniture"]:
        # Example: 32210/model
        furniture_id = furniture["uid"]
        if furniture_id in furnitures:
            # logging.warning("Duplicate furniture ID: %s", furniture_id)
            continue

        furnitures[furniture_id] = {
            # Example: "a3017175-01da-4bbb-a3f4-aa896e3fa604"
            "jid": furniture["jid"],
            # Example: Chair
            "category": furniture_categories.get(furniture["jid"], "unknown"),
        }

    return furnitures


def get_house_layout(definition, furniture_categories):
    instances = _get_instances(definition)
    meshes = _get_meshes(definition)
    materials = _get_materials(definition)
    furnitures = _get_furnitures(definition, furniture_categories)
    return {
        "instances": instances,
        "meshes": meshes,
        "materials": materials,
        "furnitures": furnitures,
    }


def _get_shader(maya_ctl, material):
    if material["color_mode"] != "color":
        logging.warning("Fallback to color shader for unknown material: %s" % material)
        material["color"] = 4294967295

    _get_shader.colors = _get_shader.colors if hasattr(_get_shader, "colors") else {}

    shader_name = "color_shader_%s" % material["color"]
    # Check if the shader has been created
    if shader_name in _get_shader.colors:
        return shader_name

    color = _get_material_color_array(material["color"])
    maya_ctl.send_python_command(
        "shader = cmds.shadingNode('lambert', asShader=True, name='%s')" % shader_name
    )
    maya_ctl.send_python_command(
        "shader_grp = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name='%s_grp')"
        % shader_name
    )
    # Set the shader color (RGB)
    maya_ctl.send_python_command(
        "cmds.setAttr(f'{shader}.color', %f, %f, %f, type='double3')"
        % (color[0], color[1], color[2])
    )
    # Set transparency (Alpha is 1 - given alpha value)
    maya_ctl.send_python_command(
        "cmds.setAttr(f'{shader}.transparency', %f, %f, %f, type='double3')"
        % ((1 - color[3],) * 3)
    )
    # Connect shader to shading group
    maya_ctl.send_python_command(
        "cmds.connectAttr(f'{shader}.outColor', f'{shader_grp}.surfaceShader', force=True)"
    )
    maya_ctl.send_python_command("shaders['%s'] = shader_grp" % shader_name)
    return shader_name


def _get_euler_from_quaternion(qx, qy, qz, qw):
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw  # in radians


def _add_furniture_to_scene(maya_ctl, object_id, furniture):
    furniture_name = "%s_%05d" % (furniture["category"], object_id)

    if furniture["position"][1] > 1:
        maya_ctl.send_python_command(
            "cmds.pointLight(position=[%f, %f, %f], name='%s', intensity=20)"
            % (
                furniture["position"][0],
                furniture["position"][1],
                furniture["position"][2],
                furniture_name,
            )
        )
        maya_ctl.send_python_command("cmds.parent('%s', 'lights')" % furniture_name)

    maya_ctl.send_python_command(
        "cmds.file('%s', i=True, gr=True, gn='furniture_group', mergeNamespacesOnClash=True, namespace='%s')"
        % (furniture["model"].replace("\\", "/"), furniture_name)
    )
    # Set the position and rotation of the furniture
    maya_ctl.set_object_world_transform("furniture_group", furniture["position"])
    maya_ctl.set_object_local_rotation(
        "furniture_group",
        np.rad2deg(_get_euler_from_quaternion(*furniture["rotation"])),
    )
    # Scale the furniture
    maya_ctl.set_object_attribute("furniture_group", "scaleX", furniture["scale"][0])
    maya_ctl.set_object_attribute("furniture_group", "scaleY", furniture["scale"][1])
    maya_ctl.set_object_attribute("furniture_group", "scaleZ", furniture["scale"][2])
    # Triangulate the furniture
    maya_ctl.send_command("polyTriangulate -ch 1 furniture_group")
    maya_ctl.send_command("select -r furniture_group")
    maya_ctl.send_command("DeleteHistory")
    # Rename the furniture
    maya_ctl.send_python_command("cmds.parent('furniture_group', 'furniture')")
    maya_ctl.send_python_command(
        "cmds.rename('furniture_group', '%s')" % furniture_name
    )


def _get_openmaya_mesh(vertices, faces):
    """
    Blender mesh.from_pydata() -> Maya mesh.Create(OpenMaya)

    More information
        Blender
            https://docs.blender.org/api/current/bpy.types.Mesh.html?highlight=from_pydata#bpy.types.Mesh.from_pydata
        Maya
            https://help.autodesk.com/view/MAYAUL/2017/ENU/?guid=__py_ref_class_open_maya_1_1_m_fn_mesh_html
            https://forums.autodesk.com/t5/maya-programming/create-mesh-from-list/td-p/7575371
            https://forums.cgsociety.org/t/creating-polygon-object-from-scratch-with-pymel/1845044/2
    """
    unique_vertices = {}
    om_vertices = []
    om_faces = []

    for v in vertices:
        if str(v) not in unique_vertices:
            unique_vertices[str(v)] = len(unique_vertices)
            om_vertices.append(v)
    for face in faces:
        om_faces.append([unique_vertices[str(vertices[v_idx])] for v_idx in face])

    return om_vertices, om_faces


def _add_mesh_to_scene(maya_ctl, object_id, mesh, shader_name):
    mesh_name = "%s_%05d" % (mesh["type"], object_id)

    maya_ctl.send_python_command("cmds.select(all=True, hierarchy=True)")
    maya_ctl.send_python_command("currentObjs = cmds.ls(selection=True )")
    maya_ctl.send_python_command("meshFn = OpenMaya.MFnMesh()")
    maya_ctl.send_python_command(
        "vertices = []; polygonFaces = []; polygonConnects = []"
    )

    vertices, faces = _get_openmaya_mesh(
        mesh["vertices"].tolist(), mesh["faces"].tolist()
    )
    for v in vertices:
        maya_ctl.send_python_command(
            f"vertices.append(OpenMaya.MPoint({v[0]}, {v[1]}, {v[2]}))"
        )
    for f in faces:
        maya_ctl.send_python_command(f"polygonFaces.append({len(f)})")
        maya_ctl.send_python_command(f"polygonConnects += {f}")

    # Create mesh in Maya
    maya_ctl.send_python_command(
        "meshFn.create(vertices, polygonFaces, polygonConnects)"
    )
    maya_ctl.send_python_command(
        "cmds.sets(meshFn.name(), edit=True, forceElement='initialShadingGroup')"
    )
    # Rename mesh in Maya
    maya_ctl.send_python_command(
        "curr_name = cmds.listRelatives(meshFn.name(), fullPath=True, parent=True)[0]"
    )
    maya_ctl.send_python_command("cmds.rename(curr_name, '%s')" % mesh_name)

    group_name = ""
    require_hide = False
    require_project_uv = False
    if "Ceiling" in mesh["type"]:
        group_name = "ceilings"
        require_project_uv = ("planar", "y")
        require_hide = True
    elif "Floor" in mesh["type"]:
        group_name = "floors"
        require_project_uv = ("planar", "y")
    elif "Window" in mesh["type"]:
        group_name = "windows"
    else:
        group_name = "structure"
        require_project_uv = ("cylindrical", "z")

    # Group meshes
    maya_ctl.send_python_command("cmds.parent('%s', '%s')" % (mesh_name, group_name))
    if require_hide:
        maya_ctl.send_python_command("cmds.hide()")
    if require_project_uv:
        maya_ctl.send_python_command(
            "cmds.polyProjection('%s.f[0:]', type='%s', md='%s')"
            % (mesh_name, require_project_uv[0], require_project_uv[1])
        )
    # Link shader to mesh
    maya_ctl.send_python_command(
        "cmds.sets('%s', edit=True, forceElement=shaders['%s'])"
        % (mesh_name, shader_name)
    )


def add_instance_meshes_to_scene(maya_ctl, layout, model_dir):
    n_objects = 0
    # Create shared shaders
    maya_ctl.send_python_command("shaders = {}")
    for k, v in tqdm(layout["instances"].items()):
        n_objects += 1
        if k.startswith("furniture/"):
            if v["ref"] not in layout["furnitures"]:
                # logging.warning("Furniture not found: %s", v["ref"])
                continue

            furniture = layout["furnitures"][v["ref"]]
            furniture = {
                "model": os.path.join(model_dir, furniture["jid"], "raw_model.obj"),
                "texture": os.path.join(model_dir, furniture["jid"], "texture.png"),
                "category": furniture["category"],
            }
            if not os.path.exists(furniture["model"]):
                # logging.warning("Model not found: %s", furniture["model"])
                continue
            elif not os.path.exists(furniture["texture"]):
                # logging.warning("Texture not found: %s", furniture["texture"])
                continue
            else:
                _add_furniture_to_scene(
                    maya_ctl, n_objects, dict(list(furniture.items()) + list(v.items()))
                )
        elif k.startswith("mesh/"):
            mesh = layout["meshes"][v["ref"]]
            shader_name = _get_shader(
                maya_ctl, layout["materials"][mesh["material_id"]]
            )
            _add_mesh_to_scene(maya_ctl, n_objects, mesh, shader_name)
        else:
            logging.warning("Unknown instance type: %s", k)
            continue


def main(maya_ctl, layout_dir, model_dir, output_dir):
    ROOT_GROUP = "house"
    GROUPS = [
        "structure",
        "ceilings",
        "floors",
        "windows",
        "doors",
        "furniture",
        "lights",
    ]

    logging.info("Loading furniture categories ...")
    with open(os.path.join(model_dir, "model_info.json"), "r") as f:
        models = json.load(f)
        furniture_categories = {
            m["model_id"]: _get_furniture_category(m) for m in models
        }

    logging.info("Exporting Layouts to USD files ...")
    layouts = os.listdir(layout_dir)
    for lf in tqdm(layouts):
        output_file_path = os.path.join(
            output_dir, lf.replace(".json", ".usd")
        ).replace("\\", "/")
        if os.path.exists(output_file_path):
            continue

        with open(os.path.join(layout_dir, lf), "r") as f:
            layout = get_house_layout(json.load(f), furniture_categories)

        # Initialize a new scene
        maya_ctl.set_new_scene()
        maya_ctl.send_python_command("cmds.group(em=True, name='%s')" % ROOT_GROUP)
        for g in GROUPS:
            maya_ctl.send_python_command("cmds.group(em=True, name='%s')" % g)

        # Add meshes to scene
        add_instance_meshes_to_scene(maya_ctl, layout, model_dir)
        # Move all groups to the "house" group
        for g in GROUPS:
            maya_ctl.send_python_command("cmds.parent('%s', '%s')" % (g, ROOT_GROUP))
        # Output the scene to USD
        maya_ctl.send_python_command(
            "cmds.file('%s', force=True, options='exportUVs=1', type='USD Export', exportAll=True)"
            % output_file_path
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.DEBUG,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout_dir", default=os.path.join(PROJECT_HOME, os.pardir, "3D-FRONT")
    )
    parser.add_argument(
        "--model_dir", default=os.path.join(PROJECT_HOME, os.pardir, "3D-FUTURE")
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(PROJECT_HOME, os.pardir, "USD")
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
    maya_ctl = helpers.maya.MayaController(port=args.port)
    maya_ctl.send_python_command("from maya.api import OpenMaya")
    logging.info("Connected to Maya Server.")
    # Load the Maya USD plugin
    maya_ctl.send_python_command("cmds.loadPlugin('mayaUsdPlugin')")
    if not maya_ctl.send_python_command(
        "cmds.pluginInfo('mayaUsdPlugin', query=True, loaded=True)"
    ).startswith("1"):
        logging.error("Failed to load the Maya USD plugin.")
        sys.exit(1)

    main(maya_ctl, args.layout_dir, args.model_dir, args.output_dir)
