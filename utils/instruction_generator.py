# -*- coding: utf-8 -*-
#
# @File:   instruction_generator.py
# @Author: Haozhe Xie
# @Date:   2025-05-31 19:42:51
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-09 15:50:09
# @Email:  root@haozhexie.com

import json
import random


class InstructionGenerator:
    @staticmethod
    def generate_instruction(inst_metadata):
        inst_metadata = json.loads(inst_metadata)
        tmpl = InstructionGenerator._get_instruction_template(inst_metadata["task"])

        object_desc = random.choice(inst_metadata.get("objects", [""]))
        container_desc = random.choice(inst_metadata.get("containers", [""]))
        return tmpl.format_map({"object": object_desc, "container": container_desc})

    @staticmethod
    def _get_instruction_template(task):
        pick_action = random.choice(
            ["pick up", "grasp", "catch", "grab", "get hold of"]
        )
        place_action = random.choice(
            ["place on", "put on", "set on", "position on", "return to", "deposit in"]
        )
        if task == "pick":
            return f"{pick_action} the {{object}}."
        elif task == "place":
            return f"{pick_action} the {{object}} and {place_action} the {{container}}."
        else:
            raise ValueError(f"Unknown task: {task}")
