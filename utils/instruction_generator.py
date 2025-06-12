# -*- coding: utf-8 -*-
#
# @File:   instruction_generator.py
# @Author: Haozhe Xie
# @Date:   2025-05-31 19:42:51
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-06-12 16:37:45
# @Email:  root@haozhexie.com

import random


class InstructionGenerator:
    @staticmethod
    def generate_instruction(filename):
        tokens = filename.split("_")
        action_type = tokens[0]
        object_name = tokens[2][:-3]
        object_dyn = "the rolling" if tokens[2][-1:] == "d" else "the static"

        return "%s %s %s.\n" % (
            InstructionGenerator._get_action_name(action_type),
            object_dyn,
            InstructionGenerator._get_object_name(object_name),
        )

    @staticmethod
    def _get_action_name(action):
        if action == "pick":
            return random.choice(["Pick up", "Grasp", "Catch", "Grab", "Get hold of"])

        raise ValueError("Unknown action type: %s" % action)

    @staticmethod
    def _get_object_name(object_name):
        if object_name == "fcan":
            return "food can"
        elif object_name == "dbottle":
            return "drink bottle"
        elif object_name == "wbottle":
            return "water bottle"
        else:
            return object_name
