# -*- coding: utf-8 -*-
#
# @File:   maya.py
# @Author: Wensi Ai (@wensi-ai)
# @Date:   2025-03-20 14:46:01
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-03-24 16:52:26
# @Email:  root@haozhexie.com
#
# Ref: https://github.com/arnold-benchmark/Usdify/blob/main/controller.py

import socket

import numpy as np


class MayaController:
    """
    This is controller to `remotely` control Maya to make character animations.
    The default local host is *127.0.0.1*

    :param PORT: port to connect to the local socket server, defaults to 0
    :type PORT: int

    :ivar client: socket client
    """

    def __init__(self, host="127.0.0.1", port=12345):
        """Constructor method"""
        # connect to Maya server
        ADDR = (host, port)
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(ADDR)

    def send_command(self, command: str):
        """Send a string command to the socket server (Maya side)

        :param command: a string command
        :type command: str
        """
        # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # client.connect(ADDR)
        command = command.encode()  # the command from external editor to maya

        my_message = command
        self.client.send(my_message)
        data = self.client.recv(16384)  # receive the result info
        # client.close()
        # print(data)
        # ret = str(data.decode(encoding="ASCII"))
        ret = data.decode("utf-8")
        return ret

    def send_python_command(self, command: str):
        """Send a string command to the socket server (Maya side)

        :param command: a string command
        :type command: str
        """
        # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # client.connect(ADDR)
        command = 'python("{}")'.format(command)
        command = command.encode()  # the command from external editor to maya

        my_message = command
        self.client.send(my_message)
        data = self.client.recv(16384)  # receive the result info
        # client.close()
        # print(data)
        # ret = str(data.decode(encoding="ASCII"))
        ret = data.decode("utf-8")
        return ret

    def close(self):
        """Close socket client"""
        self.client.close()

    def __del__(self):
        self.close()

    # --------------------------------SET-----------------------------------------
    def set_new_scene(self):
        """Set new Maya empty scene"""
        send_message = "file -f -new;"
        return self.send_command(send_message)

    def Set_current_time_frame(self, time_frame: int):
        """Set timeline in the Maya scene

        :param time_frame: key frame
        :type time_frame: int
        """
        send_message = "currentTime -edit" + " " + str(time_frame) + ";"
        return self.send_command(send_message)

    def set_object_world_transform(self, object_name: str, location: list):
        """Set world absolute location for object with location [x, y, z]

        :param object_name: the name of the object in hierarchy view
        :type object_name: str
        :param location: location in list [x, y, z]
        :type location: list
        """
        send_message = "select -replace " + object_name + ";"
        send_message += (
            "move -absolute "
            + str(location[0])
            + " "
            + str(location[1])
            + " "
            + str(location[2])
            + ";"
        )
        return self.send_command(send_message)

    def move_object_world_relative(self, object_name: str, location: list):
        """Move an object relatively with location [x, y, z]

        :param object_name: the name of the object in hierarchy view
        :type object_name: str
        :param location: location in list [x, y, z]
        :type location: list
        """
        send_message = "select -replace " + object_name + ";"
        send_message += (
            "move -relative "
            + str(location[0])
            + " "
            + str(location[1])
            + " "
            + str(location[2])
            + ";"
        )
        return self.send_command(send_message)

    def set_object_local_transform(self, object_name: str, location: list):
        """Set world absolute location for object with location [x,y] or [x, y, z]

        :param object_name: the name of the object in hierarchy view
        :type object_name: str
        :param location: location in list [x, y, z]
        :type location: list
        """
        send_message = "select -replace " + object_name + ";" + "move -relative "
        for value in location:
            send_message += str(value) + " "

        send_message += ";"
        return self.send_command(send_message)

    def set_object_local_rotation(self, object_name: str, rotation: list):
        """Set world absolute location for object with rotation [x,y] or [x, y, z] in degree

        :param object_name: the name of the object in hierarchy view
        :type object_name: str
        :param location: location in list [x, y, z]
        :type location: list
        """
        send_message = "select -replace " + object_name + ";" + "rotate -relative "
        for value in rotation:
            send_message += str(value) + "deg "

        send_message += ";"
        return self.send_command(send_message)

    def set_current_key_frame_for_attribute(self, object_name: str, attr_name: str):
        """Set keyframe for object attribute

        :param object_name: name of the object in hierarchy view
        :type object_name: str
        :param attr_name: name of attribute
        :type attr_name: list
        """
        send_message = "select -r " + object_name + ";"
        send_message += "setKeyframe -at " + attr_name + ";"
        return self.send_command(send_message)

    def set_current_key_frame_for_position_and_rotation(self, object_name: str):
        """Set keyframe for object position and rotation

        :param object_name: name of the object in hierarchy view
        :type object_name: str
        """
        send_message = "select -r " + object_name + ";"
        send_message += "setKeyframe -at translate;"
        send_message += "setKeyframe -at rotate;"
        return self.send_command(send_message)

    def set_current_key_frame_for_objects(self, object_list):
        """Set keyframe for a list of objects

        :param object_list: a list of object names
        :type object_list: list
        """
        send_message = "setKeyframe {"
        for obj in object_list:
            send_message += '"' + str(obj) + '", '
        send_message = send_message[:-2] + "};"
        return self.send_command(send_message)

    def set_object_attribute(self, object_name: str, attr_name: str, value: float):
        """Set object attriture with value

        :param object_name: name of the object
        :type object_name: str
        :param attr_name: attribute of the object
        :type attr_name: str
        :param value: attribute value
        :type attr_name: float
        """

        send_message = (
            "setAttr " + object_name + "." + attr_name + " " + str(value) + ";"
        )
        return self.send_command(send_message)

    # def set_multiple_attributes(self, attributes: dict):
    #     '''
    #     Set object attriture with value

    #     :param attributes: dictionary of facial attributes to be set
    #     :type attributes: dict
    #     '''
    #     for joint, attr_dict in attributes.items():
    #         for name, value in attr_dict.items():
    #             self.set_object_attribute(joint, name, value)

    def undo(self):
        """Send undo command to socket server (Maya)"""
        send_message = "undo;"
        rec_message = self.send_command(send_message)
        return rec_message

    def undo_to_beginning(self, max_step=200):
        """
        Undo Maya file to beginning

        :param max_step: max undo steps, default 200
        :type max_step: int, optional
        """
        for _ in range(max_step):
            rec_message = self.undo()
            if "There are no more commands to undo." in rec_message:
                print("(UndoToBeginning)Undo steps:", _)
                return

    def screenshot(self, save_file: str, camera="persp", width=1024, height=1024):
        """
        Take maya screen shot and save to picture

        :param save_file: save file name
        :type save_file: str
        :param camera: camera name in the Maya scene, default "persp"
        :type camera: str, optional
        :param width: screenshot width, default 1024
        :type width: int, optional
        :param height: screenshot height, default 1024
        :type height: int, optional

        """
        send_message = "string $editor = `renderWindowEditor -q -editorName`;\n"
        # send_message += "string $myCamera = " + camera + ";\n"
        send_message += 'string $myFilename ="' + save_file + '";\n'
        send_message += (
            "render -x " + str(width) + " -y " + str(height) + " " + camera + ";\n"
        )
        send_message += "renderWindowEditor -e -wi $myFilename $editor;"

        recv_message = self.send_command(send_message)
        print("(ScreenShot)", recv_message)
        return recv_message

    # ---------------------------GET-----------------------------------
    def get_all_objects(self):
        """
        Get all the objects from Maya scene

        :return recv_message: a list of object names
        :rtype: list
        """
        send_message = "ls;"
        recv_message = self.send_command(send_message)
        return recv_message.rstrip("\x00").rstrip("\n").split("\t")

    def get_time_slider_range(self):
        """
        Get the range of time slider

        :return: [min, max] representing the minimum and maximum values of the timeline
        :rtype: list
        """
        send_message = "playbackOptions -q -minTime"
        recv_message_1 = self.send_command(send_message)
        recv_message_1 = recv_message_1.rstrip("\x00").rstrip("\n").split("\t")

        send_message = "playbackOptions -q -maxTime"
        recv_message_2 = self.send_command(send_message)
        recv_message_2 = recv_message_2.rstrip("\x00").rstrip("\n").split("\t")

        return [int(float(recv_message_1[0])), int(float(recv_message_2[0]))]

    def get_object_world_transform(self, object_name: str):
        """
        Get object world location

        :param object_name: name of the object
        :type object_name: str

        :return: cordindate [x,y,z] of the object
        :rtype: list
        """
        send_message = "xform -q -t -ws " + object_name + ";"
        recv_message = self.send_command(send_message)
        recv_message = recv_message.rstrip("\x00").rstrip("\n").split("\t")
        return [np.around(float(_), decimals=2) for _ in recv_message]

    def get_object_local_roation(self, object_name: str):
        """
        Get object local rotation

        :param object_name: name of the object
        :type object_name: str

        :return: cordindate [x,y,z] of the object
        :rtype: list
        """
        send_message = "xform -q -ro -os " + object_name + ";"
        recv_message = self.send_command(send_message)
        recv_message = recv_message.rstrip("\x00").rstrip("\n").split("\t")
        return [np.around(float(_), decimals=2) for _ in recv_message]

    def get_object_attribute(self, object_name: str, attr_name: str):
        """
        Get attribute value of an object

        :param object_name: name of the object
        :type object_name: str

        :param attr_name: name of the attribute
        :type attr_name: str

        :return: attribute value
        :rtype: float
        """
        send_message = "getAttr " + object_name + "." + attr_name + ";"
        recv_message = self.send_command(send_message)
        recv_message = recv_message.rstrip("\x00").rstrip("\n").split("\t")
        return [np.around(float(_), decimals=2) for _ in recv_message]

    def get_body_information(self, joint_list: list):
        """
        Get a list of information of the joint list

        :param joint_list: a list of joint names
        :type joint_list: list

        :return: a list with joint positions
        :rtype: list
        """
        BODY_INFO_DIC = {}
        for joint_name in joint_list:
            info_dict = {}
            world_transform = self.get_object_world_transform(joint_name)
            info_dict["world_transform"] = world_transform

            local_transform = self.get_object_attribute(joint_name, "translate")
            info_dict["local_transform"] = local_transform

            local_rotation = self.get_object_attribute(joint_name, "rotate")
            info_dict["local_rotation"] = local_rotation

            BODY_INFO_DIC[joint_name] = info_dict

        return BODY_INFO_DIC
