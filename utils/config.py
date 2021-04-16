# !/usr/bin/env python
# -*- coding:utf-8 -*-
import configparser
import os


def _init():
    global _global_dict
    _global_dict = {}


def set_value(name, value):
    _global_dict[name] = value


def get_value(name, defValue=None):
    try:
        return _global_dict[name]
    except KeyError:
        return defValue


def get_directory():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
    return str(root_path)


_init()
# print(os.getcwd())
cf = configparser.ConfigParser()
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
cf.read(str(root_path) + "/file/properties.ini", encoding='utf-8')
set_value('conf', cf)

project_name = root_path.split(os.path.sep)[-1]
project_root_path = os.path.abspath(os.path.dirname(__file__)).split(project_name)[0] + project_name
print(project_root_path)