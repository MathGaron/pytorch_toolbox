# -*- coding: utf-8 -*-
"""io: In Out utility functions

This module provide helper function to load/save common data structures


"""

import yaml


def yaml_load(file_path):
    with open(file_path, "r") as file_descriptor:
        data = yaml.load(file_descriptor)
        return data


def yaml_dump(file_path, data):
    with open(file_path, 'w') as file_descriptor:
        yaml.dump(data, file_descriptor)
