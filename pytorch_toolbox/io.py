# -*- coding: utf-8 -*-
"""io: In Out utility functions

This module provide helper function to load/save common data structures


"""

import yaml


def yaml_loader(filepath):
    with open(filepath, "r") as file_descriptor:
        data = yaml.load(file_descriptor)
        return data