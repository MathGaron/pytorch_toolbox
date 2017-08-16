''' The visualization class provides an easy access to some of the visdom functionalities'''
import visdom
from visdom import Visdom
import numpy as np


class Visualization:

    items_to_visualize = {}
    windows = {}
    iterator = 0
    vis = Visdom()

    def __init__(self):
        pass

    @classmethod
    def visualize(cls, item, name):
        cls.iterator += 1
        if(name not in cls.items_to_visualize):
            cls.new_item(item, name)
        else:
            cls.update_item(item, name)
        cls.items_to_visualize[name] = item

    @classmethod
    def new_item(cls, item, name):
        win = cls.vis.line(
            X=np.array([cls.iterator, cls.iterator]),
            Y=np.array([item, item])
        )
        cls.windows[name] = win

    @classmethod
    def update_item(cls, item, name):
        cls.vis.updateTrace(
            X=np.array([cls.iterator-1, cls.iterator]),
            Y=np.array([cls.items_to_visualize[name], item]),
            win=cls.windows[name]
        )