'''
The visualization class provides an easy access to some of the visdom functionalities
Accept as input a number that will be ploted over time or an image of type np.ndarray
'''

from visdom import Visdom
import numpy as np
import numbers


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
        if name not in cls.items_to_visualize:
            cls.new_item(item, name)
        else:
            cls.update_item(item, name)
        cls.items_to_visualize[name] = item

    @classmethod
    def new_item(cls, item, name):
        if isinstance(item, numbers.Number):
            win = cls.vis.line(
                X=np.array([cls.iterator, cls.iterator]),
                Y=np.array([item, item])
            )
            cls.windows[name] = win
        elif isinstance(item, np.ndarray):
            win = cls.vis.image(
                item,
                opts=dict(title=name, caption=name),
            )
            cls.windows[name] = win
        else:
            print("type not supported for visualization")

    @classmethod
    def update_item(cls, item, name):
        if isinstance(item, numbers.Number):
            cls.vis.updateTrace(
                X=np.array([cls.iterator-1, cls.iterator]),
                Y=np.array([cls.items_to_visualize[name], item]),
                win=cls.windows[name]
            )
        elif isinstance(item, np.ndarray):
            cls.vis.image(
                item,
                opts=dict(title=name, caption=name),
                win=cls.windows[name]
            )
