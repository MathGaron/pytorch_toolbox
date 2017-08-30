import numpy as np


class Multiply(object):
    """Given an interval, return the input image multiplied by this value
    :param min_multiplier: minimum value for the interval
    :param max_multiplier: maximum value for the interval
    """
    def __init__(self, min_multiplier, max_multiplier):
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

    def __call__(self, numpy):
        multiplier = np.random.randint(self.min_multiplier, self.max_multiplier, size=1)
        img = (numpy * multiplier).astype(np.float32)
        return img
