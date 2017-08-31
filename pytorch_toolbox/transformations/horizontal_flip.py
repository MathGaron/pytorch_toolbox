import numpy as np


class HorizontalFlip(object):
    def __call__(self, numpy):
        img = np.flip(numpy, axis=1).copy()
        return img
