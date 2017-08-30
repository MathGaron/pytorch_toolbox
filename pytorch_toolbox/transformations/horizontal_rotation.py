import random
import numpy as np


class HorizontalRotation(object):
    def __call__(self, numpy):
        # apply random rotation
        rotation_amplitude = random.randint(0, numpy.shape[1])
        img = np.roll(numpy, rotation_amplitude, axis=1)
        return img
