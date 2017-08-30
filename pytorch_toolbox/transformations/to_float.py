import numpy as np


class ToFloat(object):
    def __call__(self, numpy):
        return numpy.astype(np.float32)
