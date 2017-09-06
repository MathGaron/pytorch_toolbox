import numpy as np


class ToFloat(object):
    """
    Convert numpy array to float 32 bit
    """
    def __call__(self, numpy):
        return numpy.astype(np.float32)
