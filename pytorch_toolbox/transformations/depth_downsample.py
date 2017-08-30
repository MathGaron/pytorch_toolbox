import random
import numpy as np
from skimage.measure import block_reduce
import scipy.misc

class DepthDownsample(object):
    def __init__(self, proba=1):
        self.proba = proba

    def __call__(self, numpy_img):
        w, h, c = numpy_img.shape
        if random.uniform(0, 1) < self.proba:
            new_img = block_reduce(numpy_img[:, :, -1], block_size=(3, 3), func=np.mean)[1:-1, 1:-1]
            new_img = scipy.misc.imresize(new_img, (w, h), interp="nearest", mode="F")
            numpy_img[:, :, -1] = new_img
        return numpy_img
