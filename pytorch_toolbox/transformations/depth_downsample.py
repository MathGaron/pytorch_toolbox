import random
import numpy as np
from skimage.measure import block_reduce
import scipy.misc


class DepthDownsample(object):
    """
    Compute downsampling of depth channel ( last channel)
    input : nd.array batch of images : [N, H, W, C]
    output : nd.array batch of images : [N, H, W, C]
    """
    def __init__(self, block_size=(3, 3), proba=1):
        """
        :param block_size: tuple of size 2, size of downsampling block
        :param proba: float (0-1) probability of downsampling the image
        """
        self.proba = proba
        self.block_size = block_size

    def __call__(self, numpy_img):
        h, w, c = numpy_img.shape
        if random.uniform(0, 1) < self.proba:
            new_img = block_reduce(numpy_img[:, :, -1], block_size=self.block_size, func=np.mean)[1:-1, 1:-1]
            new_img = scipy.misc.imresize(new_img, (h, w), interp="nearest", mode="F")
            numpy_img[:, :, -1] = new_img
        return numpy_img
