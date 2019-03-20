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


class DepthGaussianNoise(object):
    """
        Add gaussian noise on depth channel ( last channel)
        input : nd.array batch of images : [N, H, W, C]
        output : nd.array batch of images : [N, H, W, C]
    """
    def __init__(self, proba, gaussian_std):
        """
        :param proba: float (0-1) probability of adding noise
        :param gaussian_std: standard deviation of gaussian noise
        """
        self.probability = proba
        self.gaussian_std = gaussian_std

    def __call__(self, numpy_img):
        if random.uniform(0, 1) < self.probability:
            noise = random.uniform(0, self.gaussian_std)
            numpy_img[:, :, -1] = self.add_noise(numpy_img[:, :, -1], noise)
        return numpy_img

    @staticmethod
    def add_noise(img, gaussian_std):
        type = img.dtype
        copy = img.astype(np.float)
        gaussian_noise = np.random.normal(0, gaussian_std, img.shape)
        copy = (gaussian_noise + copy)
        if type == np.uint8:
            copy[copy < 0] = 0
            copy[copy > 255] = 255
        return copy.astype(type)


class DepthHolesNoise(object):
    """
        Add depth holes noise on depth channel ( last channel)
        the holes are predifined patches with random scale. A random number of patches is added to depth at random
        position.
        input : nd.array batch of images : [N, H, W, C]
        output : nd.array batch of images : [N, H, W, C]
    """
    def __init__(self, max_holes):
        self.max_holes = max_holes
        pattern0 = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 0, 0, 1],
                             [1, 1, 0, 0, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]])

        pattern1 = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 0, 0, 1],
                             [1, 1, 0, 1, 1],
                             [1, 1, 0, 0, 1],
                             [1, 1, 0, 0, 1]])

        pattern2 = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 0, 0, 1],
                             [1, 0, 0, 0, 1],
                             [1, 1, 0, 1, 1],
                             [1, 1, 1, 1, 1]])

        pattern3 = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 0, 1, 1],
                             [1, 0, 0, 1, 1],
                             [1, 1, 0, 0, 1],
                             [1, 1, 0, 0, 1]])

        pattern4 = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 0, 0, 1],
                             [1, 1, 0, 0, 1],
                             [0, 0, 0, 1, 1],
                             [0, 0, 1, 1, 1]])

        pattern5 = np.array([[1, 1, 0, 1, 1],
                             [1, 0, 0, 0, 1],
                             [1, 0, 0, 0, 1],
                             [1, 1, 0, 0, 1],
                             [1, 1, 0, 0, 0]])

        self.patterns = [pattern0, pattern1, pattern2, pattern3, pattern4, pattern5]

    def __call__(self, numpy_img):
        n_holes = random.randint(0, self.max_holes)
        for i in range(n_holes):
            x = random.randint(0, numpy_img.shape[1] - 20)
            y = random.randint(0, numpy_img.shape[0] - 20)
            pattern = random.choice(self.patterns)
            pyr_size = random.randint(1, 4)
            pattern_size = (pattern.shape[0] * pyr_size, pattern.shape[1] * pyr_size)
            pattern = scipy.misc.imresize(pattern, pattern_size, interp="nearest")
            pattern[pattern > 0] = 1
            numpy_img[y:y + pattern_size[1], x:x + pattern_size[0], -1] *= pattern
        return numpy_img
