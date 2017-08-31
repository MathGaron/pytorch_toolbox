import random
import numpy as np


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