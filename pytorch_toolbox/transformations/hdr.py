import cv2
import numpy as np
import random


class Multiply(object):
    """
        Multiply hdr image by a random constant.
        input : nd.array batch of images : [N, H, W, C]
        output : nd.array batch of images : [N, H, W, C]
    """
    def __init__(self, min_interval, max_interval):
        """
        :param min_interval: minimum value for the constant
        :param max_interval: maximum value for the constant
        """
        self.min_interval = min_interval
        self.max_interval = max_interval

    def __call__(self, numpy):
        multiplier = np.random.randint(self.min_interval, self.max_interval, size=1)
        img = numpy * multiplier.astype(type(numpy))
        return img


class HorizontalRotation(object):
    """
        Apply random rotation to the hdr image.
        input : nd.array batch of images : [N, H, W, C]
        output : nd.array batch of images : [N, H, W, C]
    """
    def __call__(self, numpy):
        rotation_amplitude = random.randint(0, numpy.shape[1])
        img = np.roll(numpy, rotation_amplitude, axis=1)
        return img


class ToneMapper(object):
    """
        Tonemap HDR image using Durand tonemapper. This algorithm decomposes image into two layers: base layer and
        detail layer using bilateral filter and compresses contrast of the base layer thus preserving all the details.
        This implementation uses regular bilateral filter from opencv.
        input : nd.array batch of images : [N, H, W, C]
        output : nd.array batch of images : [N, H, W, C] type uint8
    """
    def __init__(self, gamma=2.2):
        """
        :param gamma: positive value for gamma correction. Gamma value of 1.0 implies no correction, gamma equal to
        2.2f is suitable for most displays. Generally gamma > 1 brightens the image and gamma < 1 darkens it.
        """
        self.gamma = gamma

    def __call__(self, numpy_img):
        opencv_img = cv2.merge((numpy_img[0], numpy_img[1], numpy_img[2]))
        tonemap1 = cv2.createTonemapDurand(self.gamma)
        tonemap_img = tonemap1.process(opencv_img)
        tonemap_img_8bit = np.clip(tonemap_img * 255, 0, 255).astype('uint8')

        reshape_tonemap_img_8bit = np.empty([3, 64, 128])
        for i in range(3):
            reshape_tonemap_img_8bit[i, :, :] = tonemap_img_8bit[:, :, i]

        return reshape_tonemap_img_8bit
