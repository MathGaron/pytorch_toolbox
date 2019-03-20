import random
import scipy
import numpy as np
import scipy.stats as st
from skimage.color import rgb2hsv, hsv2rgb
import torch


class NumpyImage2Tensor(object):
    def __call__(self, numpy):
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(numpy.shape) == 2:
            numpy = numpy[:, :, np.newaxis]
        img = numpy.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img


class Resize(object):
    """
        Resize image
        input : nd.array batch of images : [N, H, W, C]
        output : nd.array batch of images : [N, H, W, C]
    """

    def __init__(self, size, interpolation='bilinear'):
        """
        :param size: (h, w)
        :param interpolation: 'nearest', 'lanczos', 'bilinear', 'bicubic', 'cubic'
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, numpy):
        resized = scipy.misc.imresize(numpy, self.size, self.interpolation)
        return resized


class Normalize(object):
    """
        will normalize each channel of the torch.*Tensor, i.e.
        channel = (channel - mean) / std

        input : torch tensor batch of images : [N, C, H, W]
        output : torch tensor batch of images : [N, C, H, W]
    """

    def __init__(self, mean, std):
        """

        :param mean: iterator with value per channel : ex : [R, G, B]
        :param std: iterator with value per channel : ex : [R, G, B]
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


class ImageBlur(object):
    """
        Add gaussian blur noise on channel (RGB being the first 3 and depth the last)
        Note that depth is treated individually, it can be disabled by setting a probability of 0.
        The kernel size is randomly selected.
        input : nd.array batch of images : [N, H, W, C]
        output : nd.array batch of images : [N, H, W, C]
    """
    def __init__(self, rgb_proba, depth_proba, kernel_max_size):
        """
        :param rgb_proba:  probability to add blur on first 3 channels
        :param depth_proba: probability to add blur on last channel
        :param kernel_max_size: maximum blur kernel size
        """
        self.rgb_probability = rgb_proba
        self.depth_probability = depth_proba
        self.kernel_max_size = kernel_max_size

    def __call__(self, numpy_img):
        if random.uniform(0, 1) < self.rgb_probability:
            kernel_size = random.randint(3, self.kernel_max_size)
            kernel = self.gkern(kernel_size)
            if numpy_img.shape[2] == 4:
                for c in range(numpy_img.shape[2] - 1):
                    numpy_img[:, :, c] = scipy.signal.convolve2d(numpy_img[:, :, c], kernel, mode='same')
                if random.uniform(0, 1) < self.depth_probability:
                    kernel_size = random.randint(3, self.kernel_max_size)
                    kernel = self.gkern(kernel_size)
                    numpy_img[:, :, -1] = scipy.signal.convolve2d(numpy_img[:, :, -1], kernel, mode='same')
            else:
                for c in range(numpy_img.shape[2]):
                    numpy_img[:, :, c] = scipy.signal.convolve2d(numpy_img[:, :, c], kernel, mode='same')

        return numpy_img

    @staticmethod
    def gkern(kernlen=21, nsig=2):
        """
        :param kernlen: size of the Gaussian
        :param nsig: Standard deviation for Gaussian kernel.
        :return: Returns a 2D Gaussian kernel array.
        """
        interval = (2 * nsig + 1.) / kernlen
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel


class ImageHSVNoise(object):
    """
        Shift HSV values if rgb image.
        TODO: shift is predefined...we should make it configurable?
        input : nd.array batch of images : [N, H, W, C]
        output : nd.array batch of images : [N, H, W, C]
    """
    def __init__(self, h_proba, s_proba, v_proba):
        self.h_probability = h_proba
        self.s_probability = s_proba
        self.v_probability = v_proba

    def __call__(self, numpy_img):
        rgb = numpy_img[:, :, :3].astype(np.uint8)
        hsv = rgb2hsv(rgb)
        if random.uniform(0, 1) < self.h_probability:
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-0.05, 0.05))
        if random.uniform(0, 1) < self.s_probability:
            hsv[:, :, 1] = (hsv[:, :, 1] + random.uniform(-0.1, 0.3))
        if random.uniform(0, 1) < self.v_probability:
            hsv[:, :, 2] = (hsv[:, :, 2] + random.uniform(-0.1, 0.5))
        hsv = np.clip(hsv, 0, 1)
        rgb = hsv2rgb(hsv) * 255
        numpy_img[:, :, :3] = rgb
        return numpy_img


class HorizontalFlip(object):
    """
        Randomly flip an image.
        input : nd.array batch of images : [N, H, W, C]
        output : nd.array batch of images : [N, H, W, C]
    """
    def __call__(self, numpy_img):
        random_flip = random.randint(0, 1)
        if random_flip:
            img = np.flip(numpy_img, axis=1).copy()
        else:
            img = numpy_img
        return img
