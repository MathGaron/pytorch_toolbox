"""
    data_transforms is a bunch of transform object useful for image processing and more
    Note : Compose comes from torchvision package

    TODO: depthTransform should be appliable to multiple channels
"""

import torch
import random
import numpy as np
import cv2
import scipy.signal
import scipy.misc
import scipy.stats as st
from skimage.color import rgb2hsv, hsv2rgb
from skimage.measure import block_reduce


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


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
    def __init__(self, size, interpolation='bilinear'):
        """
        :param size: (w, h)
        :param interpolation: 'nearest', 'lanczos', 'bilinear', 'bicubic', 'cubic'
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, numpy):
        resized = scipy.misc.imresize(numpy, self.size, self.interpolation)
        return resized


class ToFloat(object):
    def __call__(self, numpy):
        return numpy.astype(np.float32)


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


class BoundingBoxTransform(object):
    def __call__(self, bb):
        for i in range(bb.shape[0]):
            bb[i][2] = bb[i][0] + bb[i][2]
            bb[i][3] = bb[i][1] + bb[i][3]

        inds = np.zeros((bb.shape[0], 1), dtype=np.float32)
        rois = np.hstack((inds, bb))
        return rois


class DepthGaussianNoise(object):
    def __init__(self, proba, gaussian_std):
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


class DepthHolesNoise(object):
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


class ImageBlur(object):
    def __init__(self, rgb_proba, depth_proba, kernel_max_size):
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
        """Returns a 2D Gaussian kernel array."""

        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel


class ImageHue(object):
    def __init__(self, h_proba, s_proba, v_proba):
        self.h_probability = h_proba
        self.s_probability = s_proba
        self.v_probability = v_proba

    def __call__(self, numpy_img):
        # copy = numpy_img.copy()
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
        # image_compare(copy[:, :, :3].astype(np.uint8), numpy_img[:, :, :3].astype(np.uint8), 1)
        return numpy_img

class ToneMapper(object):
    def __init__(self):
        pass

    def __call__(self, numpy_img):
        opencv_img = cv2.merge((numpy_img[0], numpy_img[1], numpy_img[2]))
        tonemap1 = cv2.createTonemapDurand(gamma=2.2)
        tonemap_img = tonemap1.process(opencv_img)
        tonemap_img_8bit = np.clip(tonemap_img * 255, 0, 255).astype('uint8')

        reshape_tonemap_img_8bit = np.empty([3, 64, 128])
        for i in range(3):
             reshape_tonemap_img_8bit[i, :, :] = tonemap_img_8bit[:, :, i]

        return reshape_tonemap_img_8bit
