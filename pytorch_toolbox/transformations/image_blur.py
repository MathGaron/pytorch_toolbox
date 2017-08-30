import random
import scipy
import numpy as np
import scipy.stats as st


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

        interval = (2 * nsig + 1.) / kernlen
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
