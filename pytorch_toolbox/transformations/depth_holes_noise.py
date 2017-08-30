import numpy as np
import random
import scipy.misc


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
