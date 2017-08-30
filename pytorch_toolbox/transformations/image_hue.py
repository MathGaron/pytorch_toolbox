import random
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb


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
