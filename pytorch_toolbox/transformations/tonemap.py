import cv2
import numpy as np


class ToneMapper(object):
    def __init__(self, gamma=2.2):
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
