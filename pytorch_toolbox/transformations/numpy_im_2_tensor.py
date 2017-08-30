import torch
import numpy as np


class NumpyImage2Tensor(object):
    def __call__(self, numpy):
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(numpy.shape) == 2:
            numpy = numpy[:, :, np.newaxis]
        img = numpy.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img
