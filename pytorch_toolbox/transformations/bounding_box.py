import numpy as np

class BoundingBoxRect2Point(object):
    """
    Transform rect bounding box to point wise box:
    rect bounding box define a bounding box by left up coord and width, height : [x1, y1, w, h]
    point wise bounding box define a bounding box by left up coord and right bottom coord : [x1, y1, x2, y2]
    input : nd.array Shape [N, 4]
    output : nd.array Shape [N, 4]
    """
    def __call__(self, bb):
        new_bb = bb.copy()
        new_bb[:, 2] = bb[:, 0] + bb[:, 2]
        new_bb[:, 3] = bb[:, 1] + bb[:, 3]
        return new_bb


class BoundingBoxRatio(object):
    """
    Take image coordinate bounding box and normalize it between 0, 1
    input : nd.array Shape [N, 4]
    output nd.array Shape [N, 4]
    """
    def __init__(self, w, h):
        """
        :param w: image width
        :param h: image height
        """
        self.w = w
        self.h = h

    def __call__(self, bb):
        new_bb = bb.copy()
        new_bb[:, 0] /= self.w
        new_bb[:, 2] /= self.w
        new_bb[:, 1] /= self.h
        new_bb[:, 3] /= self.h
        return new_bb


class BBClean(object):
    """
    input is bounding box of type rect : [x, y, w, h]
    Remove wrong bounding box ( width or height of 0)
    input : nd.array Shape [N, 4]
    output nd.array Shape [N, 4]
    """
    def __call__(self, bb):
        mask = (bb[:, 2:4] == 0).any(axis=1)
        neg_mask = np.bitwise_not(mask)
        new_bb = bb[neg_mask, :]
        return new_bb


class BBPick(object):
    """
        Keep n Bounding boxes
        if n is larger than quantity of inputs, the output will wrap the inputs to fill the missing values
        input : nd.array Shape [N, 4]
        output nd.array Shape [n, 4]
    """
    def __init__(self, n):
        self.n = n

    def __call__(self, bb):
        new_bb = np.zeros((self.n, bb.shape[1]), dtype=bb.dtype)
        if bb.shape[0] >= self.n:
            new_bb[:, :] = bb[:self.n, :]
        else:
            for i in range(new_bb.shape[0]):
                j = i % bb.shape[0]
                new_bb[i, :] = bb[j, :]
        return new_bb