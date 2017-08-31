

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