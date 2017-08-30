import numpy as np


class BoundingBoxTransform(object):
    def __call__(self, bb):
        for i in range(bb.shape[0]):
            bb[i][2] = bb[i][0] + bb[i][2]
            bb[i][3] = bb[i][1] + bb[i][3]

        inds = np.zeros((bb.shape[0], 1), dtype=np.float32)
        rois = np.hstack((inds, bb))
        return rois
