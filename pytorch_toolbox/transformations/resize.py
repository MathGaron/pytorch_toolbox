import scipy.misc


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
