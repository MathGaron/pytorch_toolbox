import scipy.misc

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