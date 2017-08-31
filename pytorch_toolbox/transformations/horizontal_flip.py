import numpy as np


class HorizontalFlip(object):
    def __call__(self, numpy):
        random_flip = random.randint(0, 1)
        if random_flip:
            img = np.flip(numpy, axis=1).copy()
        else:
            img = numpy
        return img