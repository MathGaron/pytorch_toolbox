class Normalize(object):
    """
        will normalize each channel of the torch.*Tensor, i.e.
        channel = (channel - mean) / std

        input : torch tensor batch of images : [N, C, H, W]
        output : torch tensor batch of images : [N, C, H, W]
    """

    def __init__(self, mean, std):
        """

        :param mean: iterator with value per channel : ex : [R, G, B]
        :param std: iterator with value per channel : ex : [R, G, B]
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor
