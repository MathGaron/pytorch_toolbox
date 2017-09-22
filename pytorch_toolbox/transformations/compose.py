class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        transforms.Compose([
             transforms.CenterCrop(10),
             transforms.ToTensor(),
        ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
