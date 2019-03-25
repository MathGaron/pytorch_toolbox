"""utils : general utility

    This module provide simple functions and class for general use.

"""
import torch
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class AverageMeter(object):
    """
    Compute and store running average values.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def classification_accuracy(prediction, target, top_k=(1,)):
    """
    Compute classification accuracy given prediction and target
    #todo: use python lists instead of pytorch tensors
    :param prediction: tensor of predicted classes
    :param target: tensor of target classes
    :param top_k:
    :return:
    """
    maxk = max(top_k)
    batch_size = target.size(0)
    prediction = torch.exp(prediction)
    _, pred = prediction.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0], pred