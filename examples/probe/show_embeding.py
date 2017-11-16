from PIL import Image
from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
from examples.classification.cat_dog_net import CatDogNet
from pytorch_toolbox.transformations.image import Resize, Normalize, NumpyImage2Tensor
from pytorch_toolbox.transformations.to_float import ToFloat
from pytorch_toolbox.transformations.compose import Compose


def random_rect(n):
    rect = np.random.uniform(0, 1, (n, 2)).repeat(2, axis=1)
    hw = np.random.uniform(0.01, 0.5, (n, 2))
    rect[:, 1::2] += hw
    rect = np.clip(rect, 0, 1)
    return rect


def convert_image(image):
    imagenet_mean = [123, 116, 103]
    imagenet_std = [58, 57, 57]
    transformations = Compose([Resize((128, 128)),
                               ToFloat(),
                               NumpyImage2Tensor(),
                               Normalize(mean=imagenet_mean, std=imagenet_std)])
    return transformations(image).unsqueeze(0)


if __name__ == '__main__':

    backend = "cpu"     # cpu|cuda
    model_path = "cat_dog_model.pth.tar"

    #
    #   Instantiate model and load checkpoint
    #
    model = CatDogNet()
    model.load(model_path)
    model.eval()
    cat_path = "images/cat.jpg"

    # Load test images and prepare it for input in the loaded network
    cat_img_numpy = np.array(Image.open(cat_path).convert('RGB'))
    cat_img_tensor = convert_image(cat_img_numpy)
    # Occluded versions of the image
    for i in range(10):
        cat_img_occluded_numpy = cat_img_numpy.copy()
        rects = random_rect(10)
        rects[:, :2] *= cat_img_occluded_numpy.shape[1]
        rects[:, 2:] *= cat_img_occluded_numpy.shape[0]
        rects = rects.astype(np.int)
        for rect in rects:
            cat_img_occluded_numpy[rect[2]:rect[3], rect[0]:rect[1]] = 0
        cat_img_occluded_tensor = convert_image(cat_img_occluded_numpy)
        cat_img_tensor = torch.cat((cat_img_tensor, cat_img_occluded_tensor))


    cat_img = Variable(cat_img_tensor)


