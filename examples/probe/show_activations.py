from PIL import Image
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from examples.classification.cat_dog_net import CatDogNet
from pytorch_toolbox.transformations.image import Resize, Normalize, NumpyImage2Tensor
from pytorch_toolbox.transformations.to_float import ToFloat
from pytorch_toolbox.transformations.compose import Compose
from pytorch_toolbox.probe.activation import show_activations

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

    imagenet_mean = [123, 116, 103]
    imagenet_std = [58, 57, 57]
    transformations = Compose([Resize((128, 128)),
                                ToFloat(),
                                NumpyImage2Tensor(),
                                Normalize(mean=imagenet_mean, std=imagenet_std)])
    cat_img = Variable(transformations(cat_img_numpy).unsqueeze(0))

    """
    Show activation of input image.
    """
    prediction = model(cat_img)
    activations = model.load_activations()
    for name, feature in activations.items():
        show_activations(feature[0], name)
    plt.show()

