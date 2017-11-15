from PIL import Image
from torch.autograd import Variable
import numpy as np
import math
import matplotlib.pyplot as plt
from examples.classification.cat_dog_net import CatDogNet
from pytorch_toolbox.transformations.image import Resize, Normalize, NumpyImage2Tensor
from pytorch_toolbox.transformations.to_float import ToFloat
from pytorch_toolbox.transformations.compose import Compose


def show_activations(feature, min=None, max=None):
    root = int(math.sqrt(feature.shape[1]))
    fig, axes = plt.subplots(root, root)
    for i in range(root):
        for j in range(root):
            ax = axes[i][j]
            ax.imshow(feature[0, i * root + j, :, :], vmin=min, vmax=max)
    plt.show()

if __name__ == '__main__':

    backend = "cpu"     # cpu|cuda
    model_path = "cat_dog_model.pth.tar"

    #
    #   Instantiate model and load checkpoint
    #
    model = CatDogNet()
    model.load(model_path)
    model.eval()
    idx2class = ["cat", "dog"]
    cat_path = "cat.jpg"
    dog_path = "dog.jpg"
    cat_img = Image.open(cat_path).convert('RGB')
    dog_img = Image.open(dog_path).convert('RGB')

    # Here we use the following transformations:
    # ToTensor = convert numpy to torch tensor (in float value between 0 and 1.0
    # Normalize = with respect to imagenet parameters
    imagenet_mean = [123, 116, 103]
    imagenet_std = [58, 57, 57]
    # transfformations are a series of transform to pass to the input data. Here we have to build a list of
    # transforms for each inputs to the network's forward call
    transformations = Compose([Resize((128, 128)),
                                ToFloat(),
                                NumpyImage2Tensor(),
                                Normalize(mean=imagenet_mean, std=imagenet_std)])

    cat_img = transformations(cat_img).unsqueeze(0)
    dog_img = transformations(dog_img).unsqueeze(0)

    prediction = model(Variable(cat_img)).data.cpu().numpy()

    activations = model.load_activations()

    for name, feature in activations.items():
        print(name)
        show_activations(feature)

    print(idx2class[np.argmax(prediction)])
