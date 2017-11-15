from PIL import Image
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from examples.classification.cat_dog_net import CatDogNet
from pytorch_toolbox.transformations.image import Resize, Normalize, NumpyImage2Tensor
from pytorch_toolbox.transformations.to_float import ToFloat
from pytorch_toolbox.transformations.compose import Compose
from pytorch_toolbox.probe.activation import compare_activations

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
    # Occluded version of the image
    cat_img_occluded_numpy = cat_img_numpy.copy()
    cat_img_occluded_numpy[121:150, 70:135, :] = 0
    cat_img_occluded_numpy[20:100, 150:210, :] = 0
    cat_img_occluded_numpy[150:240, 140:220, :] = 0

    imagenet_mean = [123, 116, 103]
    imagenet_std = [58, 57, 57]
    transformations = Compose([Resize((128, 128)),
                                ToFloat(),
                                NumpyImage2Tensor(),
                                Normalize(mean=imagenet_mean, std=imagenet_std)])
    cat_img = Variable(transformations(cat_img_numpy).unsqueeze(0))
    cat_occluded_img = Variable(transformations(cat_img_occluded_numpy).unsqueeze(0))

    """
    Show activation difference between two images, in this case we compare a cat and an occluded cat
    """
    # Show input images
    titles = ["Non occluded cat", "Occluded cat"]
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(cat_img_numpy)
    axes[0].set_title(titles[0])
    axes[1].imshow(cat_img_occluded_numpy)
    axes[1].set_title(titles[1])
    fig.tight_layout()
    plt.show()

    # Show activation difference
    prediction, prediction_occ = compare_activations(model, cat_img, cat_occluded_img, cmin=0, cmax=2.5)
    plt.show()

    # Show predictions
    prediction = prediction.data.cpu().numpy()
    prediction_occ = prediction_occ.data.cpu().numpy()
    fig, axes = plt.subplots(2)
    predictions = [prediction, prediction_occ]
    for ax, prediction, title in zip(axes, predictions, titles):
        ax.bar(np.arange(2), prediction[0])
        ax.set_ylim([0, 1])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["cat", "dog"])
        ax.set_ylabel("probability")
        ax.set_title(title)
    plt.show()