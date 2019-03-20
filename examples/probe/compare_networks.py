from PIL import Image
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from examples.classification.cat_dog_net import CatDogNet
from pytorch_toolbox.transformations.image import Resize, Normalize, NumpyImage2Tensor
from pytorch_toolbox.transformations.to_float import ToFloat
from pytorch_toolbox.transformations.compose import Compose
from pytorch_toolbox.probe.activation import show_activations, compare_networks_activations

if __name__ == '__main__':

    backend = "cpu"     # cpu|cuda
    model_A_path = "cat_dog_model.pth.tar"
    model_B_path = "cat_dog_model_1_epoch.pth.tar"

    #
    #   Instantiate model and load checkpoint
    #
    modelA = CatDogNet()
    modelA.load(model_A_path)
    modelA.eval()
    modelB = CatDogNet()
    modelB.load(model_B_path)
    modelB.eval()


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
    predictionA, predictionB = compare_networks_activations(modelA, modelB, cat_img,
                                                            error_func=lambda a, b: np.abs(a-b),
                                                            cmin=0, cmax=2.5)
    plt.show()

    # Show predictions
    prediction = predictionA.data.cpu().numpy()
    prediction_occ = predictionB.data.cpu().numpy()
    fig, axes = plt.subplots(2)
    predictions = [prediction, prediction_occ]
    for ax, prediction in zip(axes, predictions):
        ax.bar(np.arange(2), prediction[0])
        ax.set_ylim([0, 1])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["cat", "dog"])
        ax.set_ylabel("probability")
    plt.show()
