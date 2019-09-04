import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pytorch_toolbox.train_loop import TrainLoop
from torch.utils import data

from pytorch_toolbox.transformations.image import Resize, Normalize, NumpyImage2Tensor
from pytorch_toolbox.transformations.to_float import ToFloat
from pytorch_toolbox.transformations.compose import Compose

if __name__ == '__main__':

    #
    #   Load configurations
    #
    parser = argparse.ArgumentParser(description='Eval Cat and Dog Example')
    parser.add_argument('-m', '--modelpath', help="Model path", required=True)
    parser.add_argument('-d', '--dataset', help="Dataset path", required=True)

    parser.add_argument('-k', '--backend', help="backend : cuda | cpu", action="store", default="cuda")
    arguments = parser.parse_args()

    data_path = arguments.dataset
    model_path = arguments.modelpath
    backend = arguments.backend

    #
    #   Instantiate models/loaders/etc.
    #
    model, loader_class = TrainLoop.load_from_output_directory(model_path)

    # Here we use the following transformations:
    # ToTensor = convert numpy to torch tensor (in float value between 0 and 1.0
    # Normalize = with respect to imagenet parameters
    imagenet_mean = [123, 116, 103]
    imagenet_std = [58, 57, 57]
    # transfformations are a series of transform to pass to the input data. Here we have to build a list of
    # transforms for each inputs to the network's forward call
    transformations = [Compose([Resize((128, 128)),
                                ToFloat(),
                                NumpyImage2Tensor(),
                                Normalize(mean=imagenet_mean, std=imagenet_std)])]

    dataset = loader_class(os.path.join(data_path, "valid"), transformations)
    dataset_loader = data.DataLoader(dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=1,
                                   )

    for inputs, gt in dataset_loader:
        output = model(inputs[0])
        prediction = np.argmax(np.exp(output[0].detach().cpu().numpy()))

        labels = ["cat", "dog"]
        inputs_np = inputs[0].cpu().numpy()[0].transpose((1, 2, 0))
        inputs_np = inputs_np * imagenet_std + imagenet_mean
        plt.imshow(inputs_np.astype(np.uint8))
        plt.title("GT : {}, Prediction : {}".format(labels[gt[0]], labels[prediction]))
        plt.show()


