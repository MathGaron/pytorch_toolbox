import sys
import os
import numpy as np
from multiprocessing import cpu_count

import torch
from torch import optim
from torch.utils import data

from examples.classification.loader import CatVsDogLoader
from examples.classification.net import CatVSDogNet
from pytorch_toolbox.io import yaml_loader
from pytorch_toolbox.utils import classification_accuracy
from pytorch_toolbox.train_loop import TrainLoop
import pytorch_toolbox.data_transforms as dt
from pytorch_toolbox.visualization.epoch_callbacks import visdom_print, console_print
from pytorch_toolbox.visualization.visdom_handler import VisdomHandler


def classification_accuracy_callback(prediction, target):
    """
    This is a simple callback that compute the score accuracy
    Note that the callback take as input every output of the network/target, here we explicitly use the first output
    which is the class from classification branch
    :param prediction:
    :param target:
    :return:
    """
    prec1, _ = classification_accuracy(prediction[0].data, target[0], top_k=(1, 1))
    return prec1[0]


class batch_visualization_callback:
    """
    This callback class will remember the number of time it is called. This way we do not update the image at every
    batch. It also keep a dict idx_to_class for visualization purpose
    """
    def __init__(self, idx_to_class, update_rate=10):
        self.count = 0
        self.update_rate = update_rate
        self.idx_to_class = idx_to_class

    def __call__(self, prediction, data, target, istrain):
        """
        This is a simple callback that send some results to visdom
        :param prediction:
        :param target:
        :return:
        """

        if self.count % self.update_rate == 0:
            vis = VisdomHandler()

            # Unormalize an image and convert it to uint8
            img = data[0][0].cpu().numpy()
            std = np.array([58, 57, 57], dtype=np.float32)
            mean = np.array([123, 116, 103], dtype=np.float32)
            std = std[:, np.newaxis, np.newaxis]
            mean = mean[:, np.newaxis, np.newaxis]
            img = img * std + mean
            img = img.astype(np.uint8)

            # log softmax output to class string
            prediction_index = np.argmax(prediction[0][0].data.cpu().numpy())
            prediction_class = self.idx_to_class[prediction_index]

            # send to visdom with prediction as caption
            vis.visualize(img, "test", caption="prediction : {}".format(prediction_class))

        self.count += 1


if __name__ == '__main__':

    #
    #   Load configurations
    #
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "train_config.yml"
    configs = yaml_loader(config_path)

    data_path = configs["data_path"]
    output_path = configs["output_path"]
    backend = configs["backend"]
    batch_size = configs["batch_size"]
    epochs = configs["epochs"]
    use_shared_memory = configs["use_shared_memory"] == "True"
    number_of_core = int(configs["number_of_core"])
    learning_rate = float(configs["learning_rate"])

    if number_of_core == -1:
        number_of_core = cpu_count()

    #
    #   Instantiate models/loaders/etc.
    #
    model = CatVSDogNet()
    loader_class = CatVsDogLoader

    # Here we use the following transformations:
    # ToTensor = convert numpy to torch tensor (in float value between 0 and 1.0
    # Normalize = with respect to imagenet parameters
    imagenet_mean = [123, 116, 103]
    imagenet_std = [58, 57, 57]
    # transfformations are a series of transform to pass to the input data. Here we have to build a list of
    # transforms for each inputs to the network's forward call
    transformations = [dt.Compose([dt.Resize((128, 128)),
                                   dt.ToFloat(),
                                   dt.NumpyImage2Tensor(),
                                   dt.Normalize(mean=imagenet_mean, std=imagenet_std)])]

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_dataset = loader_class(os.path.join(data_path, "train"), transformations)
    valid_dataset = loader_class(os.path.join(data_path, "valid"), transformations)

    # Instantiate the data loader needed for the train loop. These use dataset object to build random minibatch
    # on multiple cpu core
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=number_of_core,
                                   pin_memory=use_shared_memory,
                                   drop_last=True,
                                   )

    val_loader = data.DataLoader(valid_dataset,
                                 batch_size=batch_size,
                                 num_workers=number_of_core,
                                 pin_memory=use_shared_memory,
                                 )

    # Instantiate the train loop and train the model.
    train_loop_handler = TrainLoop(model, train_loader, val_loader, optimizer, backend)
    # We can add any number of callback to compute score or any meanful value from the predictions
    train_loop_handler.add_score_callback([classification_accuracy_callback])
    # We can add any number of callbacks to handle epoch's data (loss, timings, scores)
    train_loop_handler.add_epoch_callback([console_print, visdom_print])
    train_loop_handler.add_batch_callback([batch_visualization_callback(train_dataset.idx_to_class)])
    train_loop_handler.loop(epochs, output_path)

    print("Training Complete")
