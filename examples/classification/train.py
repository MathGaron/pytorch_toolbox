import sys
import os
from multiprocessing import cpu_count

from torch import nn, optim
from torch.utils import data

from examples.classification.loader import CatVsDogLoader
from examples.classification.net import CatVSDogNet
from pytorch_toolbox.io import yaml_loader
from pytorch_toolbox.utils import classification_accuracy
from pytorch_toolbox.train_loop import TrainLoop
import pytorch_toolbox.data_transforms as dt


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
    criterion = [nn.NLLLoss()]                      # Here we just want a classification loss for the signle output
    callbacks = [classification_accuracy_callback]  # Here we add a callback that will use the predictions and targets
                                                    # and compute the % prediction accuracy

    # Here we use the following transformations:
    # ToTensor = convert numpy to torch tensor (in float value between 0 and 1.0
    # Normalize = with respect to imagenet parameters
    imagenet_mean = [123, 116, 103]
    imagenet_std = [58, 57, 57]
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
    train_loop_handler = TrainLoop(model, train_loader, val_loader, criterion, optimizer, backend)
    train_loop_handler.add_prediction_callback(callbacks)
    train_loop_handler.loop(epochs, output_path)

    print("Training Complete")
