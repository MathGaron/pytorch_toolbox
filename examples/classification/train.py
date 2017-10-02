import sys
import os
from multiprocessing import cpu_count

from torch import optim
from torch.utils import data

from examples.classification.loader import CatVsDogLoader
from examples.classification.net import CatVSDogNet
from examples.classification.cat_dog_callback import CatDogCallback
from pytorch_toolbox.io import yaml_loader
from pytorch_toolbox.train_loop import TrainLoop
from pytorch_toolbox.transformations.image import Resize, Normalize, NumpyImage2Tensor
from pytorch_toolbox.transformations.to_float import ToFloat
from pytorch_toolbox.transformations.compose import Compose


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

    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
    transformations = [Compose([Resize((128, 128)),
                                ToFloat(),
                                NumpyImage2Tensor(),
                                Normalize(mean=imagenet_mean, std=imagenet_std)])]

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
    # We can add any number of callbacks to handle data during training
    train_loop_handler.add_callback([CatDogCallback(10, train_dataset.idx_to_class, output_path)])
    train_loop_handler.loop(epochs, output_path)

    print("Training Complete")
