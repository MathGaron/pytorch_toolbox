import argparse
import os
from multiprocessing import cpu_count

from torch import optim
from torch.utils import data

from examples.classification.cat_dog_loader import CatDogLoader
from examples.classification.cat_dog_net import CatDogNet
from examples.classification.cat_dog_callback import CatDogCallback
from pytorch_toolbox.train_loop import TrainLoop
from pytorch_toolbox.transformations.image import Resize, Normalize, NumpyImage2Tensor
from pytorch_toolbox.transformations.to_float import ToFloat
from pytorch_toolbox.transformations.compose import Compose


if __name__ == '__main__':

    #
    #   Load configurations
    #
    parser = argparse.ArgumentParser(description='Train Cat and Dog Example')
    parser.add_argument('-o', '--output', help="Output path", required=True)
    parser.add_argument('-d', '--dataset', help="Dataset path", required=True)

    parser.add_argument('-l', '--learningrate', help="learning rate", action="store", default=0.001, type=float)
    parser.add_argument('-m', '--sharememory', help="Activate share memory", action="store_true")
    parser.add_argument('-b', '--loadbest', help="Load best model before training", action="store_true")
    parser.add_argument('-n', '--ncore', help="number of cpu core to use, -1 is all core", action="store", default=-1, type=int)
    parser.add_argument('-g', '--gradientclip', help="Activate gradient clip", action="store_true")
    parser.add_argument('-e', '--epoch', help="number of epoch", action="store", default=25, type=int)
    parser.add_argument('-k', '--backend', help="backend : cuda | cpu", action="store", default="cuda")
    parser.add_argument('-s', '--batchsize', help="Size of minibatch", action="store", default=64, type=int)
    parser.add_argument('-t', '--tensorboard', help="Size of minibatch", action="store_true")

    arguments = parser.parse_args()

    data_path = arguments.dataset
    output_path = arguments.output
    backend = arguments.backend
    batch_size = arguments.batchsize
    epochs = arguments.epoch
    use_shared_memory = arguments.sharememory
    number_of_core = arguments.ncore
    learning_rate = arguments.learningrate
    load_best = arguments.loadbest
    gradient_clip = arguments.gradientclip
    use_tensorboard = arguments.tensorboard

    if number_of_core == -1:
        number_of_core = cpu_count()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #
    #   Instantiate models/loaders/etc.
    #
    model = CatDogNet()
    loader_class = CatDogLoader

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
    train_loop_handler = TrainLoop(model, train_loader, val_loader, optimizer, backend, gradient_clip, use_tensorboard=use_tensorboard)
    # We can add any number of callbacks to handle data during training
    train_loop_handler.add_callback([CatDogCallback(10, train_dataset.idx_to_class, output_path, reset_files=not load_best)])
    train_loop_handler.loop(epochs, output_path, load_best_checkpoint=load_best)

    print("Training Complete")
