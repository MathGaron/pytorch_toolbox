import matplotlib.pyplot as plt
from examples.classification.cat_dog_net import CatDogNet
from pytorch_toolbox.probe.runtime import compute_test_time, compute_train_time

if __name__ == '__main__':

    #
    #   Instantiate model and load checkpoint
    #
    max_batch_size = 50
    step_size = 1
    network_class = CatDogNet
    input_size = (3, 128, 128)

    compute_test_time(network_class, input_size, max_batch_size, step_size, is_cuda=True)
    plt.show()

    compute_train_time(network_class, input_size, max_batch_size, step_size, is_cuda=True, backward_only=False)
    plt.show()

