import math
import matplotlib.pyplot as plt
import numpy as np


def show_activations(feature, title="", min=None, max=None):
    """
    Plot all channels in a matplotlib figure
    :param feature: numpy array [c, h, w]
    :param title:   figure title
    :param min:     minimum value in colormap
    :param max:     maximum value in colormap
    :return:
    """
    if len(feature.shape) != 3:
        # todo : should handle vector activation?
        return
    root = int(math.sqrt(feature.shape[0]))
    fig, axes = plt.subplots(root, root)
    fig.suptitle(title, fontsize="x-large")
    for i in range(root):
        for j in range(root):
            ax = axes[i][j]
            ax.imshow(feature[i * root + j, :, :], vmin=min, vmax=max)


def compare_activations(network, imageA, imageB, error_func=np.subtract, cmin=None, cmax=None):
    """
    Compare activation of two images given a network.
    :param network:     network_base child
    :param imageA:      autograd variable
    :param imageB:      autograd variable
    :param error_func:  the error function takes two input : numpy features [c, h, w]
    :return:
    """
    predictionA = network(imageA)
    activationsA = network.load_activations()

    predictionB = network(imageB)
    activationsB = network.load_activations()

    for name in activationsA.keys():
        error = error_func(activationsA[name][0], activationsB[name][0])
        show_activations(error, name, min=cmin, max=cmax)
    return predictionA, predictionB


def compare_networks_activations(networkA, networkB, image, error_func=np.subtract, cmin=None, cmax=None):
    predictionA = networkA(image)
    activationsA = networkA.load_activations()

    predictionB = networkB(image)
    activationsB = networkB.load_activations()

    for name in activationsA.keys():
        error = error_func(activationsA[name][0], activationsB[name][0])
        show_activations(error, name, min=cmin, max=cmax)
    return predictionA, predictionB
