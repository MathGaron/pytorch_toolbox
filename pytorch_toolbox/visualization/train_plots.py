import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


def train_plot(training_logger, validation_logger, order=None):

    keys = list(training_logger.keys())
    keys.remove("Process Time")
    keys.remove("Load Time")

    if order is not None:
        keys = order

    for i, key in enumerate(keys):
        plt.subplot(len(keys), 1, i+1)
        X = np.arange(len(training_logger[key]))
        ax = sns.lineplot(X, training_logger[key], color="blue", label="Train", marker="o")
        ax = sns.lineplot(X, validation_logger[key], color="red", label="Valid", marker="o")
        ax.legend()
        plt.title(key)
