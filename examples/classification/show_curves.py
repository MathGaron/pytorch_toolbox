from pytorch_toolbox.logger import Logger
import matplotlib.pyplot as plt

from pytorch_toolbox.visualization.train_plots import train_plot

if __name__ == '__main__':
    train_log = Logger()
    valid_log = Logger()

    train_log.load("training_data.log")
    valid_log.load("validation_data.log")

    train_plot(train_log, valid_log)
    plt.show()