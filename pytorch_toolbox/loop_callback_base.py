import abc
import csv
import os

from pytorch_toolbox.logger import Logger
from pytorch_toolbox.train_state import TrainingState


class LoopCallbackBase(object):
    def __init__(self):
        self.epoch_training_logger = Logger()
        self.epoch_validation_logger = Logger()
        self.batch_logger = Logger()

    def epoch_(self, state: TrainingState):
        # update the epoch logs
        batch_average = self.batch_logger.get_averages()
        if state.training_mode:
            self.epoch_training_logger.set_dict(batch_average)
            self.epoch_training_logger["Loss"] = state.training_average_loss
        else:
            self.epoch_validation_logger.set_dict(batch_average)
            self.epoch_validation_logger["Loss"] = state.training_average_loss

        self.epoch_training_logger["Process Time"] = state.average_batch_processing_time
        self.epoch_training_logger["Load Time"] = state.average_data_loading_time
        self.epoch(state)
        self.batch_logger.reset()

    def batch_(self, state: TrainingState):
        self.batch(state)

    @abc.abstractmethod
    def batch(self, state: TrainingState):
        """
        Will be called after each minibatches
        :param state:       Contains information of the training state (see in train_loop.py)
        :return:
        """
        pass

    @abc.abstractmethod
    def epoch(self, state: TrainingState):
        """
        Function called at the end of each epoch.
        :param state:       Contains information of the training state (see in train_loop.py)
        :return:
        """
        pass

    def save_epoch_data(self, path: str, state: TrainingState):
        """
        Will add some state information to the logger before saving it
        :param path:
        :param state:
        :return:
        """
        if state.training_mode:
            string = "training"
            logger = self.epoch_training_logger
        else:
            string = "validation"
            logger = self.epoch_validation_logger

        filename = "{}_data.log".format(string)
        file_path = os.path.join(path, filename)
        logger.save(file_path)

    def print_batch_data(self, state: TrainingState, order=None):
        """
        Will make a pretty console print with given information
        :param state:
        :param order:  list of string containing the label of the data to show as extra. None will print them all
        :return:
        """
        mode = "Train" if state.training_mode else "Valid"
        loss = state.training_average_loss if state.training_mode else state.validation_average_loss
        print(
            ' {}\t || Loss: {:.3f} | Load Time {:.3f}s | Batch Time {:.3f}s'.format(mode,
                                                                                    loss,
                                                                                    state.average_data_loading_time,
                                                                                    state.average_batch_processing_time))
        batch_average = self.batch_logger.get_averages()
        labels = batch_average.keys()
        if order is not None:
            labels = order
        for label in labels:
            print('\t\t || {} : {:.3f}'.format(label, batch_average[label]))
