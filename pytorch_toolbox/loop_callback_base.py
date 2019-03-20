import abc
import csv

from pytorch_toolbox.train_state import TrainingState


class LoopCallbackBase(object):
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

    @staticmethod
    def file_print(filename: str, state: TrainingState, extra_data: list):
        """
        Will print the data in a csv file
        :param filename:    File name to save
        :param state:
        :param extra_data:  Any extra stuff to save
        :return:
        """
        f = open(filename, 'a')
        loss = state.training_average_loss if state.training_mode else state.validation_average_loss
        try:
            writer = csv.writer(f)
            writer.writerow([state.average_data_loading_time, state.average_batch_processing_time, loss] + extra_data)
        finally:
            f.close()

    @staticmethod
    def console_print(state: TrainingState, extra_data: list):
        """
        Will make a pretty console print with given information
        :param state:
        :param extra_data:  list of extra data
        :return:
        """
        mode = "Train" if state.training_mode else "Valid"
        loss = state.training_average_loss if state.training_mode else state.validation_average_loss
        print(
            ' {}\t || Loss: {:.3f} | Load Time {:.3f}s | Batch Time {:.3f}s'.format(mode,
                                                                                    loss,
                                                                                    state.average_data_loading_time,
                                                                                    state.average_batch_processing_time))
        for i, acc in enumerate(extra_data):
            print('\t || Acc {}: {:.3f}'.format(i, acc))
