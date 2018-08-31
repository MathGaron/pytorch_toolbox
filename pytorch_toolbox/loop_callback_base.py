import abc
import csv


class LoopCallbackBase(object):
    @abc.abstractmethod
    def batch(self, predictions, network_inputs, targets, info, is_train=True, tensorboard_logger=None):
        """
        Will be called after each minibatches
        :param predictions:
        :param network_inputs:
        :param targets:
        :param is_train:
        :param tensorboard_logger:
        :return:
        """
        pass

    @abc.abstractmethod
    def epoch(self, epoch, loss, data_time, batch_time, is_train=True, tensorboard_logger=None):
        """
        Function called at the end of each epoch.
        :param epoch:       current epoch
        :param loss:        average loss
        :param data_time:   average data load time
        :param batch_time:  average batch processing time
        :param is_train:    in training stage or valication stage
        :param tensorboard_logger: TensorboardLogger from ./visualization
        :return:
        """
        pass

    @staticmethod
    def file_print(filename, loss, data_time, batch_time, extra_data):
        """
        Will print the data in a csv file
        :param filename:
        :param loss:
        :param data_time:
        :param batch_time:
        :param extra_data:
        :return:
        """
        f = open(filename, 'a')
        try:
            writer = csv.writer(f)
            writer.writerow([data_time, batch_time, loss] + extra_data)
        finally:
            f.close()

    @staticmethod
    def console_print(loss, data_time, batch_time, extra_data, is_train):
        """
        Will make a pretty console print with given information
        :param loss:        average loss during epoch
        :param data_time:   average data load time during epoch
        :param batch_time:  average batch load time during epoch
        :param extra_data:  list of extra data
        :param isvalid:
        :return:
        """
        state = "Train" if is_train else "Valid"
        print(
            ' {}\t || Loss: {:.3f} | Load Time {:.3f}s | Batch Time {:.3f}s'.format(state, loss, data_time, batch_time))
        for i, acc in enumerate(extra_data):
            print('\t || Acc {}: {:.3f}'.format(i, acc))
