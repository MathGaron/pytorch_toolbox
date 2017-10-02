import abc
import csv


class LoopCallbackBase(object):
    @abc.abstractmethod
    def batch(self, predictions, network_inputs, targets, isvalid=True):
        """
        Will be called after each minibatches
        :param predictions:
        :param network_inputs:
        :param targets:
        :param isvalid:
        :return:
        """
        pass

    @abc.abstractmethod
    def epoch(self, loss, data_time, batch_time, isvalid=True):
        """
        Function called at the end of each epoch.
        :param loss:        average loss
        :param data_time:   average data load time
        :param batch_time:  average batch processing time
        :param isvalid:
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
    def console_print(loss, data_time, batch_time, extra_data, isvalid):
        """
        Will make a pretty console print with given information
        :param loss:        average loss during epoch
        :param data_time:   average data load time during epoch
        :param batch_time:  average batch load time during epoch
        :param extra_data:      list of extra data
        :param isvalid:
        :return:
        """
        state = "Valid" if isvalid else "Train"
        print(
            ' {}\t || Loss: {:.3f} | Load Time {:.3f}s | Batch Time {:.3f}s'.format(state, loss, data_time, batch_time))
        for i, acc in enumerate(extra_data):
            print('\t || Acc {}: {:.3f}'.format(i, acc))

    @staticmethod
    def visdom_print(loss, data_time, batch_time, extra_data, isvalid):
        """
        Will send the information to visdom for visualisation
        :param loss:
        :param data_time:
        :param batch_time:
        :param extra_data:
        :param istrain:
        :return:
        """
        from pytorch_toolbox.visualization.visdom_handler import VisdomHandler

        state = "Valid" if isvalid else "Train"
        vis = VisdomHandler()
        vis.visualize(loss, '{} loss'.format(state))
        vis.visualize(data_time, '{} data load time'.format(state))
        vis.visualize(batch_time, '{} batch processing time'.format(state))
        for i, acc in enumerate(extra_data):
            vis.visualize(acc, '{} score {}'.format(i, state))
