import abc


class LoopCallbackBase(object):

    def __init__(self):
        '''We could save the states and temporal data
        self.epoch_data = []
        self.output_path = ''
        '''

    @abc.abstractmethod
    def batch(self, prediction, data_feeded, target, istest=True):
        ''' After each minibatch, .batch() will be executed, return values could be processed later
            Input:  prediction, directly from the net
                    data_feeded, data feed to the network
                    target, target / ground truth
            Return: each call() can return any data type
            After each minibatch, the return will be appended to a list,
            which could be further processed in the epoch() function
        '''
        pass

    @abc.abstractmethod
    def epoch(self, list_of_batch_returns, istest=True):
        ''' Catch up the output from batch()
            After each epoch, .epoch() will be executed, there is no return values
            Input: list of data returned from batch().
                   As in the e.g.: [np.arrray(2, 3), {'key':values}, ...], length is the number of batches

            Keep the historical data at each epoch, then you can write all of them onto disk
        '''
        pass

    @staticmethod
    def console_print(loss, data_time, batch_time, scores, istrain):
        """
        Will make a pretty console print with given information
        :param loss:        average loss during epoch
        :param data_time:   average data load time during epoch
        :param batch_time:  average batch load time during epoch
        :param scores:      list of scores
        :param istrain:
        :return:
        """
        state = "Train" if istrain else "Valid"
        print(
            ' {}\t || Loss: {:.3f} | Load Time {:.3f}s | Batch Time {:.3f}s'.format(state, loss, data_time, batch_time))
        for i, acc in enumerate(scores):
            print('\t || Acc {}: {:.3f}'.format(i, acc))

    @staticmethod
    def visdom_print(loss, data_time, batch_time, scores, istrain):
        """
        Will send the information to visdom for visualisation
        :param loss:
        :param data_time:
        :param batch_time:
        :param scores:
        :param istrain:
        :return:
        """
        from pytorch_toolbox.visualization.visdom_handler import VisdomHandler

        state = "Train" if istrain else "Valid"
        vis = VisdomHandler()
        vis.visualize(loss, '{} loss'.format(state))
        vis.visualize(data_time, '{} data load time'.format(state))
        vis.visualize(batch_time, '{} batch processing time'.format(state))
        for i, acc in enumerate(scores):
            vis.visualize(acc, '{} score {}'.format(i, state))
