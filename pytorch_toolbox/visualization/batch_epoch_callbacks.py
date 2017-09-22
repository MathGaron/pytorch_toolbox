import abc


class LoopCallbackBase(object):

    def __init__(self):
        '''We could save the states and temporal data
        self.epoch_data = []
        self.output_path = ''
        '''

    @abc.abstractmethod
    def batch(prediction, data_feeded, target, istest=True):
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
