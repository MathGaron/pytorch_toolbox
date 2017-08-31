import abc


class batch_epoch_callback_base(object):

    def __init__(self):
        '''We could define anything, to save the temporal data'''
        self.epoch_data = []

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
        raise NotImplementedError
        return None  # any type of data; e.g.: np.arrray(2, 3), {'key':values}

    @abc.abstractmethod
    def epoch(self, list_of_batch_returns, istest=True):
        ''' Catch up the output from batch()
            After each epoch, .epoch() will be executed, there is no return values
            Input: list of data returned from batch().
            as in the e.g.: [np.arrray(2, 3), {'key':values}, ...], length is the number of batches
            This is different than the score_callbacks, which has return
        '''
        # keep the historical data at each epoch, then you can write all of them into disk
        self.epoch_data.append(None)
        pass
