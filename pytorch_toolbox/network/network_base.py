import torch.nn as nn
from abc import ABCMeta, abstractmethod


class NetworkBase(nn.Module):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, x):
        """
        Define forward as required by nn.module
        :param x:
        :return:
        """
        pass

    @abstractmethod
    def loss(self, predictions, targets):
        """
        Define criterion on which the train loop will call .backward().
        Has to return a single value
        :param predictions: List of network outputs : [output1, output2, ..., outputn]
        :param targets:     List of target labels : [label1, label2, ..., labeln]
        :return:
        """
        pass