import torch.nn as nn
from abc import ABCMeta, abstractmethod


class NetworkBase(nn.Module):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def loss(self, predictions, targets):
        pass