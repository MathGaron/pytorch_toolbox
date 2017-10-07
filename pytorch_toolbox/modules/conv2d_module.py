import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    """
    Generic convolution block

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dropout=False, dropout_rate=0.25,
                 batchnorm=True, maxpool=True, maxpool_size=2, activation=None):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dropout:
        :param dropout_rate:
        :param batchnorm:
        :param maxpool:
        :param maxpool_size:
        :param activation:      Function from function api
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batchnorm else None
        self.dropout = nn.Dropout2d(dropout_rate) if dropout else None
        self.maxpool = nn.MaxPool2d(maxpool_size) if maxpool else None
        self.activation = activation

    def forward(self, x):

        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)
        if self.maxpool:
            x = self.maxpool(x)
        if self.dropout:
            x = self.dropout(x)
        return x