import torch.nn as nn
import torch


class FCBlock(nn.Module):
    """
        generic Fully connected block
    """
    def __init__(self, in_features, out_features, dropout=True, dropout_rate=0.5,
                 batchnorm=True, activation=None):
        """

        :param in_features:
        :param out_features:
        :param dropout:
        :param dropout_rate:
        :param batchnorm:
        :param activation:      Function from function api
        """
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features) if batchnorm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout else None
        self.activation = activation

    def forward(self, x):

        x = self.fc(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x