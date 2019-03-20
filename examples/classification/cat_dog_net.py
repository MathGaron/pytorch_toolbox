import torch.nn.functional as F
import torch.nn as nn
from pytorch_toolbox.network_base import NetworkBase
from pytorch_toolbox.modules.conv2d_module import ConvBlock
from pytorch_toolbox.modules.fc_module import FCBlock


class CatDogNet(NetworkBase):
    def __init__(self):
        super(CatDogNet, self).__init__()
        self.conv1 = ConvBlock(3, 24, 5, dropout=True, batchnorm=True, maxpool=True, activation=F.elu)
        self.conv2 = ConvBlock(24, 48, 3, dropout=True, batchnorm=True, maxpool=True, activation=F.elu)
        self.conv3 = ConvBlock(48, 48, 3, dropout=True, batchnorm=True, maxpool=True, activation=F.elu)
        self.conv4 = ConvBlock(48, 48, 3, dropout=True, batchnorm=True, maxpool=True, activation=F.elu)

        self.view_size = 48 * 6 * 6

        self.fc1 = FCBlock(self.view_size, 250, dropout=True, batchnorm=True, activation=F.elu)
        self.fc2 = FCBlock(250, 2, dropout=False, batchnorm=False, activation=None)

        self.criterion = nn.NLLLoss()

    def forward(self, x):
        x = self.conv1(x)
        self.probe_activation["conv1"] = x
        x = self.conv2(x)
        self.probe_activation["conv2"] = x
        x = self.conv3(x)
        self.probe_activation["conv3"] = x
        x = self.conv4(x)
        self.probe_activation["conv4"] = x

        x = x.view(-1, self.view_size)
        x = self.fc1(x)
        self.probe_activation["lin1"] = x
        x = self.fc2(x)
        if self.training:
            x = F.log_softmax(x)
        else:
            x = F.softmax(x)
        return x

    def loss(self, predictions, targets):
        return self.criterion(predictions[0], targets[0])