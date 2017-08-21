import torch.nn.functional as F
import torch.nn as nn
from pytorch_toolbox.network.network_base import NetworkBase


class CatVSDogNet(NetworkBase):
    def __init__(self):
        super(CatVSDogNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 5)
        self.conv1_bn = nn.BatchNorm2d(24)
        self.dropout1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(24, 48, 3)
        self.conv2_bn = nn.BatchNorm2d(48)
        self.dropout2 = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(48, 48, 3)
        self.conv3_bn = nn.BatchNorm2d(48)
        self.dropout3 = nn.Dropout2d(0.25)
        self.conv4 = nn.Conv2d(48, 96, 3)
        self.conv4_bn = nn.BatchNorm2d(96)

        self.view_size = 96 * 6 * 6
        self.fc1 = nn.Linear(self.view_size, 250)
        self.fc_bn1 = nn.BatchNorm1d(250)
        self.fc2 = nn.Linear(250, 2)

        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()

        self.criterion = nn.NLLLoss()

    def forward(self, x):
        x = self.dropout1(F.max_pool2d(F.elu(self.conv1(x)), 2))
        x = self.dropout2(F.max_pool2d(F.elu(self.conv2_bn(self.conv2(x))), 2))
        x = self.dropout3(F.max_pool2d(F.elu(self.conv3_bn(self.conv3(x))), 2))
        x = F.max_pool2d(F.elu(self.conv4_bn(self.conv4(x))), 2)
        x = x.view(-1, self.view_size)
        x = self.dropout1(x)
        x = F.elu(self.fc_bn1(self.fc1(x)))
        x = self.dropout2(x)
        x = F.log_softmax(self.fc2(x))
        return x

    def loss(self, predictions, targets):
        return self.criterion(predictions[0], targets[0])
