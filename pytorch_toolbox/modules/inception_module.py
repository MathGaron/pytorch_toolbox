import torch.nn as nn
import torch
import torch.nn.functional as F


class InceptionBase(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionBase, self).__init__()

        _channels1x1 = int(out_channels / 4)
        _channels3x3 = int(out_channels / 4)
        _channels5x5 = int(out_channels / 4)
        _channels_bc = int(out_channels / 4)

        self.branch1x1 = BasicConv2d(in_channels, _channels1x1, kernel_size=1)
        self.branch3x3 = BasicConv2d(in_channels, _channels3x3, kernel_size=3, padding=1)
        self.branch5x5 = BasicConv2d(in_channels, _channels5x5, kernel_size=5, padding=2)

        self.branch_pool = BasicConv2d(in_channels, _channels_bc, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
