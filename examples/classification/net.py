import torch.nn.functional as F
import torch.nn as nn


class CatVSDogNet(nn.Module):
    def __init__(self):
        super(CatVSDogNet, self).__init__()
        pass

    def forward(self, x):
        return x