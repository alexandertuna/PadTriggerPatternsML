"""
Model architectures for pads classification
"""

import torch
from torch import nn
from torch.nn import functional as F

from . import constants

class OneHotFullyConnected(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc_0 = nn.Linear(constants.PADS, 256)
        self.fc_1 = nn.Linear(256, 128)
        self.fc_2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc_0(x))
        x = F.relu(self.fc_1(x))
        x = torch.sigmoid(self.fc_2(x))
        return x
