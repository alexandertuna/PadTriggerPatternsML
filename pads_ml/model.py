"""
Model architectures for pads classification
"""

from torch import nn
from torch.nn import functional as F

from pads_ml import constants

import logging
logger = logging.getLogger(__name__)

class OneHotFullyConnected(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(constants.PADS, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        logger.info(f"Model parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x):
        return self.model(x)
