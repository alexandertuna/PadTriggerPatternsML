"""
Generate lines with origin at (0, 0) and randomly distributed with
  eta from [1.0, 2.5] and phi from [-pi/8, pi/8]
"""

import numpy as np
import pandas as pd
import geopandas as gpd
# np.random.seed(42)
from shapely.geometry import Point

from pads_ml import constants

import logging
logger = logging.getLogger(__name__)


def y_to_quad(y: float) -> int:
    for quad in range(constants.QUADS):
        if y < constants.RMAXS[quad]:
            return quad
    raise ValueError(f"y {y} is out of range")


class Lines:

    def __init__(self, num: int, smear: float = 0.0) -> None:

        self.num = num
        self.df = pd.DataFrame()

        # use ATLAS eta distribution
        logger.info("Creating lines with randomly sampled parameters")
        eta = np.random.normal(constants.ATLAS_ETA_MU, constants.ATLAS_ETA_SIGMA, 10 * self.num)
        eta = eta[ (eta > constants.ETA_MIN) & (eta < constants.ETA_MAX) ][:self.num]

        # Generate random r (mid) and phi
        self.df["eta"] = eta
        self.df["phi"] = np.random.uniform(constants.PHI_MIN, constants.PHI_MAX, self.num)

        # Calculate stuff
        self.df["theta"] = 2 * np.arctan(np.exp(-self.df["eta"]))

        # Project the lines to each layer
        logger.info("Projecting lines to each layer")
        for layer in range(constants.LAYERS):
            self.df[f"x_{layer}"] = constants.ZS[layer] * np.tan(self.df["theta"]) * np.cos(self.df["phi"]) + np.random.normal(0, smear, self.num)
            self.df[f"y_{layer}"] = constants.ZS[layer] * np.tan(self.df["theta"]) * np.sin(self.df["phi"]) + np.random.normal(0, smear, self.num)
            self.df[f"quad_{layer}"] = self.df[f"y_{layer}"].apply(y_to_quad)
            self.df[f"point_{layer}"] = gpd.points_from_xy(self.df[f"x_{layer}"], self.df[f"y_{layer}"])

