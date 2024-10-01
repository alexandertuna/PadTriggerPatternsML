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
        logger.info("Creating lines with randomly sampled parameters")

        use_atlas = False

        # use ATLAS eta distribution
        if use_atlas:

            # generate and trim
            eta = np.random.normal(constants.ATLAS_ETA_MU, constants.ATLAS_ETA_SIGMA, 10 * self.num)
            eta = eta[ (eta > constants.ETA_MIN) & (eta < constants.ETA_MAX) ][:self.num]

            # calculate
            self.df["eta"] = eta
            self.df["phi"] = np.random.uniform(constants.PHI_MIN, constants.PHI_MAX, self.num)
            self.df["theta"] = 2 * np.arctan(np.exp(-self.df["eta"]))

        # use uniform x,y distribution
        else:

            # generate more than necessary (some will fall outside bounds)
            x_mid = np.random.uniform(-1100, 1100, 10 * self.num)
            y_mid = np.random.uniform(900, 4800, 10 * self.num)
            z_mid = constants.ZMID
            phi = np.arctan2(y_mid, x_mid)
            indices = (phi > constants.PHI_MIN) & (phi < constants.PHI_MAX)

            # trim to num
            x_mid = x_mid[indices][:self.num]
            y_mid = y_mid[indices][:self.num]
            self.df["phi"] = phi[indices][:self.num]

            # calculate
            r = np.sqrt(x_mid**2 + y_mid**2)
            self.df["theta"] = np.arctan2(r, z_mid)
            self.df["eta"] = -np.log(np.tan(self.df["theta"] / 2))

        # Project the lines to each layer, and smear
        logger.info("Projecting lines to each layer")
        for layer in range(constants.LAYERS):
            self.df[f"x_{layer}"] = constants.ZS[layer] * np.tan(self.df["theta"]) * np.cos(self.df["phi"]) + np.random.normal(0, smear, self.num)
            self.df[f"y_{layer}"] = constants.ZS[layer] * np.tan(self.df["theta"]) * np.sin(self.df["phi"]) + np.random.normal(0, smear, self.num)
            self.df[f"quad_{layer}"] = self.df[f"y_{layer}"].apply(y_to_quad)
            self.df[f"point_{layer}"] = gpd.points_from_xy(self.df[f"x_{layer}"], self.df[f"y_{layer}"])

