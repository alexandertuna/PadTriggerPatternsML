"""
Generate lines with origin at (0, 0) and randomly distributed with
  eta from [1.0, 2.5] and phi from [-pi/8, pi/8]
"""

import numpy as np
import pandas as pd
np.random.seed(42)
from shapely.geometry import Point

from . import constants


def y_to_quad(y: float) -> int:
    for quad in range(constants.QUADS):
        if y < constants.RMAXS[quad]:
            return quad
    raise ValueError(f"y {y} is out of range")


class Lines:

    def __init__(self, num: int):
        self.num = num

        self.df = pd.DataFrame()

        # Generate random eta and phi
        self.df["eta"] = np.random.uniform(constants.ETA_MIN, constants.ETA_MAX, self.num)
        self.df["phi"] = np.random.uniform(constants.PHI_MIN, constants.PHI_MAX, self.num)

        # Calculate stuff
        self.df["theta"] = 2 * np.arctan(np.exp(-self.df["eta"]))

        # Project the lines to each layer
        for layer in range(constants.LAYERS):
            self.df[f"x_{layer}"] = constants.ZS[layer] * np.tan(self.df["theta"]) * np.cos(self.df["phi"])
            self.df[f"y_{layer}"] = constants.ZS[layer] * np.tan(self.df["theta"]) * np.sin(self.df["phi"])
            self.df[f"quad_{layer}"] = self.df[f"y_{layer}"].apply(y_to_quad)
            self.df[f"point_{layer}"] = self.df.apply(lambda row: Point(row[f"x_{layer}"], row[f"y_{layer}"]), axis=1)

