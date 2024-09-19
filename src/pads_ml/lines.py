"""
Generate lines with origin at (0, 0) and randomly distributed with
  eta from [1.0, 2.5] and phi from [-pi/8, pi/8]
"""

import numpy as np
import pandas as pd

from . import constants

class Lines:

    def __init__(self, num: int):
        self.num = num

        self.df = pd.DataFrame()

        # Generate random eta and phi
        self.df["eta"] = np.random.uniform(constants.ETA_MIN, constants.ETA_MAX, self.num)
        self.df["phi"] = np.random.uniform(constants.PHI_MIN, constants.PHI_MAX, self.num)

        # Calculate stuff
        self.df["theta"] = 2 * np.arctan(np.exp(-self.df["eta"]))
        for layer in range(constants.LAYERS):
            self.df[f"x_at_layer_{layer}"] = constants.ZS[layer] * np.tan(self.df["theta"]) * np.cos(self.df["phi"])
            self.df[f"y_at_layer_{layer}"] = constants.ZS[layer] * np.tan(self.df["theta"]) * np.sin(self.df["phi"])

