import numpy as np
import pandas as pd

from . import constants
from .pads import Pads
from .lines import Lines
from .traverser import Traverser

import logging
logger = logging.getLogger(__name__)

class SignalGenerator:
    def __init__(self, num: int, pads: Pads):

        logger.info("Creating lines and traversing pads")
        lines = Lines(num)
        traverser = Traverser(lines, pads)

        if len(lines.df) != len(traverser.df):
            raise ValueError(f"Lines and Traverser have different lengths: {len(lines.df)} != {len(traverser.df)}")

        # drop the per-layer columns from lines
        columns = [
            f"{coord}_{layer}"
            for coord in ["x", "y", "quad", "point"]
            for layer in range(constants.LAYERS)
        ]
        lines.df = lines.df.drop(columns=columns)

        logger.info("Merging lines and pad-traverser")
        self.df = lines.df.merge(
            traverser.df,
            how="left",
            left_index=True,
            right_index=True,
        )

class NoiseGenerator:
    """
    Generate noise events.
    In each layer, a noise pad is randomly selected from the pads in that layer
    """
    def __init__(self, num: int, pads: Pads):
        logger.info("Creating pads randomly")
        self.df = pd.DataFrame({
            f"pad_{layer}": np.random.choice(pads.layer[layer].index, size=num)
            for layer in range(constants.LAYERS)
        })

