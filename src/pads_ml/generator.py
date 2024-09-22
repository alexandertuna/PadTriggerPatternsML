import numpy as np
import pandas as pd

from . import constants
from .pads import Pads
from .lines import Lines
from .traverser import Traverser

class SignalGenerator:
    def __init__(self, num: int, pads_filename: str):
        lines = Lines(num)
        pads = Pads(pads_filename)
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
    def __init__(self, num: int, pads_filename: str):
        pads = Pads(pads_filename)
        self.df = pd.DataFrame({
            f"pad_{layer}": np.random.randint(0, len(pads.layer[layer]), num)
            for layer in range(constants.LAYERS)
        })

