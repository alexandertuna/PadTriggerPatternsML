import numpy as np
import pandas as pd

from pads_ml import constants
from pads_ml.pads import Pads
from pads_ml.lines import Lines
from pads_ml.traverser import Traverser
from pads_ml.preprocess import DataPreparer

import logging
logger = logging.getLogger(__name__)

class EverythingGenerator:

    def __init__(
            self,
            num: int,
            pads: Pads,
            smear: float,
            onehot: bool,
            bkg: str,
        ) -> None:

        self.onehot = onehot

        logger.info("Creating signal")
        self.signal = SignalGenerator(num, pads, smear)

        logger.info("Creating noise")
        if bkg == "random":
            self.noise = NoiseGenerator(num, pads)
        elif bkg == "smear":
            self.noise = SignalGenerator(num, pads, smear*5)
        else:
            raise ValueError(f"Unknown background type: {bkg}")

        if self.onehot:
            logger.info("Preparing data")
            dp = DataPreparer(self.signal.df, self.noise.df)
            dp.prepare()
            self.features = dp.features
            self.labels = dp.labels


    def save(self, prefix: str) -> None:
        logger.info("Saving to file")
        self.signal.df.to_parquet(f"{prefix}.signal.parquet")
        self.noise.df.to_parquet(f"{prefix}.noise.parquet")
        if self.onehot:
            np.save(f"{prefix}.features.npy", self.features)
            np.save(f"{prefix}.labels.npy", self.labels)


class SignalGenerator:
    def __init__(self, num: int, pads: Pads, smear: float):

        logger.info(f"Creating lines and traversing pads with smear={smear}")
        lines = Lines(num, smear=smear)
        traverser = Traverser(lines, pads)

        if len(lines.df) != len(traverser.df):
            raise ValueError(f"Lines and Traverser have different lengths: {len(lines.df)} != {len(traverser.df)}")

        logger.info("Dropping per-layer info")
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

