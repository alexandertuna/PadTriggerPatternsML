import numpy as np
import pandas as pd

from pads_ml import constants
from typing import Tuple

import logging
logger = logging.getLogger(__name__)

class DataPreparer:

    def __init__(self, signal: pd.DataFrame, noise: pd.DataFrame) -> None:

        self.signal = signal
        self.noise = noise

    def prepare(self) -> None:

        # convert to one-hot
        def onehotify(df: pd.DataFrame) -> np.array:

            onehot = pd.DataFrame({
                f"pad_{layer}": pd.Categorical(
                    df[f"pad_{layer}"],
                    categories=constants.PADS_PER_LAYER[layer]
                )
                for layer in range(constants.LAYERS)
            })

            return pd.get_dummies(onehot).values

        signal, noise = onehotify(self.signal), onehotify(self.noise)
        logger.info(f"Signal shape, noise shape: {signal.shape}, {noise.shape}")

        # remove rows with too few pads
        signal = signal[ (signal.sum(axis=1) >= constants.PADS_REQUIRED) ]
        noise = noise[ (noise.sum(axis=1) >= constants.PADS_REQUIRED) ]
        smaller = min(len(signal), len(noise))
        signal, noise = signal[ : smaller], noise[ : smaller]
        logger.info(f"Signal shape, noise shape: {signal.shape}, {noise.shape}")

        # combine, make labels
        self.features = np.concatenate([signal, noise], axis=0)
        self.labels = np.zeros(shape=(len(self.features), 1))
        self.labels[:len(signal), :] = 1

        # shuffle
        indices = np.arange(len(self.features))
        np.random.shuffle(indices)
        self.features = self.features[indices]
        self.labels = self.labels[indices]

