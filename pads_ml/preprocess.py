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

        # remove rows with too few pads
        signal = signal[ (signal.sum(axis=1) >= constants.PADS_REQUIRED) ]
        noise = noise[ : len(signal) ]

        # combine, make labels
        self.features = np.concatenate([signal, noise], axis=0)
        self.labels = np.zeros(shape=(len(self.features), 1))
        self.labels[:len(signal), :] = 1

        # shuffle
        indices = np.arange(len(self.features))
        np.random.shuffle(indices)
        self.features = self.features[indices]
        self.labels = self.labels[indices]



    def prepare2(self) -> None:

        raise Exception("This is more robust, but slower")
        signal = self.remove_negative_ones(self.signal)
        noise = self.remove_negative_ones(self.noise)

        signal_one_hot = self.convert_to_one_hot(signal)
        noise_one_hot = self.convert_to_one_hot(noise)

        self.features, self.labels = self.combine_and_shuffle(signal_one_hot, noise_one_hot)


    def remove_negative_ones(self, df: pd.DataFrame) -> None:

        raise Exception("This is more robust, but slower")
        df[ df == -1 ] = np.nan
        return df


    def convert_to_one_hot(self, df: pd.DataFrame) -> np.array:

        raise Exception("This is more robust, but slower")
        # Get the pad columns as np.array, and mask nan
        pad_columns = df[ [f"pad_{i}" for i in range(constants.LAYERS)] ].values
        valid_entries_mask = ~np.isnan(pad_columns)

        # Create a row index array
        row_indices = np.arange(len(df)) # Shape: len(df)
        row_indices = row_indices[:, np.newaxis] # Shape: (len(df), 1)
        row_indices = np.broadcast_to(row_indices, pad_columns.shape) # Broadcast to (len(df), 8)

        # Specify the valid entries
        valid_row_indices = row_indices[valid_entries_mask]
        valid_pad_columns = pad_columns[valid_entries_mask].astype(int)

        # Set the one-hot values (intensive)
        logger.info(f"Doing the heavy lifting ...")
        one_hot_array = np.zeros((len(df), constants.PADS), dtype=int)
        one_hot_array[valid_row_indices, valid_pad_columns] = 1

        return one_hot_array


    def combine_and_shuffle(self, signal: np.array, noise: np.array) -> Tuple[np.array, np.array]:

        raise Exception("This is more robust, but slower")
        signal = signal[ (signal.sum(axis=1) >= constants.PADS_REQUIRED) ]
        noise = noise[ : len(signal) ]
        features = np.concatenate([signal, noise], axis=0)

        labels = np.zeros(shape=(len(features), 1))
        labels[:len(signal), :] = 1

        indices = np.arange(len(features))
        np.random.shuffle(indices)
        features = features[indices]
        labels = labels[indices]

        logger.info(f"Shape: {features.shape}")
        return features, labels


