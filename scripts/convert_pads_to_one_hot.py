import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)

from pads_ml import constants

def main():

    fname = Path("signal.2024_09_22_17_43_09.100000.parquet")
    # fname = Path("signal.2024_09_22_17_44_41.1000000.parquet")

    logging.info(f"Opening {fname} ...")
    df = pd.read_parquet(fname)
    df[ df == -1 ] = np.nan

    logging.info(f"Converting to one-hot ...")
    one_hot = convert_to_one_hot(df)

    # write to file
    outname = fname.with_suffix(".one_hot.npy")
    logging.info(f"Writing to: {outname}")
    np.save(outname, one_hot)


def convert_to_one_hot(df: pd.DataFrame) -> Tuple[np.array, np.array]:

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
    logging.info(f"Doing the heavy lifting ...")
    one_hot_array = np.zeros((len(df), constants.PADS), dtype=int)
    one_hot_array[valid_row_indices, valid_pad_columns] = 1

    # Check that the sum of each row is 8
    print(np.sum(one_hot_array, axis=1))

    #for row in range(len(df)):
    #    for col in range(constants.LAYERS):
    #        one_hot_array[row_indices[row], pad_columns[row, col]] = 1

    return one_hot_array


if __name__ == "__main__":
    main()
