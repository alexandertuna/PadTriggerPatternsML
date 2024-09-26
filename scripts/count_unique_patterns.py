import argparse
import numpy as np
import pandas as pd

from pads_ml import constants

import logging
logging.basicConfig(level=logging.INFO)


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Input dataframe (parquet) for checking pad patterns", required=True)
    return parser.parse_args()


def main():

    # CL args
    ops = options()

    # Get input dataframe
    logging.info(f"Reading data from {ops.i}")
    df = pd.read_parquet(ops.i)

    # Only consider pads from this dataframe
    cols = [f"pad_{i}" for i in range(constants.LAYERS)]
    df = df[cols]
    logging.info(df)
    logging.info(f"Input dataframe shape: {df.shape}")

    # Drop duplicates
    nodups = df.drop_duplicates()
    logging.info(f"Drop duplicates shape: {nodups.shape}")

    # Only consider valid pads
    allvalid = nodups[ nodups.apply(lambda row: all(row != -1) & all(row != np.nan), axis=1) ]
    logging.info(f"Only valid pads shape: {allvalid.shape}")


if __name__ == "__main__":
    main()
