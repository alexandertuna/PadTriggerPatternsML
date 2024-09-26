import argparse
import numpy as np
import time
from pathlib import Path
from pads_ml.generator import SignalGenerator, NoiseGenerator
from pads_ml.preprocess import DataPreparer
from pads_ml.pads import Pads

import logging
logging.basicConfig(level=logging.INFO)


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pads", help="Input file of pads geometry", required=True)
    parser.add_argument("--smear", help="Amount of gaussian smearing [mm] when projecting to each layer", default=10.0, type=float)
    parser.add_argument("-n", "--num", help="Number of line to simulate", default=10_000, type=int)
    parser.add_argument("--onehot", help="Convert to one-hot numpy format, too", action="store_true", default=False)
    return parser.parse_args()


def main():

    # CL args
    ops = options()
    pads = Pads(ops.pads)
    now = time.strftime("%Y_%m_%d_%H_%M_%S")
    num = ops.num

    logging.info("Generating signal")
    signal = SignalGenerator(num, pads, ops.smear)
    logging.info(signal.df)

    logging.info("Generating noise")
    noise = NoiseGenerator(num, pads)
    logging.info(noise.df)

    logging.info("Writing to file")
    signal.df.to_parquet(f"signal.{now}.{num}.parquet")
    noise.df.to_parquet(f"noise.{now}.{num}.parquet")

    if ops.onehot:
        logging.info("Converting to one-hot np.array")
        dp = DataPreparer(signal.df, noise.df)
        dp.prepare()
        np.save(f"features.{now}.{num}.npy", dp.features)
        np.save(f"labels.{now}.{num}.npy", dp.labels)


if __name__ == "__main__":
    main()


