import numpy as np
import time
from pathlib import Path
from pads_ml.generator import SignalGenerator, NoiseGenerator
from pads_ml.preprocess import DataPreparer
from pads_ml.pads import Pads

import logging
logging.basicConfig(level=logging.INFO)

def main():
    pads_path = Path("data/STGCPadTrigger.np.A05.txt")
    pads = Pads(pads_path)
    now = time.strftime("%Y_%m_%d_%H_%M_%S")
    num = 10_000

    logging.info("Generating signal")
    signal = SignalGenerator(num, pads)
    logging.info(signal.df)

    logging.info("Generating noise")
    noise = NoiseGenerator(num, pads)
    logging.info(noise.df)

    logging.info("Writing to file")
    signal.df.to_parquet(f"signal.{now}.{num}.parquet")
    noise.df.to_parquet(f"noise.{now}.{num}.parquet")

    logging.info("Converting to one-hot np.array")
    #dp = DataPreparer(signal.df, noise.df)
    #dp.prepare()
    #np.save(f"features.{now}.{num}.npy", dp.features)
    #np.save(f"labels.{now}.{num}.npy", dp.labels)

if __name__ == "__main__":
    main()


