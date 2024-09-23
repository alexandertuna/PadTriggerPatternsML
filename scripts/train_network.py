import numpy as np
import pandas as pd
from pathlib import Path

from pads_ml.train import OneHotFullyConnectedTrainer

import logging
logging.basicConfig(level=logging.INFO)

def main():
    # signal_name = Path("signal.2024_09_22_12_53_38.100000.parquet")
    # noise_name = Path("noise.2024_09_22_12_53_38.100000.parquet")
    # signal_name = Path("signal.2024_09_22_17_44_41.1000000.parquet")
    # noise_name = Path("noise.2024_09_22_17_44_41.1000000.parquet")
    # signal_name = Path("signal.2024_09_22_17_43_09.100000.one_hot.npy")
    # noise_name = Path("noise.2024_09_22_17_43_09.100000.one_hot.npy")
    features_name = Path("combined.2024_09_22_17_43_09.100000.one_hot.features.npy")
    labels_name = Path("combined.2024_09_22_17_43_09.100000.one_hot.labels.npy")

    logging.info(f"Opening {features_name} ...")
    features = np.load(features_name)

    logging.info(f"Opening {labels_name} ...")
    labels = np.load(labels_name)

    logging.info(f"Creating model ...")
    trainer = OneHotFullyConnectedTrainer(features, labels)

    logging.info(f"Training model ...")
    trainer.train()

if __name__ == "__main__":
    main()
