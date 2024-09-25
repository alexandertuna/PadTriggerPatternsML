import numpy as np
import pandas as pd
from pathlib import Path

from pads_ml import constants

def main():

    # tmp checking 
    signal_path = Path("signal.2024_09_24_16_33_51.1000000.parquet")
    cols = [f"pad_{i}" for i in range(constants.LAYERS)]
    df = pd.read_parquet(signal_path)
    df = df[cols]
    print(df)
    nodups = df.drop_duplicates()
    print(nodups.shape)
    #print(nodups)
    allvalid = nodups[ nodups.apply(lambda row: all(row != -1), axis=1) ]
    print(allvalid.shape)
    # print(allvalid)

    return

    # paths
    features_path = Path("features.2024_09_24_12_58_48.1000000.npy")
    labels_path = Path("labels.2024_09_24_12_58_48.1000000.npy")

    # load data
    features = np.load(features_path)
    labels = np.load(labels_path)

    # separate signal and noise
    is_signal = labels[:, 0] == 1
    signal = features[is_signal]
    noise = features[~is_signal]
    print(signal.shape)
    print(signal)
    print(noise.shape)
    print(noise)

    # count unique rows
    unique_signal = np.unique(signal, axis=0)
    unique_noise = np.unique(noise, axis=0)
    print(f"Unique signal: {unique_signal.shape[0]}")
    print(f"Unique noise: {unique_noise.shape[0]}")

if __name__ == "__main__":
    main()
