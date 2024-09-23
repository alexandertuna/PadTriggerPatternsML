import numpy as np
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)

MINIMUM = 8

def main():
    signal_fname = Path("signal.2024_09_23_08_12_35.100000.one_hot.npy")
    noise_fname = Path("noise.2024_09_23_08_12_35.100000.one_hot.npy")

    logging.info(f"Opening {signal_fname}")
    logging.info(f"Opening {noise_fname}")
    signal = np.load(signal_fname)
    noise = np.load(noise_fname)

    logging.info(f"Skipping signal with too few pads")
    signal = signal[ (signal.sum(axis=1) >= MINIMUM) ]
    noise = noise[ : len(signal) ]
    features = np.concatenate([signal, noise], axis=0)

    logging.info(f"Create labels")
    labels = np.zeros(shape=(len(features), 1))
    labels[len(signal) : , :] = 1

    logging.info(f"Shuffling")
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    features = features[indices]
    labels = labels[indices]

    features_fname = Path("combined.2024_09_23_08_12_35.100000.one_hot.features.npy")
    labels_fname = Path("combined.2024_09_23_08_12_35.100000.one_hot.labels.npy")
    logging.info(f"Writing {features.shape} to {features_fname}")
    logging.info(f"Writing {labels.shape} to {labels_fname}")
    np.save(features_fname, features)
    np.save(labels_fname, labels)

if __name__ == "__main__":
    main()
