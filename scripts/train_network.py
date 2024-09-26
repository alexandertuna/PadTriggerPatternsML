import argparse
from pathlib import Path
import time

from pads_ml.train import OneHotFullyConnectedTrainer

import logging
logging.basicConfig(level=logging.INFO)

NOW = time.strftime("%Y_%m_%d_%H_%M_%S")
print(NOW)

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Dir of training files of features and labels", default="train")
    parser.add_argument("--valid", help="Dir of validation filse of features and labels", default="valid")
    return parser.parse_args()


def main():

    # CL args
    ops = options()
    features_train_path = list(Path(ops.train).glob("*features*npy"))
    features_valid_path = list(Path(ops.valid).glob("*features*npy"))
    labels_train_path = list(Path(ops.train).glob("*labels*npy"))
    labels_valid_path = list(Path(ops.valid).glob("*labels*npy"))

    logging.info(f"Creating model")
    trainer = OneHotFullyConnectedTrainer(
        features_train_path,
        features_valid_path,
        labels_train_path,
        labels_valid_path,
    )

    logging.info(f"Training model")
    trainer.train()

    output = Path(f"model.{NOW}.pt")
    logging.info(f"Saving model to {output}")
    trainer.save(path=output)

if __name__ == "__main__":
    main()
