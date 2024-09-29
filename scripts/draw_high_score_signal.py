"""
Given a model, features, and labels, 
make plots of the features with the highest scores
"""

import argparse
from pathlib import Path

from pads_ml.highest_score import DrawHighestScore
from pads_ml import constants

import logging
logging.basicConfig(level=logging.INFO)


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--features", help="Input numpy file of features", required=True)
    parser.add_argument("-l", "--labels", help="Input numpy file of labels", required=True)
    parser.add_argument("-m", "--model", help="Input torch file of model", required=True)
    parser.add_argument("-p", "--pads", help="Input file of pads geometry", default=constants.PADS_PATH)
    parser.add_argument("-n", "--num", help="Number of patterns to draw", default=10, type=int)
    parser.add_argument("-o", "--output", help="Output pdf file to draw", default="intersecting_pads.pdf")
    return parser.parse_args()


def main():

    # CL args
    ops = options()
    num = ops.num
    pads_path = Path(ops.pads)
    features_paths = sorted(list(Path().glob(ops.features)))
    labels_paths = sorted(list(Path().glob(ops.labels)))
    model_path = Path(ops.model)
    pdf_path = Path(ops.output)

    # go
    drawer = DrawHighestScore(num, pads_path, features_paths, labels_paths, model_path, pdf_path)
    drawer.draw()


if __name__ == "__main__":
    main()

