import argparse
import time
from pads_ml import constants
from pads_ml.generator import EverythingGenerator
from pads_ml.pads import Pads

import logging
logging.basicConfig(level=logging.INFO)


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pads", help="Input file of pads geometry", default=constants.PADS_PATH)
    parser.add_argument("-s", "--smear", help="Amount of gaussian smearing [mm] when projecting to each layer", default=0.0, type=float)
    parser.add_argument("-n", "--num", help="Number of line to simulate", default=10_000, type=int)
    parser.add_argument("-b", "--background", help="Type of background simulation", default="smear", choices=["smear", "random"])
    parser.add_argument("--onehot", help="Convert to one-hot numpy format, too", action="store_true", default=False)
    return parser.parse_args()


def main():

    # CL args
    ops = options()
    pads = Pads(ops.pads)
    now = time.strftime("%Y_%m_%d_%H_%M_%S")
    num = ops.num

    logging.info("Generating everything")
    gen = EverythingGenerator(num, pads, ops.smear, ops.onehot, ops.background)
    gen.save(f"gen.{now}.{num}")


if __name__ == "__main__":
    main()


