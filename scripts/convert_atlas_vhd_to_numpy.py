import argparse
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

from pads_ml import constants
from pads_ml.pads import Pads
PADS = Pads(constants.PADS_PATH)


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input text file", default="data/LS_pack_new_phis.txt")
    parser.add_argument("-o", "--output", help="Output npy file", default="data/LS_pack_new_phis.npy")
    parser.add_argument("-c", "--csv", help="Output csv file", default="data/LS_pack_new_phis.csv")
    return parser.parse_args()


def main():

    # CL args
    ops = options()

    # read text file
    logging.info(f"Reading {ops.input}")
    arr = np.loadtxt(ops.input, delimiter=" ", dtype=int)
    logging.info(f"Shape: {arr.shape}")
    logging.info(f"Array: {arr}")

    # get rid of phi id (the last column)
    arr = arr[:, :-1]
    logging.info(f"Shape: {arr.shape}")
    logging.info(f"Array: {arr}")

    # split up pfebs and pad channels
    pfebs = arr[:, range(0, 16, 2)]
    chans = arr[:, range(1, 16, 2)]
    logging.info(pfebs.shape)
    logging.info(chans.shape)

    # convert to indices
    new_arr = np.vectorize(pfeb_and_pad_channel_to_index)(pfebs, chans)
    new_arr = new_arr.astype(int)
    logging.info(new_arr.shape)
    np.save(ops.output, new_arr)
    np.savetxt(ops.csv, new_arr, delimiter=" ", fmt="%d")

def pfeb_and_pad_channel_to_index(pfeb: int, pad: int) -> int:
    idx = PADS.df[ (PADS.df["pfeb"] == pfeb) & (PADS.df["pad_channel"] == pad) ].index
    if len(idx) != 1:
        logging.warning(f"Found {len(idx)} matches for pfeb {pfeb} and pad {pad}")
        return -1
    return idx.item()


if __name__ == "__main__":
    main()
