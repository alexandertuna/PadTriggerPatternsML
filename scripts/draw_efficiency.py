import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pads_ml import constants
from pads_ml.utils import onehotify, undo_onehot

import logging
logging.basicConfig(level=logging.INFO)

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input signal parquet file", default="train/gen.2024_10_03_15_47_37.100000.signal.parquet")
    # parser.add_argument("-r", "--ranking", help="Feature ranking npy file", default="features.ranked.npy")
    parser.add_argument("-r", "--ranking", help="Feature ranking npy file", default="data/LS_pack_new_phis.npy")
    parser.add_argument("--pdf", help="Output pdf file for plots", default="efficiency.pdf")
    return parser.parse_args()


def main():

    # CL args
    ops = options()

    # get parquet
    input = pd.read_parquet(Path(ops.input))
    input_arr = input[ [f"pad_{i}" for i in range(8)] ].to_numpy()
    input_onehot = onehotify(input)

    CHECKS = range(40, 45)

    print(input_arr)
    for row in CHECKS:
        print("signal", row, input_arr[row])

    # get ranking
    ranking = np.load(Path(ops.ranking)) # [: 10]
    if ranking.shape[1] == constants.PADS:
        ranking = undo_onehot(ranking)
    # remove rows with -1
    ranking = ranking[~(ranking == -1).any(axis=1)]
    # print("ranking", ranking[:10])

    # onehotify parquet
    # onehot = onehotify(input)
    # print("onehot", onehot)
    # print(onehot.sum(axis=1))

    # get indices of parquet with enough pads
    denom_mask = input_onehot.sum(axis=1) >= constants.PADS_REQUIRED
    print("denom_mask", denom_mask)

    # get indices of onehotified in ranking
    # this is currently a slow implementation
    logging.info("Checking if each signal pattern exists in the ranking")
    numer_mask = np.zeros(len(input), dtype=bool)

    # apparently this is a vectorized way to check if row inside 2d array
    ## numer_mask = (ranking[:, np.newaxis] == input_arr).all(axis=2).any(axis=0)

    ## # check 6/8 instead of 8/8
    ## # Vectorized approach to find if a row in features is "good" according to the criterion
    ## # Create a boolean mask for whether each element of each row in features matches each row in good_examples
    ## matches = (input_arr[:, np.newaxis, :] == ranking[np.newaxis, :, :])
    ## # Sum along the last axis to get the number of matches for each feature-good_example pair
    ## matches_sum = matches.sum(axis=2)
    ## # Check if there are at least 6 matches for any row in good_examples
    ## numer_mask = (matches_sum >= 6).any(axis=1)

    # check for 3/4 & 3/4
    # Create boolean masks for matching the first 4 elements and the last 4 elements
    matches_first_half = (input_arr[:, None, :4] == ranking[None, :, :4])
    matches_second_half = (input_arr[:, None, 4:] == ranking[None, :, 4:])

    # Sum along the last axis to get the number of matches for each half
    matches_sum_first_half = matches_first_half.sum(axis=2)
    matches_sum_second_half = matches_second_half.sum(axis=2)

    # Check if there are at least 3 matches in both halves for any row in ranking
    numer_mask = ((matches_sum_first_half >= 3) & (matches_sum_second_half >= 3)).any(axis=1)
    numer_mask &= denom_mask


    # for i, row in enumerate(input_arr):
    #     if i % 10000 == 0:
    #         print(i)
    #     for rank in ranking:
    #         if (row == rank).all():
    #             numer_mask[i] = True
    #             break

    # print("numer_mask_00_10", list(numer_mask[0:10]))
    # print("numer_mask_10_20", list(numer_mask[10:20]))
    # print("numer_mask_20_30", list(numer_mask[20:30]))
    # print("numer_mask_30_40", list(numer_mask[30:40]))
    print("numer_mask_40_50", list(numer_mask[40:50]))
    print("numer_mask", numer_mask)

    # check
    print("denom_mask.sum()", denom_mask.sum())
    print("numer_mask.sum()", numer_mask.sum())

    # make x, y from theta, phi, zmid
    input["r"] = constants.ZMID * np.tan(input["theta"])
    input["x"] = input["r"] * np.cos(input["phi"])
    input["y"] = input["r"] * np.sin(input["phi"])

    # get denom and numer
    cols = ["x", "y"]
    denom_unbinned = input[cols][denom_mask]
    numer_unbinned = input[cols][numer_mask]
    # print("denom_unbinned", denom_unbinned)
    # print("numer_unbinned", numer_unbinned)

    # make histogramdd of signal, denom and numer
    denom_numpy = denom_unbinned.to_numpy()
    numer_numpy = numer_unbinned.to_numpy()
    print("denom_numpy", denom_numpy.shape)
    print("numer_numpy", numer_numpy.shape)
    xbins = np.linspace(-1100, 1100, 100)
    ybins = np.linspace(800, 4800, 200)
    denom_hist, (xedges, yedges) = np.histogramdd(denom_numpy, bins=(xbins, ybins))
    numer_hist, (xedges, yedges) = np.histogramdd(numer_numpy, bins=(xbins, ybins))
    # print(xedges)
    # print(yedges)

    # divide
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = np.true_divide(numer_hist, denom_hist)
        efficiency[~np.isfinite(efficiency)] = 0
    print("efficiency", efficiency)

    # plot
    xmesh, ymesh = np.meshgrid(xedges, yedges)
    # plt.pcolormesh(xmesh, ymesh, efficiency.T, cmap='gist_heat_r')
    plt.pcolormesh(xmesh, ymesh, efficiency.T, vmin=0.1, cmap='hot_r')
    plt.colorbar(label='Efficiency')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Efficiency')
    plt.savefig(ops.pdf)

    pass


if __name__ == "__main__":
    main()
