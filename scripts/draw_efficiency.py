import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pads_ml import constants
from pads_ml.utils import onehotify

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input signal parquet file", default="train/gen.2024_10_02_11_01_40.100000.signal.parquet")
    parser.add_argument("-r", "--ranking", help="Feature ranking npy file", default="features.ranked.npy")
    parser.add_argument("--pdf", help="Output pdf file for plots", default="efficiency.pdf")
    return parser.parse_args()


def main():

    # CL args
    ops = options()

    # get parquet
    input = pd.read_parquet(Path(ops.input))
    print(input)

    # get ranking
    ranking = np.load(Path(ops.ranking)) # [: 10]
    print("ranking", ranking)

    # onehotify parquet
    onehot = onehotify(input)
    print("onehot", onehot)
    print(onehot.sum(axis=1))

    # get indices of parquet with enough pads
    denom_mask = onehot.sum(axis=1) >= constants.PADS_REQUIRED
    print("denom_mask", denom_mask)

    # get indices of onehotified in ranking
    # this is currently a slow implementation
    numer_mask = np.zeros(len(onehot), dtype=bool)
    for i, row in enumerate(onehot):
        if i % 10000 == 0:
            print(i)
        for rank in ranking:
            if (row == rank).all():
                numer_mask[i] = True
                break
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
    print("denom_unbinned", denom_unbinned)
    print("numer_unbinned", numer_unbinned)

    # make histogramdd of signal, denom and numer
    denom_numpy = denom_unbinned.to_numpy()
    numer_numpy = numer_unbinned.to_numpy()
    print("denom_numpy", denom_numpy.shape)
    print("numer_numpy", numer_numpy.shape)
    xbins = np.linspace(-1100, 1100, 200)
    ybins = np.linspace(800, 4800, 200)
    denom_hist, (xedges, yedges) = np.histogramdd(denom_numpy, bins=(xbins, ybins))
    numer_hist, (xedges, yedges) = np.histogramdd(numer_numpy, bins=(xbins, ybins))
    print(xedges)
    print(yedges)

    # divide
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = np.true_divide(numer_hist, denom_hist)
        efficiency[~np.isfinite(efficiency)] = 0
    print("efficiency", efficiency)

    # plot
    xmesh, ymesh = np.meshgrid(xedges, yedges)
    plt.pcolormesh(xmesh, ymesh, efficiency.T, cmap='gist_heat_r')
    plt.colorbar(label='Efficiency')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Efficiency')
    plt.savefig(ops.pdf)

    pass


if __name__ == "__main__":
    main()
