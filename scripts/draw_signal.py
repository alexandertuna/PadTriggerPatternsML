"""
Given a file of pad positions, draw each layer of pads.
"""

from pads_ml.pads import Pads
from pads_ml import constants

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from shapely.ops import unary_union

import logging
logging.basicConfig(level=logging.INFO)

LAYER = 0

def main() -> None:

    name_pads = Path("data/STGCPadTrigger.np.A05.txt")
    # name_signal = Path("signal.2024_09_24_12_58_48.1000000.parquet")
    name_signal = Path("signal.2024_09_24_18_19_14.1000000.parquet")
    name_pdf = Path("signal.pdf")

    logging.info(f"Opening pads file: {name_pads}")
    pads = Pads(name_pads)
    layer = pads.layer[LAYER]

    logging.info(f"Opening signal file: {name_signal}")
    signal = pd.read_parquet(name_signal)

    logging.info(f"Drawing signal and pads to: {name_pdf}")
    with PdfPages(name_pdf) as pdf:
        draw_pads_and_signal(layer, signal, pdf)


def draw_pads_and_signal(layer: pd.DataFrame, signal: pd.DataFrame, pdf: PdfPages) -> None:

    fig, ax = plt.subplots(figsize=(4, 4))

    # draw signal histogram
    z = constants.ZS[LAYER]
    x = z * np.tan(signal["theta"]) * np.cos(signal["phi"])
    y = z * np.tan(signal["theta"]) * np.sin(signal["phi"])
    xbins = np.linspace(*get_lim("x"), 100)
    ybins = np.linspace(*get_lim("y"), 100)
    _, _, _, im = ax.hist2d(x, y, bins=[xbins, ybins], cmin=0.5, cmap="autumn_r")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Simulated muons")

    # draw quads (unions of pads)
    union = unary_union(layer["geometry"])
    for poly in union.geoms:
        x, y = poly.exterior.xy
        ax.plot(x, y, color="black", linewidth=0.5)

    # make pretty
    ax.set_xlim(*get_lim("x"))
    ax.set_ylim(*get_lim("y"))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    plt.subplots_adjust(left=0.18, right=0.90, top=0.95, bottom=0.12)
    pdf.savefig(fig)
    plt.close()


def get_lim(axis: str, buffer: float=0.10):
    if axis not in ["x", "y"]:
        raise ValueError(f"Invalid axis: {axis}")
    min, max = (constants.SECTOR_XMIN, constants.SECTOR_XMAX) if axis == "x" else \
               (constants.SECTOR_YMIN, constants.SECTOR_YMAX)
    buffer = buffer * (max - min)
    return min - buffer, max + buffer

if __name__ == "__main__":
    main()
