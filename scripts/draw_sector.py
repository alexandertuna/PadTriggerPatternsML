"""
Given a file of pad positions, draw each layer of pads.
"""

from pads_ml.pads import Pads
from pads_ml import constants

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import logging
logging.basicConfig(level=logging.INFO)


def main() -> None:

    name_i = "data/STGCPadTrigger.np.A05.txt"
    name_o = "detector.pdf"

    logging.info(f"Opening pads file: {name_i}")
    pads = Pads(name_i)

    logging.info(f"Drawing pads to: {name_o}")
    with PdfPages(name_o) as pdf:
        for layer in range(constants.LAYERS):
            logging.info(f"Drawing layer: {layer}")
            draw_layer(pads.layer[layer], pdf)


def draw_layer(pads: pd.DataFrame, pdf: PdfPages) -> None:
    # draw pads (rows)
    fig, ax = plt.subplots(figsize=(4, 4))
    for _, row in pads.iterrows():
        poly = row["geometry"]
        x, y = poly.exterior.xy
        ax.plot(x, y, color="black", linewidth=0.5)

    # make pretty
    ax.set_xlim(*get_lim("x"))
    ax.set_ylim(*get_lim("y"))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.12)
    pdf.savefig(fig)
    plt.close()


def get_lim(axis: str, buffer: float=0.05):
    if axis not in ["x", "y"]:
        raise ValueError(f"Invalid axis: {axis}")
    min, max = (constants.SECTOR_XMIN, constants.SECTOR_XMAX) if axis == "x" else \
               (constants.SECTOR_YMIN, constants.SECTOR_YMAX)
    buffer = buffer * (max - min)
    return min - buffer, max + buffer

if __name__ == "__main__":
    main()
