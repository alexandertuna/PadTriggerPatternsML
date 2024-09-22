"""
Given a file of pad positions, draw each layer of pads.
"""

from pads_ml.pads import Pads
from pads_ml import constants

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def main() -> None:

    name = "data/STGCPadTrigger.np.A05.txt"
    pads = Pads(name)

    with PdfPages("detector.pdf") as pdf:
        for layer in range(constants.LAYERS):
            draw(pads.layer[layer], pdf)


def draw(pads: pd.DataFrame, pdf: PdfPages) -> None:
    # draw pads (rows)
    fig, ax = plt.subplots(figsize=(4, 4))
    for _, row in pads.iterrows():
        poly = row["geometry"]
        x, y = poly.exterior.xy
        ax.plot(x, y, color="black", linewidth=0.5)

    # make pretty
    min_x, min_y = constants.SECTOR_XMIN, constants.SECTOR_YMIN
    max_x, max_y = constants.SECTOR_XMAX, constants.SECTOR_YMAX
    buffer_x = 0.05 * (max_x - min_x)
    buffer_y = 0.05 * (max_y - min_y)
    ax.set_xlim(min_x - buffer_x, max_x + buffer_x)
    ax.set_ylim(min_y - buffer_y, max_y + buffer_y)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.12)
    pdf.savefig(fig)
    plt.close()

if __name__ == "__main__":
    main()
