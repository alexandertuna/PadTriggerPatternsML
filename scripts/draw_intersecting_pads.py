"""
From a file of generated lines and its intersecting pads,
draw the intersecting pads.
"""

import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from typing import Tuple, List
from shapely.geometry import Polygon

from pads_ml.pads import Pads
from pads_ml import constants

import logging
logging.basicConfig(level=logging.INFO)

from cycler import cycler
mpl.rcParams["axes.prop_cycle"] = cycler('color', [
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
])

# Enum
MAX = 0
MIN = 1


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", help="Input parquet file of pad patterns", required=True)
    parser.add_argument("--pads", help="Input file of pads geometry", required=True)
    parser.add_argument("-n", "--num", help="Number of patterns to draw", default=20, type=int)
    parser.add_argument("-o", "--output", help="Output pdf file to draw", default="intersecting_pads.pdf")
    return parser.parse_args()


def main():

    # CL args
    ops = options()
    num_lines = ops.num
    logging.info(f"Getting patterns from {ops.parquet}")
    logging.info(f"Getting pads from {ops.pads}")
    gen = pd.read_parquet(ops.parquet)
    pads = Pads(ops.pads)

    # Draw
    logging.info(f"Drawing {num_lines} patterns to {ops.output}")
    with PdfPages(ops.output) as pdf:
        for line in tqdm(range(num_lines)):
            row = gen.iloc[line]
            pad_polygons = []
            for layer in range(constants.LAYERS):
                i_pad = row[f"pad_{layer}"].astype(int)
                if i_pad == -1:
                    pad_polygons.append(None)
                else:
                    pad_polygons.append(pads.df.iloc[i_pad]["geometry"])
            draw(pad_polygons, pdf)


def draw(polygons: List[Polygon], pdf: PdfPages):
    """
    Draw a list of polygons to pdf
    Draw the polygon with transparent fill
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    for ip, poly in enumerate(polygons):
        if poly is None:
            continue
        x, y = poly.exterior.xy
        ax.plot(x, y, label=f"L{ip}")
        ax.fill(x, y, alpha=0.1)
    ax.legend(frameon=False, fontsize=6)
    min_x, min_y = feature_xy(polygons, MIN)
    max_x, max_y = feature_xy(polygons, MAX)
    buffer_x = 0.12 * (max_x - min_x)
    buffer_y = 0.12 * (max_y - min_y)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_xlim(min_x - buffer_x, max_x + buffer_x)
    ax.set_ylim(min_y - buffer_y, max_y + buffer_y)
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.12)
    pdf.savefig()
    plt.close()

def feature_xy(
        polygons: List[Polygon],
        feature: int,
    ) -> Tuple[float]:
    """
    Return a feature of a list of polygons
    """
    if feature not in [MAX, MIN]:
        raise ValueError(f"feature {feature} not recognized")
    if feature == MIN:
        x = min([poly.bounds[0] for poly in polygons if poly is not None])
        y = min([poly.bounds[1] for poly in polygons if poly is not None])
    elif feature == MAX:
        x = max([poly.bounds[2] for poly in polygons if poly is not None])
        y = max([poly.bounds[3] for poly in polygons if poly is not None])
    return x, y


if __name__ == "__main__":
    main()

