import argparse
from pads_ml.pads import Pads
import logging

logging.basicConfig(level=logging.INFO)


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Input file, whole wheel", required=True)
    parser.add_argument("-o", help="Output file, A05 only", required=True)
    return parser.parse_args()


def main():

    # CL args
    ops = options()

    # get input pads
    logging.info(f"Reading pads from {ops.i}")
    pads = Pads(ops.i, create_polygons=False)

    # require wheel A, sector 5 (numbered from 1)
    logging.info("Selecting A05")
    df_A05 = pads.df[
        (pads.df["wheel"] == 0xA)
        & (pads.df["sector"] == 4)
    ]

    # write to disk
    logging.info(f"Writing pads to {ops.o}")
    df_A05.to_csv(ops.o, sep=' ', index=False)


if __name__ == "__main__":
    main()


