"""
Given a parquet file of pad patterns, write to a ROOT file
"""

import argparse
import pandas as pd

from pads_ml import constants

import ROOT

import logging

logging.basicConfig(level=logging.INFO)


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tree", help="Tree name for output", default="tree")
    parser.add_argument("-i", "--input", help="Input parquet file", required=True)
    parser.add_argument("-o", "--output", help="Output ROOT file", default="out.root")
    return parser.parse_args()


def main() -> None:

    # CL args
    ops = options()

    # Open input file
    logging.info(f"Reading parquet file: {ops.input}")
    df = pd.read_parquet(ops.input)
    print(df)

    # Write to ROOT file
    logging.info(f"Writing to ROOT file: {ops.output} / {ops.tree}")
    to_root(df, ops.output, ops.tree)


def to_root(df: pd.DataFrame, filename: str, treename: str) -> None:
    """
    Write a DataFrame to a ROOT file
    """
    keys = [f"pad_{i}" for i in range(constants.LAYERS)]
    data = {key: df[key].values for key in keys}
    rdf = ROOT.RDF.FromNumpy(data)
    rdf.Snapshot(treename, filename)


if __name__ == "__main__":
    main()
