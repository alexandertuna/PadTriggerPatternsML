import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import Any

from . import constants

class Traverser:

    def __init__(
        self,
        lines: pd.DataFrame,
        pads: Any,
    ):
        self.lines = lines
        self.pads = pads

        # For each layer, find which pad (if any) contains the line
        self.df = pd.DataFrame()
        for layer in range(constants.LAYERS):
            lines_gdf = gpd.GeoDataFrame(self.lines, geometry=f"point_{layer}")
            pads_gdf = gpd.GeoDataFrame(self.pads.layer[layer], geometry="geometry")
            joined = lines_gdf.sjoin(pads_gdf, how="left", predicate="within")
            self.df[f"intersecting_pad_{layer}"] = joined["index_right"]

        print(self.df)

        # for layer in range(constants.LAYERS):
        #     self.df[f"pad_{layer}"] = self.pads[self.pads["layer"] == layer].apply(self._find_pad, axis=1)

