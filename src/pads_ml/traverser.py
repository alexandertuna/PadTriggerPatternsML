import pandas as pd
import geopandas as gpd

from . import constants
from .pads import Pads
from .lines import Lines

class Traverser:

    def __init__(
        self,
        lines: Lines,
        pads: Pads,
    ):

        # For each layer, find which pad (if any) contains the line
        self.df = pd.DataFrame()
        for layer in range(constants.LAYERS):
            lines_gdf = gpd.GeoDataFrame(lines.df, geometry=f"point_{layer}")
            pads_gdf = gpd.GeoDataFrame(pads.layer[layer], geometry="geometry")
            joined = lines_gdf.sjoin(pads_gdf, how="left", predicate="within")
            self.df[f"pad_{layer}"] = joined["index_right"].fillna(-1).astype(int)

