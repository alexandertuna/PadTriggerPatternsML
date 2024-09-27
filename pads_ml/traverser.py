import pandas as pd
import geopandas as gpd

from pads_ml import constants
from pads_ml.pads import Pads
from pads_ml.lines import Lines

import logging
logger = logging.getLogger(__name__)

class Traverser:

    def __init__(
        self,
        lines: Lines,
        pads: Pads,
    ):

        # For each layer, find which pad (if any) contains the line
        logger.info("Finding pads in each layer for the lines")
        self.df = pd.DataFrame()
        for layer in range(constants.LAYERS):
            lines_gdf = gpd.GeoDataFrame(lines.df, geometry=f"point_{layer}")
            pads_gdf = gpd.GeoDataFrame(pads.layer[layer], geometry="geometry")
            joined = lines_gdf.sjoin(pads_gdf, how="left", predicate="within")
            self.df[f"pad_{layer}"] = joined["index_right"].fillna(-1).astype(int)

