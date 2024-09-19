import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from . import constants

class Pads:
    """
    Pads holds a pandas dataframe of pad geometry
    """
    def __init__(self,
                 filename: str,
                 create_polygons: bool = True,
                 ) -> None:

        # read file
        self.filename = filename
        self.df = pd.read_csv(self.filename, sep="\s+")

        # convert hex to int
        hexify = lambda x: int(x, 16)
        self.df["wheel"] = self.df["wheel_hex"].apply(hexify)
        self.df["id"] = self.df["id_hex"].apply(hexify)

        # convert pfeb to layer, quad
        self.df["layer"] = self.df["pfeb"] % constants.LAYERS
        self.df["quad"] = self.df["pfeb"] // constants.LAYERS

        # if requested: polygons
        if create_polygons:
            def create_polygon(row):
                points = [
                    (row["x0"], row["y0"]),
                    (row["x1"], row["y1"]),
                    (row["x3"], row["y3"]),
                    (row["x2"], row["y2"]),
                ]
                return Polygon(points)

            self.df["geometry"] = self.df.apply(create_polygon, axis=1)
            self.gdf = gpd.GeoDataFrame(self.df, geometry="geometry")
