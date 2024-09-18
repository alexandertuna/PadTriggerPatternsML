import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

class Pads:
    """
    Pads holds a pandas dataframe of pad geometry
    """
    def __init__(self,
                 filename: str,
                 create_polygons: bool = True,
                 ) -> None:

        self.filename = filename
        self.df = pd.read_csv(self.filename, sep="\s+")

        hexify = lambda x: int(x, 16)
        self.df["wheel"] = self.df["wheel_hex"].apply(hexify)
        self.df["id"] = self.df["id_hex"].apply(hexify)

        if create_polygons:
            def create_polygon(row):
                points = [
                    (row["x0"], row["y0"]),
                    (row["x1"], row["y1"]),
                    (row["x2"], row["y2"]),
                    (row["x3"], row["y3"]),
                ]
                return Polygon(points)

            self.df["geometry"] = self.df.apply(create_polygon, axis=1)
            self.gdf = gpd.GeoDataFrame(self.df, geometry="geometry")
