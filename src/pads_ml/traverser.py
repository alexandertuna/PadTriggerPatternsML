import pandas as pd
from shapely.geometry import Point

from . import constants

class Traverser:

    def __init__(
        self,
        lines: pd.DataFrame,
        pads: pd.DataFrame,
    ):
        self.lines = lines
        self.pads = pads

        # Find the pad which passes through the line
        self.df = pd.DataFrame()

        layer = 0
        intersecting_pads = []
        for _, line in self.lines.iterrows():
            intersection = False
            point = Point(line[f"x_at_layer_{layer}"], line[f"y_at_layer_{layer}"])
            for _, pad in self.pads.iterrows():
                if pad["layer"] != layer:
                    continue
                if pad["geometry"].contains(point):
                    print(pad["geometry"])
                    intersecting_pads.append(pad["geometry"])
                    intersection = True
                    break
            if not intersection:
                intersecting_pads.append(None)
        self.df["intersecting_pad"] = intersecting_pads

        # for layer in range(constants.LAYERS):
        #     self.df[f"pad_at_layer_{layer}"] = self.pads[self.pads["layer"] == layer].apply(self._find_pad, axis=1)

