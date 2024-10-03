from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shapely
from shapely.geometry import Polygon

import logging
logger = logging.getLogger(__name__)

from pads_ml import constants
from pads_ml.pads import Pads
from pads_ml.inference import Inference

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

class DrawHighestScore:

    def __init__(
        self,
        num: int,
        pads_path: Path,
        features_paths: List[Path],
        labels_paths: List[Path],
        model_path: Path,
        pdf_path: Path,
    ) -> None:

        self.num = num
        self.pads = Pads(pads_path)
        self.features_paths = features_paths
        self.labels_paths = labels_paths
        self.model_path = model_path
        self.pdf_path = pdf_path
        self.inference = Inference(model_path)
        self.pads_union = self.get_pads_union()
        self.processed = False


    def process_data(self) -> None:
        self.features, self.labels = self.get_features_and_labels()
        self.predictions = self.get_predictions()
        self.unique_indices, self.ranking = self.get_unique_ranking()
        self.features = self.features[self.unique_indices]
        self.labels = self.labels[self.unique_indices]
        self.predictions = self.predictions[self.unique_indices]
        self.processed = True


    def draw(self) -> None:
        if not self.processed:
            self.process_data()
        self.draw_highest_score()


    def write(self) -> None:
        fname = Path("features.ranked.npy")
        logger.info(f"Writing ranked features to {fname}")
        if not self.processed:
            self.process_data()
        features = self.features[self.ranking][:self.num]
        print("self.features.shape", self.features.shape)
        print("self.ranking.shape", self.ranking.shape)
        print("self.features[self.ranking].shape", self.features[self.ranking].shape)
        print("features.shape", features.shape)
        np.save(fname, features)


    def get_pads_union(self) -> int: # honestly idk what this returns
        LAYER = 0
        return shapely.ops.unary_union(self.pads.layer[LAYER]["geometry"])


    def get_features_and_labels(self) -> Tuple[np.array, np.array]:
        features = np.concatenate([np.load(path) for path in self.features_paths])
        labels = np.concatenate([np.load(path) for path in self.labels_paths])
        return features, labels


    def get_predictions(self) -> np.array:
        return self.inference.predict(torch.Tensor(self.features)).detach().numpy()


    def get_unique_ranking(self) -> np.array:
        logger.info(f"Getting unique ranking (slow)")
        _, unique_indices = np.unique(self.features, axis=0, return_index=True)
        # unique_indices = np.arange(len(self.features))
        return unique_indices, np.flip(np.argsort(self.predictions[unique_indices], axis=0)).squeeze()


    def draw_highest_score(self) -> None:
        with PdfPages(self.pdf_path) as pdf:
            for i in range(self.num):
                self.draw_feature(i, pdf)


    def draw_feature(self, i: int, pdf: PdfPages) -> None:
        feature = self.features[self.ranking[i]]
        prediction = self.predictions[self.ranking[i]]

        pad_numbers = np.flatnonzero(feature)
        pad_polygons = self.pads.df.iloc[pad_numbers]["geometry"]
        assert len(pad_polygons) == constants.LAYERS, "Need npads = nlayers for now"
        self.draw_polygons(pad_polygons, prediction.item(), pdf)


    def draw_polygons(self, polygons: List[Polygon], prediction: float, pdf: PdfPages):
        """
        Draw a list of polygons to pdf
        Draw the polygon with transparent fill
        """
        ncols = 2
        fig, ax = plt.subplots(ncols=ncols, figsize=(8, 4))
        for col in range(ncols):
            for ip, poly in enumerate(polygons):
                if poly is None:
                    continue
                x, y = poly.exterior.xy
                ax[col].plot(x, y, label=f"L{ip}")
                ax[col].fill(x, y, alpha=0.1)
            ax[col].legend(frameon=False, fontsize=6)
            if col == 0:
                min_x, min_y = self.feature_xy(polygons, MIN)
                max_x, max_y = self.feature_xy(polygons, MAX)
                buffer_x = 0.12 * (max_x - min_x)
                buffer_y = 0.12 * (max_y - min_y)
                ax[col].set_xlim(min_x - buffer_x, max_x + buffer_x)
                ax[col].set_ylim(min_y - buffer_y, max_y + buffer_y)
            else:
                for poly in self.pads_union.geoms:
                    x, y = poly.exterior.xy
                    ax[col].plot(x, y, color="black", linewidth=0.5)

            ax[col].set_xlabel("x [mm]")
            ax[col].set_ylabel("y [mm]")
            ax[col].set_title(f"Prediction: {prediction:.5f}", fontsize=8)
            ax[col].tick_params(right=True, top=True)
        plt.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.12, wspace=0.25)
        pdf.savefig()
        plt.close()


    def feature_xy(
            self,
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
