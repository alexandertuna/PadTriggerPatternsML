import numpy as np
import pandas as pd

from pads_ml import constants

#
# convert pd.dataframe to one-hot np.array
#
def onehotify(df: pd.DataFrame) -> np.array:

    onehot = pd.DataFrame({
        f"pad_{layer}": pd.Categorical(
            df[f"pad_{layer}"],
            categories=constants.PADS_PER_LAYER[layer]
        )
        for layer in range(constants.LAYERS)
    })

    return pd.get_dummies(onehot).values

#
# convert one-hot np.array to dataframe-style np.array
#
def undo_onehot(features: np.array) -> np.array:
    rows, cols = features.shape

    if cols != constants.PADS:
        raise ValueError(f"Expected {constants.PADS} columns, got {cols}")

    non_onehot_array = -1 * np.ones((rows, constants.LAYERS), dtype=int)

    for layer, pads_per_layer in enumerate(constants.PADS_PER_LAYER):
        layer_onehot = features[:, pads_per_layer]
        no_pads = np.all(layer_onehot == 0, axis=1)
        hot_indices = np.argmax(layer_onehot, axis=1)
        non_onehot_array[:, layer] = np.where(no_pads, -1, hot_indices + pads_per_layer.start)

    return non_onehot_array
