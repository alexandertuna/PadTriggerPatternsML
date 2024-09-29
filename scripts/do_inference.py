import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Tuple

from pads_ml.inference import Inference

import logging
logging.basicConfig(level=logging.INFO)

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", help="Input features npy file", required=True)
    parser.add_argument("--labels", help="Input labels npy file", required=True)
    parser.add_argument("--model", help="Input model pth file", required=True)
    parser.add_argument("--pdf", help="Output pdf file for plots", required=True)
    return parser.parse_args()


def main() -> None:

    # CL args
    ops = options()
    features_path = Path(ops.features)
    labels_path = Path(ops.labels)
    model_path = Path(ops.model)
    pdf_path = Path(ops.pdf)

    # Get labels and predictions
    features, labels, predictions = get_labels_and_predictions(
        features_path,
        labels_path,
        model_path
    )
    logging.info(f"Labels: {labels}")
    logging.info(f"Predictions: {predictions}")

    # Make plots
    with PdfPages(pdf_path) as pdf:
        plot_all(
            labels,
            predictions,
            pdf,
        )
        plot_unique(
            features,
            labels,
            predictions,
            pdf,
        )


def get_labels_and_predictions(
    features_path: Path,
    labels_path: Path,
    model_path: Path
) -> Tuple[np.array, np.array, np.array]:

    # Features and model
    features = np.load(features_path)
    labels = np.load(labels_path)
    model = torch.load(model_path)
    logging.info(f"Features shape: {features.shape}")

    # Inference
    inference = Inference(model)
    predictions = inference.predict(torch.Tensor(features))
    return features, labels, predictions.detach().numpy()


def plot_all(
    labels: np.array,
    predictions: np.array,
    pdf: PdfPages,
) -> None:

    is_signal = labels == 1

    fig, ax = plt.subplots(figsize=(4, 4))
    bins = np.linspace(-10, 0, 100)
    # bins = np.linspace(-0.0005, 0, 100)
    ax.hist(np.log(predictions[is_signal]), bins=bins, alpha=0.5, label="Signal")
    ax.hist(np.log(predictions[~is_signal]), bins=bins, alpha=0.5, label="Noise")
    ax.set_xlabel("log(Probability)")
    ax.set_ylabel("Counts")
    ax.semilogy()
    ax.legend()
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.12)
    pdf.savefig(fig)
    plt.close()


def plot_unique(
    features: np.array,
    labels: np.array,
    predictions: np.array,
    pdf: PdfPages,
) -> None:

    is_signal = np.flatnonzero(labels == 1)

    logging.info(f"Getting unique (slow)")
    unique_rows, first_indices = np.unique(features, axis=0, return_index=True)
    first_signal_indices = np.intersect1d(is_signal, first_indices)
    logging.info(f"unique_rows: {unique_rows.shape}")
    logging.info(f"is_signal: {is_signal.shape}")
    logging.info(f"first_signal_indices: {first_signal_indices.shape}")

    fig, ax = plt.subplots(figsize=(4, 4))
    bins = np.linspace(-2, 0, 100)
    ax.hist(np.log(predictions[first_signal_indices]), bins=bins, alpha=0.5, label="Signal (unique)")
    ax.set_xlabel("log(Probability)")
    ax.set_ylabel("Counts")
    ax.semilogy()
    ax.legend()
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.12)
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots(figsize=(4, 4))
    bins = np.linspace(0, 1, 100)
    ax.hist(predictions[first_signal_indices], bins=bins, alpha=0.5, label="Signal (unique)")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Counts")
    ax.semilogy()
    ax.legend()
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.12)
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots(figsize=(4, 4))
    bins = np.linspace(0, 1, 100)
    ax.hist(predictions[first_signal_indices], bins=bins, alpha=0.5, label="Signal (unique)")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Counts")
    ax.legend()
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.12)
    pdf.savefig(fig)
    plt.close()


if __name__ == "__main__":
    main()
