import argparse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsmeared", help="Data file without smearing", required=True)
    parser.add_argument("--smeared", help="Data file with smearing", required=True)
    parser.add_argument("--output", help="Output file", required=True)
    return parser.parse_args()


def main() -> None:
    ops = options()
    unsmeared = pd.read_parquet(ops.unsmeared)
    smeared = pd.read_parquet(ops.smeared)
    with PdfPages(ops.output) as pdf:
        plot_diff(unsmeared, smeared, pdf)


def plot_diff(
    unsmeared: pd.DataFrame,
    smeared: pd.DataFrame,
    pdf: PdfPages,
) -> None:
    
    # plot unsmeared - smeared
    fig, ax = plt.subplots(figsize=(4, 4))
    contents, edges, _ = ax.hist(unsmeared["x_0"] - smeared["x_0"], bins=100, label="Simulated")
    centers = (edges[:-1] + edges[1:]) / 2

    # Fit a Gaussian
    def gaussian(x, a, mu, sigma):
        return a * np.exp(- (x - mu)**2 / (2 * sigma**2))
    popt, pcov = curve_fit(gaussian, centers, contents, p0=[1000, 0, 10])
    a_fit, mu_fit, sigma_fit = popt

    # Plot the fit
    x_fit = np.linspace(min(centers), max(centers), 100)
    y_fit = gaussian(x_fit, *popt)
    ax.plot(x_fit, y_fit, label=f"$\\mu={mu_fit:.0f}, \\sigma={sigma_fit:.0f}$", color="red")

    # Make it pretty
    ax.legend()
    ax.set_xlabel("x_unsmeared - x_smeared")
    ax.set_ylabel("Frequency")
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.12)
    pdf.savefig(fig)
    plt.close()

if __name__ == "__main__":
    main()
