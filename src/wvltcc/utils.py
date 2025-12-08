"""Utilities for wavelet coefficient calculations and recreating plots from the paper."""

import numpy as np
import os
import matplotlib.pyplot as plt

from .coeffs import two_term_coeff_solver
from matplotlib.colors import LinearSegmentedColormap


def create_custom_colormap():
    """Creates a custom colormap for plotting."""
    map_colors = [
        "#020604",
        "#040c08",
        "#06120c",
        "#081811",
        "#0a1e15",
        "#0c2519",
        "#0e2b1d",
        "#113121",
        "#133725",
        "#153d29",
        "#17432e",
        "#194932",
        "#1b4f36",
        "#1d553a",
        "#1f5b3e",
        "#216142",
        "#236846",
        "#256e4a",
        "#27744f",
        "#297a53",
        "#2b8057",
        "#2e865b",
        "#308c5f",
        "#329263",
        "#349867",
        "#369e6c",
        "#38a470",
        "#3aab74",
        "#3cb178",
        "#3eb77c",
        "#40bd80",
        "#45c084",
        "#4bc288",
        "#51c48c",
        "#57c690",
        "#5ec894",
        "#64ca98",
        "#6acc9c",
        "#70cea0",
        "#76d0a4",
        "#7cd3a9",
        "#82d5ad",
        "#88d7b1",
        "#8ed9b5",
        "#94dbb9",
        "#9addbd",
        "#a1dfc1",
        "#a7e1c5",
        "#ade3c9",
        "#b3e5cd",
        "#b9e7d1",
        "#bfe9d5",
        "#c5ebd9",
        "#cbeddd",
        "#d1efe1",
        "#d7f2e5",
        "#ddf4e9",
        "#e4f6ed",
        "#eaf8f1",
        "#f0faf5",
    ]
    cm = LinearSegmentedColormap.from_list("PhthaloGreen", map_colors, N=256)
    return cm


def write_to_txt(path, filename, data):
    """Writes the given data to a text file.

    Parameters
    ----------
    path : str
        The directory path where the file will be saved.
    filename : str
        The name of the file to write to.
    data : array-like
        The data to write to the file.
    """
    filepath = os.path.join(path, filename)
    np.savetxt(filepath, data)


def read_from_txt(path, filename):
    """Reads data from a text file.
    Parameters
    ----------
    path : str
        The directory path where the file is located.
    filename : str
        The name of the file to read from.

    Returns
    -------
    np.ndarray
        The data read from the file.
    """
    filepath = os.path.join(path, filename)
    return np.loadtxt(filepath)


def write_to_npy(path, filename, data):
    """Writes the given data to a NumPy binary file.

    Parameters
    ----------
    path : str
        The directory path where the file will be saved.
    filename : str
        The name of the file to write to.
    data : array-like
        The data to write to the file.
    """
    filepath = os.path.join(path, filename)
    np.save(filepath, data)


def read_from_npy(path, filename):
    """Reads data from a NumPy binary file.

    Parameters
    ----------
    path : str
        The directory path where the file is located.
    filename : str
        The name of the file to read from.

    Returns
    -------
    np.ndarray
        The data read from the file.
    """
    filepath = os.path.join(path, filename)
    return np.load(filepath)


def plot_two_term_residuals(
    wvlt: str,
    deriv_orders: np.ndarray,
    path: str,
    format: str = "pdf",
    fig_dpi: int = 100,
    fig_size: tuple = (6, 5),
):
    """Plots the residuals of the two-term connection coefficient calculations.

    Parameters
    ----------
    wvlt : str
        Name of the wavelet (must be recognized by PyWavelets).
    deriv_orders : np.ndarray
        Array of derivative orders to plot.
    path : str
        The directory path where the plot will be saved.
    """

    # initialize lists to hold residuals
    res_ref = []
    res_mom = []

    # compute residuals for each derivative order
    for order in deriv_orders:
        _, ref_res, mom_res = two_term_coeff_solver(wvlt, order, residuals=True)
        res_ref.append(ref_res)
        res_mom.append(mom_res)

    # plot refinement residuals
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    for i in range(len(deriv_orders)):
        ax.bar(i, res_ref[i], label=rf"$i = {i}$", color="#2b8057")
    ax.set_yscale("log")
    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"$|| H \cdot \Gamma^{(i)} - \frac{1}{2^i} \Gamma^{(i)} ||_{L_2}$")
    ax.grid()
    fig.tight_layout()
    fig.savefig(
        os.path.join(path, f"two_term_{wvlt}_refinement_residuals.{format}"),
        dpi=fig_dpi,
    )
    plt.close(fig)

    # plot moment residuals
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    for i in range(len(deriv_orders)):
        ax.bar(i, res_mom[i], label=rf"$i = {i}$", color="#2b8057")
    ax.set_yscale("log")
    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"$|| k^{\circ i} \cdot \Gamma^{(i)} - i!(-1)^i ||_{L_2}$")
    ax.grid()
    fig.tight_layout()
    fig.savefig(
        os.path.join(path, f"two_term_{wvlt}_momentum_residuals.{format}"),
        dpi=fig_dpi,
    )
    plt.close(fig)
