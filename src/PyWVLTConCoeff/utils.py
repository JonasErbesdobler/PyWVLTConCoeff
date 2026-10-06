"""Utilities for wavelet coefficient calculations and recreating plots from the paper."""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import Optional

from .coeffs import (
    two_term_coeff_solver,
    three_term_coeff_solver,
    four_term_coeff_solver,
)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams


def _axes_offset(ax: plt.Axes, font_size: float, em: float = 0.8) -> float:
    """Return axes coordinate offset for moving a spine by a font-scaled padding."""
    fig = ax.figure
    axes_height = fig.get_figheight() * ax.get_position().height
    if axes_height == 0:
        return 0.0
    padding_inches = (font_size / 72.0) * em
    return -padding_inches / axes_height


def _stacked_order_labels(ax: plt.Axes, deriv_orders: np.ndarray, font_size: float):
    """Label each bar with its derivative orders, stacked in one row per order."""
    line_height = 1.15 * font_size
    ax.set_xticks(np.arange(deriv_orders.shape[0]))
    ax.set_xticklabels([])
    for row in range(deriv_orders.shape[1]):
        # baseline of the row in points below the axis
        offset = -(0.35 * font_size + (row + 0.8) * line_height)
        ax.annotate(
            rf"$i_{row + 1}$",
            xy=(0, 0),
            xycoords="axes fraction",
            xytext=(-0.4 * font_size, offset),
            textcoords="offset points",
            ha="right",
            va="baseline",
        )
        for i in range(deriv_orders.shape[0]):
            ax.annotate(
                str(deriv_orders[i, row]),
                xy=(i, 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="baseline",
            )


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
    font_size: int = 14,
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

    # create the output directory if it does not exist
    os.makedirs(path, exist_ok=True)

    # set font size for plots
    rcParams.update({"font.size": font_size})

    # initialize lists to hold residuals
    res_ref = []
    res_mom = []

    # compute residuals for each derivative order
    for order in deriv_orders:
        _, ref_res, mom_res = two_term_coeff_solver(wvlt, order, residuals=True)
        res_ref.append(ref_res)
        res_mom.append(mom_res)

    # define constant offsets and sizes
    plot_vert = 3.0
    plot_horz = 2.0
    bottom_pad = 0.7 * (font_size / 14)
    top_pad = 0.2
    left_pad = 1.1 * (font_size / 14)
    right_pad = 0.2

    # calculate figure size
    fig_height = plot_vert + bottom_pad + top_pad
    fig_width = plot_horz + left_pad + right_pad

    # plot refinement residuals
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes(
        [
            left_pad / fig_width,
            bottom_pad / fig_height,
            plot_horz / fig_width,
            plot_vert / fig_height,
        ]
    )
    for i in range(len(deriv_orders)):
        ax.bar(i, res_ref[i], label=rf"$i = {i}$", color="#2b8057")
    ax.set_xlim(-0.6, deriv_orders.shape[0] - 0.4)
    ax.set_xticks(range(len(deriv_orders)))
    ax.set_xticklabels([str(i) for i in deriv_orders])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel(r"$i$")
    ax.set_ylabel("Refinement residual")
    ax.grid(visible=True, which="major", axis="y", linestyle="-")
    fig.savefig(
        os.path.join(path, f"two_term_{wvlt}_refinement_residuals.{format}"),
        dpi=fig_dpi,
    )
    plt.close(fig)

    # plot moment residuals
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes(
        [
            left_pad / fig_width,
            bottom_pad / fig_height,
            plot_horz / fig_width,
            plot_vert / fig_height,
        ]
    )
    for i in range(len(deriv_orders)):
        ax.bar(i, res_mom[i], label=rf"$i = {i}$", color="#2b8057")
    ax.set_xlim(-0.6, deriv_orders.shape[0] - 0.4)
    ax.set_xticks(range(len(deriv_orders)))
    ax.set_xticklabels([str(i) for i in deriv_orders])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel(r"$i$")
    ax.set_ylabel("Moments residual")
    ax.grid(visible=True, which="major", axis="y", linestyle="-")
    fig.savefig(
        os.path.join(path, f"two_term_{wvlt}_momentum_residuals.{format}"),
        dpi=fig_dpi,
    )
    plt.close(fig)

    pass


def plot_three_term_residuals(
    wvlt: str,
    deriv_orders: np.ndarray,
    path: str,
    format: str = "pdf",
    fig_dpi: int = 100,
    font_size: int = 14,
):
    """Plots the residuals of the three-term connection coefficient calculations.

    Parameters
    ----------
    wvlt : str
        Name of the wavelet (must be recognized by PyWavelets).
    deriv_orders : np.ndarray
        (2xN) or (Nx2) array of derivative orders to plot. A (2x2) array is
        interpreted as (2xN), i.e., each column is one set of derivative orders.
    path : str
        The directory path where the plot will be saved.
    """

    # ensure deriv_orders is of shape (Nx2)
    shp = deriv_orders.shape
    if shp[0] != 2 and shp[1] != 2:
        raise ValueError("deriv_orders must be of shape (2xN) or (Nx2)")
    if shp[0] == 2:
        deriv_orders = deriv_orders.T

    # sort derivative orders by first column
    deriv_orders = deriv_orders[np.lexsort((deriv_orders[:, 1], deriv_orders[:, 0]))]

    # create the output directory if it does not exist
    os.makedirs(path, exist_ok=True)

    # set font size for plots
    rcParams.update({"font.size": font_size})

    # initialize lists to hold residuals
    res_ref = []
    res_mom = []

    # compute residuals for each derivative order
    for i in range(deriv_orders.shape[0]):
        _, ref_res, mom_res = three_term_coeff_solver(
            wvlt, deriv_orders[i, :], residuals=True
        )
        res_ref.append(ref_res)
        res_mom.append(mom_res)

    # define constant offsets and sizes
    plot_vert = 3.0
    plot_horz = 7.0
    bottom_pad = 1.4 * (font_size / 14)
    top_pad = 0.2
    left_pad = 1.1 * (font_size / 14)
    right_pad = 0.2

    # calculate figure size
    fig_height = plot_vert + bottom_pad + top_pad
    fig_width = plot_horz + left_pad + right_pad

    # plot refinement residuals
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes(
        [
            left_pad / fig_width,
            bottom_pad / fig_height,
            plot_horz / fig_width,
            plot_vert / fig_height,
        ]
    )
    for i in range(deriv_orders.shape[0]):
        ax.bar(i, res_ref[i], color="#2b8057")
    ax.set_xlim(-0.6, deriv_orders.shape[0] - 0.4)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel(r"$i_1$")
    ax.set_ylabel("Refinement residual")
    ax.set_xticks(np.arange(deriv_orders.shape[0]))
    ax.set_xticklabels([str(i) for i in deriv_orders[:, 0]])
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("axes", -0.25))
    ax2.set_xlabel(r"$i_2$")
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([str(i) for i in deriv_orders[:, 1]])
    ax.grid(visible=True, which="major", axis="y", linestyle="-")
    fig.savefig(
        os.path.join(path, f"three_term_{wvlt}_refinement_residuals.{format}"),
        dpi=fig_dpi,
    )
    plt.close(fig)

    # plot moment residuals
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes(
        [
            left_pad / fig_width,
            bottom_pad / fig_height,
            plot_horz / fig_width,
            plot_vert / fig_height,
        ]
    )
    for i in range(deriv_orders.shape[0]):
        ax.bar(i, res_mom[i], color="#2b8057")
    ax.set_xlim(-0.6, deriv_orders.shape[0] - 0.4)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel(r"$i_1$")
    ax.set_ylabel("Moments residual")
    ax.set_xticks(range(deriv_orders.shape[0]))
    ax.set_xticklabels([str(i) for i in deriv_orders[:, 0]])
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("axes", -0.25))
    ax2.set_xlabel(r"$i_2$")
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([str(i) for i in deriv_orders[:, 1]])
    ax.grid(visible=True, which="major", axis="y", linestyle="-")
    fig.savefig(
        os.path.join(path, f"three_term_{wvlt}_momentum_residuals.{format}"),
        dpi=fig_dpi,
    )
    plt.close(fig)

    pass


def plot_four_term_residuals(
    wvlt: str,
    deriv_orders: np.ndarray,
    path: str,
    format: str = "pdf",
    fig_dpi: int = 100,
    fig_size: tuple = (16, 5),
    font_size: int = 14,
    max_order: Optional[int] = None,
):
    """Plots the residuals of the four-term connection coefficient calculations.

    Parameters
    ----------
    wvlt : str
        Name of the wavelet (must be recognized by PyWavelets).
    deriv_orders : np.ndarray
        (3xN) or (Nx3) array of derivative orders to plot. A (3x3) array is
        interpreted as (3xN), i.e., each column is one set of derivative orders.
    path : str
        The directory path where the plot will be saved.
    max_order : int, optional
        If given, only derivative orders with all individual orders less than or
        equal to max_order are plotted, and the file names get the suffix
        "_max_order_{max_order}".
    """

    # ensure deriv_orders is of shape (Nx3)
    shp = deriv_orders.shape
    if shp[0] != 3 and shp[1] != 3:
        raise ValueError("deriv_orders must be of shape (3xN) or (Nx3)")
    if shp[0] == 3:
        deriv_orders = deriv_orders.T

    # filter deriv_orders to only include those with individual orders less than or equal to max_order
    suffix = ""
    if max_order is not None:
        deriv_orders = deriv_orders[np.all(deriv_orders <= max_order, axis=1)]
        suffix = f"_max_order_{max_order}"

    # sort derivative orders by first column
    deriv_orders = deriv_orders[
        np.lexsort((deriv_orders[:, 2], deriv_orders[:, 1], deriv_orders[:, 0]))
    ]

    # create the output directory if it does not exist
    os.makedirs(path, exist_ok=True)

    # set font size for plots
    rcParams.update({"font.size": font_size})

    # initialize lists to hold residuals
    res_ref = []
    res_mom = []

    # compute residuals for each derivative order
    for i in range(deriv_orders.shape[0]):
        _, ref_res, mom_res = four_term_coeff_solver(
            wvlt, deriv_orders[i, :], residuals=True
        )
        res_ref.append(ref_res)
        res_mom.append(mom_res)

    # define constant offsets and sizes
    plot_vert = 2.6
    plot_horz = 15.0 if max_order is None else 6.0
    bottom_pad = 0.75 * (font_size / 14)
    top_pad = 0.2
    left_pad = 1.1 * (font_size / 14)
    right_pad = 0.2

    # calculate figure size
    fig_height = plot_vert + bottom_pad + top_pad
    fig_width = plot_horz + left_pad + right_pad

    # plot refinement residuals
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes(
        [
            left_pad / fig_width,
            bottom_pad / fig_height,
            plot_horz / fig_width,
            plot_vert / fig_height,
        ]
    )
    for i in range(deriv_orders.shape[0]):
        ax.bar(i, res_ref[i], color="#2b8057")
    ax.set_xlim(-0.6, deriv_orders.shape[0] - 0.4)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylabel("Refinement residual")
    _stacked_order_labels(ax, deriv_orders, font_size)
    ax.grid(visible=True, which="major", axis="y", linestyle="-")
    fig.savefig(
        os.path.join(path, f"four_term_{wvlt}_refinement_residuals{suffix}.{format}"),
        dpi=fig_dpi,
    )
    plt.close(fig)

    # plot moment residuals
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes(
        [
            left_pad / fig_width,
            bottom_pad / fig_height,
            plot_horz / fig_width,
            plot_vert / fig_height,
        ]
    )
    for i in range(deriv_orders.shape[0]):
        ax.bar(i, res_mom[i], color="#2b8057")
    ax.set_xlim(-0.6, deriv_orders.shape[0] - 0.4)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylabel("Moments residual")
    _stacked_order_labels(ax, deriv_orders, font_size)
    ax.grid(visible=True, which="major", axis="y", linestyle="-")
    fig.savefig(
        os.path.join(path, f"four_term_{wvlt}_momentum_residuals{suffix}.{format}"),
        dpi=fig_dpi,
    )
    plt.close(fig)

    pass


def plot_four_term_residuals_up_to(
    wvlt: str,
    deriv_orders: np.ndarray,
    max_order: int,
    path: str,
    format: str = "pdf",
    fig_dpi: int = 100,
    fig_size: tuple = (8, 5),
    font_size: int = 14,
):
    """Plots the residuals of the four-term connection coefficient calculations
    for all derivative orders up to max_order. Equivalent to calling
    plot_four_term_residuals with the max_order argument.

    Parameters
    ----------
    wvlt : str
        Name of the wavelet (must be recognized by PyWavelets).
    deriv_orders : np.ndarray
        (3xN) or (Nx3) array of derivative orders to plot.
    max_order : int
        The maximum individual order of the derivative to plot.
    path : str
        The directory path where the plot will be saved.
    """

    plot_four_term_residuals(
        wvlt,
        deriv_orders,
        path,
        format=format,
        fig_dpi=fig_dpi,
        fig_size=fig_size,
        font_size=font_size,
        max_order=max_order,
    )
