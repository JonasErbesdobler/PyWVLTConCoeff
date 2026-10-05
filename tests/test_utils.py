"""Tests for the plotting utilities."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from PyWVLTConCoeff import utils  # noqa: E402


def test_two_term_plot_creates_missing_directory(tmp_path):
    path = tmp_path / "figures" / "nested"
    utils.plot_two_term_residuals("db3", np.arange(3), path=str(path))
    assert (path / "two_term_db3_refinement_residuals.pdf").is_file()
    assert (path / "two_term_db3_momentum_residuals.pdf").is_file()


def test_four_term_plot_max_order(tmp_path):
    x, y, z = np.meshgrid(np.arange(2), np.arange(2), np.arange(2), indexing="ij")
    deriv_orders = np.vstack([x.ravel(), y.ravel(), z.ravel()])

    utils.plot_four_term_residuals("db2", deriv_orders, str(tmp_path), max_order=0)
    utils.plot_four_term_residuals_up_to(
        "db2", deriv_orders, max_order=0, path=str(tmp_path / "up_to")
    )

    for path in (tmp_path, tmp_path / "up_to"):
        assert (path / "four_term_db2_refinement_residuals_max_order_0.pdf").is_file()
        assert (path / "four_term_db2_momentum_residuals_max_order_0.pdf").is_file()
