"""Tests for the connection coefficient solvers."""

import itertools
import warnings

import numpy as np
import pytest

from PyWVLTConCoeff.coeffs import (
    two_term_coeff_solver,
    two_term_coeff_solver_low_mem,
    three_term_coeff_solver,
    three_term_coeff_solver_low_mem,
    four_term_coeff_solver,
    four_term_coeff_solver_low_mem,
)

WVLT = "db3"
NUM = 9  # number of non-zero shifts per dimension, 2N - 3 with N = 6
SHIFTS = range(-4, 5)

# all derivative orders up to two, for which the db3 systems are consistent
ORDERS_2 = list(itertools.product(range(3), repeat=2))
ORDERS_3 = list(itertools.product(range(3), repeat=3))

# tabulated values of the four-term (0),(1),(1) coefficient from the paper
TABLE_011 = {
    (-4, -4, -4): -3.0e-09,
    (-3, -1, -2): -0.0018982632,
    (-2, -1, -1): -0.0260262038,
    (-1, -1, -1): 0.2055946136,
    (0, -1, 0): -1.9145852962,
    (0, 0, 0): 2.3010124780,
    (0, 1, 0): -0.9221475291,
    (1, 2, 1): 0.2209473009,
    (2, 3, 2): -0.0308007421,
    (4, 4, 4): -2.5501e-06,
}


@pytest.fixture(autouse=True)
def _warnings_as_errors():
    """No solver warning is expected unless a test asks for it explicitly."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.mark.parametrize("order", range(3))
def test_two_term_residuals(order):
    coeff, ref, mom = two_term_coeff_solver(WVLT, order, residuals=True)
    assert coeff.shape == (NUM,)
    assert ref < 1e-13
    assert mom < 1e-13


@pytest.mark.parametrize("orders", ORDERS_2)
def test_three_term_residuals(orders):
    coeff, ref, mom = three_term_coeff_solver(WVLT, orders, residuals=True)
    assert coeff.shape == (NUM**2,)
    assert ref < 1e-11
    assert mom < 1e-11


@pytest.mark.parametrize("orders", ORDERS_3)
def test_four_term_residuals(orders):
    coeff, ref, mom = four_term_coeff_solver(WVLT, orders, residuals=True)
    assert coeff.shape == (NUM**3,)
    assert ref < 1e-9
    assert mom < 1e-9


def test_two_term_known_values():
    # the scaling functions are orthonormal
    np.testing.assert_allclose(
        two_term_coeff_solver(WVLT, 0), np.eye(NUM)[NUM // 2], atol=1e-13
    )

    # first derivative coefficients of db3 are known in closed form
    expected = np.array([-1 / 2920, -16 / 1095, 53 / 365, -272 / 365, 0.0])
    expected = np.concatenate((expected, -expected[-2::-1]))
    np.testing.assert_allclose(two_term_coeff_solver(WVLT, 1), expected, atol=1e-13)


def test_low_mem_matches_vectorized():
    np.testing.assert_allclose(
        two_term_coeff_solver_low_mem(WVLT, 1),
        two_term_coeff_solver(WVLT, 1),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        three_term_coeff_solver_low_mem(WVLT, [1, 2]),
        three_term_coeff_solver(WVLT, [1, 2]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        four_term_coeff_solver_low_mem(WVLT, [0, 1, 2]),
        four_term_coeff_solver(WVLT, [0, 1, 2]),
        atol=1e-12,
    )


def test_four_term_matches_paper_table():
    coeff = four_term_coeff_solver(WVLT, [0, 1, 1]).reshape(NUM, NUM, NUM)
    for (k1, k2, k3), value in TABLE_011.items():
        assert coeff[k1 + 4, k2 + 4, k3 + 4] == pytest.approx(value, abs=1e-10)


def test_four_term_shift_symmetry():
    # exchanging the two non-differentiated scaling functions leaves the coefficient unchanged
    coeff = four_term_coeff_solver(WVLT, [0, 1, 2]).reshape(NUM, NUM, NUM)
    for k1, k2, k3 in itertools.product(SHIFTS, repeat=3):
        m2, m3 = k2 - k1, k3 - k1
        if abs(m2) <= 4 and abs(m3) <= 4:
            assert coeff[k1 + 4, k2 + 4, k3 + 4] == pytest.approx(
                coeff[-k1 + 4, m2 + 4, m3 + 4], abs=1e-10
            )


def test_four_term_permutation_symmetry():
    # equal derivative orders make the coefficient symmetric in its shifts
    coeff = four_term_coeff_solver(WVLT, [1, 1, 1]).reshape(NUM, NUM, NUM)
    np.testing.assert_allclose(coeff, coeff.transpose(1, 0, 2), atol=1e-10)
    np.testing.assert_allclose(coeff, coeff.transpose(0, 2, 1), atol=1e-10)

    coeff = four_term_coeff_solver(WVLT, [0, 1, 1]).reshape(NUM, NUM, NUM)
    np.testing.assert_allclose(coeff, coeff.transpose(0, 2, 1), atol=1e-10)


def test_four_term_vanishes_without_overlap():
    coeff = four_term_coeff_solver(WVLT, [0, 1, 2]).reshape(NUM, NUM, NUM)
    for k1, k2, k3 in itertools.product(SHIFTS, repeat=3):
        if max(abs(k1 - k2), abs(k1 - k3), abs(k2 - k3)) > 4:
            assert abs(coeff[k1 + 4, k2 + 4, k3 + 4]) < 1e-11


@pytest.mark.parametrize("wvlt", ["db2", "db4", "sym4", "coif1", "bior2.2"])
def test_other_wavelets(wvlt):
    _, ref, mom = two_term_coeff_solver(wvlt, 1, residuals=True)
    assert ref < 1e-13
    assert mom < 1e-13


def test_deriv_orders_accepts_sequences():
    reference = three_term_coeff_solver(WVLT, np.array([0, 1]))
    np.testing.assert_array_equal(three_term_coeff_solver(WVLT, [0, 1]), reference)
    np.testing.assert_array_equal(three_term_coeff_solver(WVLT, (0, 1)), reference)


@pytest.mark.parametrize(
    "solver, orders",
    [
        (two_term_coeff_solver, -1),
        (two_term_coeff_solver, 1.0),
        (two_term_coeff_solver_low_mem, -1),
        (three_term_coeff_solver, [1, -1]),
        (three_term_coeff_solver, [1.0, 1.0]),
        (three_term_coeff_solver, [1, 1, 1]),
        (three_term_coeff_solver_low_mem, [1]),
        (four_term_coeff_solver, [1, 1]),
        (four_term_coeff_solver, [0, 1, -2]),
        (four_term_coeff_solver_low_mem, [[0, 1, 2]]),
    ],
)
def test_invalid_deriv_orders(solver, orders):
    with pytest.raises(ValueError):
        solver(WVLT, orders)


@pytest.mark.parametrize(
    "solver, wvlt, orders",
    [
        (two_term_coeff_solver, "db2", 2),
        (three_term_coeff_solver, "db2", [1, 1]),
        (four_term_coeff_solver, "db2", [1, 1, 1]),
        (three_term_coeff_solver, "db3", [3, 3]),
    ],
)
def test_warns_if_inconsistent(solver, wvlt, orders):
    with pytest.warns(RuntimeWarning):
        solver(wvlt, orders)
