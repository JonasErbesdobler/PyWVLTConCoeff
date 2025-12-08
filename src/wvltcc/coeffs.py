"""Main module for wavelet continuity coefficients calculations."""

import numpy as np
import pywt
import math


def two_term_coeff_solver(
    wvlt: str, deriv_order: int, residuals: bool = False
) -> np.ndarray:
    """Solves for the two-term connection coefficients of a given wavelet.

    Parameters
    ----------
    wvlt : str
        Name of the wavelet (must be recognized by PyWavelets).
    deriv_order : int
        Derivative order for which to compute the coefficients.
    residuals : bool, optional
        If True, also return the residuals of the refinement and moment equations.

    Returns
    -------
    np.ndarray
        Array of two-term connection coefficients.
    tuple, optional
        If residuals is True, returns a tuple containing the coefficients and the residuals.
    """

    # get filter bank of specified wavelet
    rec_lo = np.asarray(pywt.Wavelet(wvlt).filter_bank[2])

    # remove zeros from the reconstruction low-pass filter
    filt = rec_lo[rec_lo != 0]

    # set constants
    filt_len = len(filt)
    filt_idx = np.arange(-filt_len + 2, filt_len - 1)
    num_overlap = 2 * filt_len - 3

    # initialize the matrices for the eigenvalue problem
    two_term_ref = np.zeros((num_overlap, num_overlap))

    # populate the matrix from the refinement relations using a lookup table (more efficient)
    idx_lut = np.zeros(3 * num_overlap - 1)
    for c in range(num_overlap + 2):
        for n0 in range(filt_len):
            for n1 in range(filt_len):
                if (n0 - n1) == (c - filt_len + 1):
                    idx_lut[c + num_overlap - 1] += filt[n0] * filt[n1]
    for r in range(num_overlap):
        two_term_ref[r, :] = np.roll(idx_lut, 2 * r)[-num_overlap:]

    # populate the matrix from the refinement relations using nested loops (closely follows equations)
    # for l in range(num_overlap**2):
    #     r = l // num_overlap
    #     c = l % num_overlap
    #     k_in = c - filt_len + 2
    #     k_out = r - filt_len + 2
    #     for n0 in range(filt_len):
    #         for n1 in range(filt_len):
    #             if (n0 - n1) == (k_in - 2 * k_out):
    #                 two_term_ref[r, c] += filt[n0] * filt[n1]

    # build the least squares problem components of the refinement relation
    two_term_ref_mat = two_term_ref - (1 / 2**deriv_order) * np.eye(num_overlap)
    two_term_ref_rhs = np.zeros(num_overlap)

    # build the least squares problem components of the moment equation
    two_term_mom_lhs = np.empty((1, num_overlap))
    two_term_mom_lhs[0, :] = filt_idx**deriv_order
    two_term_mom_rhs = np.empty(1)
    two_term_mom_rhs[0] = math.factorial(deriv_order) * (-1) ** deriv_order

    # build the least squares problem
    lhs = np.vstack((two_term_ref_mat, two_term_mom_lhs))
    rhs = np.concatenate((two_term_ref_rhs, two_term_mom_rhs))

    # solve least squares approximation (numpy uses SVD under the hood)
    two_term_coeff, res, rank, s = np.linalg.lstsq(lhs, rhs, rcond=None)

    if residuals:
        # check residuals for refinement and moment equations
        ref_error = np.linalg.norm(
            np.dot(two_term_ref, two_term_coeff) - 1 / 2**deriv_order * two_term_coeff
        )
        mom_error = np.linalg.norm(
            np.dot(two_term_mom_lhs, two_term_coeff) - two_term_mom_rhs
        )

        return two_term_coeff, ref_error, mom_error

    return two_term_coeff


def three_term_coeff_solver(
    wvlt: str, deriv_orders: np.ndarray, residuals: bool = False
) -> np.ndarray:
    """Solves for the two-term connection coefficients of a given wavelet.

    Parameters
    ----------
    wvlt : str
        Name of the wavelet (must be recognized by PyWavelets).
    deriv_orders : np.ndarray
        Array of derivative orders for which to compute the coefficients.
    residuals : bool, optional
        If True, also return the residuals of the refinement and moment equations.

    Returns
    -------
    np.ndarray
        Array of two-term connection coefficients.
    tuple, optional
        If residuals is True, returns a tuple containing the coefficients and the residuals.
    """

    # get filter bank of specified wavelet
    rec_lo = np.asarray(pywt.Wavelet(wvlt).filter_bank[2])

    # remove zeros from the reconstruction low-pass
    filt = rec_lo[rec_lo != 0]

    # set constants
    filt_len = len(filt)
    filt_idx = np.arange(-filt_len + 2, filt_len - 1)
    num_nonzero_coeffs = 2 * filt_len - 3

    # get the order of derivation for each term
    i1 = deriv_orders[0]
    i2 = deriv_orders[1]

    # compute two-term coefficients for each derivation order
    two_term_i1 = two_term_coeff_solver(wvlt, i1)
    two_term_i2 = two_term_coeff_solver(wvlt, i2)

    def map_indices(j, k):
        idx = (j + filt_len - 2) * num_nonzero_coeffs + (k + filt_len - 2)
        return idx

    # populate the block matrix from refinement relations
    three_term_ref = np.zeros((num_nonzero_coeffs**2, num_nonzero_coeffs**2))
    for k1_out in filt_idx:
        for k2_out in filt_idx:
            r = map_indices(k1_out, k2_out)
            for n0 in range(filt_len):
                for n1 in range(filt_len):
                    for n2 in range(filt_len):
                        k1_in = 2 * k1_out + n1 - n0
                        k2_in = 2 * k2_out + n2 - n0
                        if (
                            np.abs(k1_in) < filt_len - 1
                            and np.abs(k2_in) < filt_len - 1
                        ):
                            c = map_indices(k1_in, k2_in)
                            three_term_ref[r, c] += filt[n0] * filt[n1] * filt[n2]

    # populate the block matrix from the moment equations
    three_term_mom_mat = np.zeros((2 * num_nonzero_coeffs, num_nonzero_coeffs**2))
    three_term_mom_rhs = np.zeros(2 * num_nonzero_coeffs)

    # moment equation for m
    for i, k1_phys in enumerate(filt_idx):
        for k2_phys in filt_idx:
            col_idx = map_indices(k1_phys, k2_phys)
            three_term_mom_mat[i, col_idx] = k2_phys**i2
        three_term_mom_rhs[i] = math.factorial(i2) * two_term_i1[i]

    # moment equation for n
    for i, k2_phys in enumerate(filt_idx):
        for k1_phys in filt_idx:
            col_idx = map_indices(k1_phys, k2_phys)
            three_term_mom_mat[i + num_nonzero_coeffs, col_idx] = k1_phys**i1
        three_term_mom_rhs[i + num_nonzero_coeffs] = math.factorial(i1) * two_term_i2[i]

    # target eigenvalue
    target_lambda = 1 / 2 ** (i1 + i2 + 0.5)

    # construct eigenvalue problem
    three_term_ref_mat = three_term_ref - target_lambda * np.eye(num_nonzero_coeffs**2)
    three_term_ref_rhs = np.zeros(num_nonzero_coeffs**2)

    # build the least squares problem
    lhs = np.vstack((three_term_ref_mat, three_term_mom_mat))
    rhs = np.concatenate((three_term_ref_rhs, three_term_mom_rhs))

    # solve least squares approximation
    three_term_coeff, res, rank, s = np.linalg.lstsq(lhs, rhs, rcond=None)

    if residuals:
        # check residuals for refinement and moment equations
        ref_error = np.linalg.norm(
            np.dot(three_term_ref, three_term_coeff) - target_lambda * three_term_coeff
        )
        mom_error = np.linalg.norm(
            np.dot(three_term_mom_mat, three_term_coeff) - three_term_mom_rhs
        )

        return three_term_coeff, ref_error, mom_error

    return three_term_coeff
