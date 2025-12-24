"""Main module for wavelet continuity coefficients calculations."""

import numpy as np
import pywt
import math


def two_term_coeff_solver_low_mem(
    wvlt: str, deriv_order: int, residuals: bool = False
) -> np.ndarray:
    """Solves for the two-term connection coefficients of a given wavelet. This is
    a low-memory implementation, i.e., it avoids constructing large intermediate matrices,
    and runs single-core. This comes at the cost of computational speed.

    TODO: Change to sparse matrices to further reduce memory usage.

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

    # check that deriv_order is integer
    if not np.issubdtype(type(deriv_order), np.integer):
        raise ValueError("deriv_order must be an integer.")

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

    # populate the matrix from the refinement relations using nested loops (closely follows equations)
    for i in range(num_overlap**2):
        r = i // num_overlap
        c = i % num_overlap
        k_in = c - filt_len + 2
        k_out = r - filt_len + 2
        for n0 in range(filt_len):
            for n1 in range(filt_len):
                if (n0 - n1) == (k_in - 2 * k_out):
                    two_term_ref[r, c] += filt[n0] * filt[n1]

    # build the least squares problem components of the refinement relation
    two_term_ref_mat = two_term_ref - (1 / 2**deriv_order) * np.eye(num_overlap)
    two_term_ref_rhs = np.zeros(num_overlap)

    # build the least squares problem components of the moment equation
    two_term_mom_lhs = np.empty((1, num_overlap))
    two_term_mom_lhs[0, :] = filt_idx**deriv_order
    two_term_mom_rhs = np.empty(1)
    two_term_mom_rhs[0] = math.factorial(deriv_order)  # * (-1) ** deriv_order
    # The (-1)^deriv_order factor very much depends on the convention used for
    # connection coefficients. Here we follow the convention used in our paper.

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


def three_term_coeff_solver_low_mem(
    wvlt: str, deriv_orders: np.ndarray, residuals: bool = False
) -> np.ndarray:
    """Solves for the two-term connection coefficients of a given wavelet. This is
    a low-memory implementation, i.e., it avoids constructing large intermediate matrices,
    and runs single-core. This comes at the cost of computational speed.

    TODO: Change to sparse matrices to further reduce memory usage.

    Parameters
    ----------
    wvlt : str
        Name of the wavelet (must be recognized by PyWavelets).
    deriv_orders : np.ndarray
        1D, len=2 int array of derivative orders for which to compute the coefficients.
    residuals : bool, optional
        If True, also return the residuals of the refinement and moment equations.

    Returns
    -------
    np.ndarray
        Array of two-term connection coefficients.
    tuple, optional
        If residuals is True, returns a tuple containing the coefficients and the residuals.
    """

    # check that deriv_orders has length 2
    if len(deriv_orders) != 2:
        raise ValueError("deriv_orders must be a 1D array of length 2.")

    # check that deriv_orders are integers
    if not np.issubdtype(deriv_orders.dtype, np.integer):
        raise ValueError("All entries in deriv_orders must be integers.")

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
    two_term_i1 = two_term_coeff_solver_low_mem(wvlt, i1)
    two_term_i2 = two_term_coeff_solver_low_mem(wvlt, i2)

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


def four_term_coeff_solver_low_mem(
    wvlt: str, deriv_orders: np.ndarray, residuals: bool = False
) -> np.ndarray:
    """Solves for the two-term connection coefficients of a given wavelet. This is
    a low-memory implementation, i.e., it avoids constructing large intermediate matrices,
    and runs single-core. This comes at the cost of computational speed.

    TODO: Change to sparse matrices to further reduce memory usage.

    Parameters
    ----------
    wvlt : str
        Name of the wavelet (must be recognized by PyWavelets).
    deriv_orders : np.ndarray
        1D, len=3 int array of derivative orders for which to compute the coefficients.
    residuals : bool, optional
        If True, also return the residuals of the refinement and moment equations.

    Returns
    -------
    np.ndarray
        Array of two-term connection coefficients.
    tuple, optional
        If residuals is True, returns a tuple containing the coefficients and the residuals.
    """

    # check that deriv_orders has length 3
    if len(deriv_orders) != 3:
        raise ValueError("deriv_orders must be a 1D array of length 3.")

    # check that deriv_orders are integers
    if not np.issubdtype(deriv_orders.dtype, np.integer):
        raise ValueError("All entries in deriv_orders must be integers.")

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
    i3 = deriv_orders[2]

    # compute three-term coefficients for each derivation order
    three_term_i1i2 = three_term_coeff_solver_low_mem(wvlt, np.array([i1, i2]))
    three_term_i1i3 = three_term_coeff_solver_low_mem(wvlt, np.array([i1, i3]))
    three_term_i2i3 = three_term_coeff_solver_low_mem(wvlt, np.array([i2, i3]))

    def map_indices(i, j, k):
        idx = (
            (i + filt_len - 2) * num_nonzero_coeffs**2
            + (j + filt_len - 2) * num_nonzero_coeffs
            + (k + filt_len - 2)
        )
        return idx

    # populate the block matrix from refinement relations
    four_term_ref = np.zeros((num_nonzero_coeffs**3, num_nonzero_coeffs**3))
    for k1_out in filt_idx:
        for k2_out in filt_idx:
            for k3_out in filt_idx:
                r = map_indices(k1_out, k2_out, k3_out)
                for n0 in range(filt_len):
                    for n1 in range(filt_len):
                        for n2 in range(filt_len):
                            for n3 in range(filt_len):
                                k1_in = 2 * k1_out + n1 - n0
                                k2_in = 2 * k2_out + n2 - n0
                                k3_in = 2 * k3_out + n3 - n0
                                if (
                                    np.abs(k1_in) < filt_len - 1
                                    and np.abs(k2_in) < filt_len - 1
                                    and np.abs(k3_in) < filt_len - 1
                                ):
                                    c = map_indices(k1_in, k2_in, k3_in)
                                    four_term_ref[r, c] += (
                                        filt[n0] * filt[n1] * filt[n2] * filt[n3]
                                    )

    # populate the block matrix from the moment equations
    four_term_mom_mat = np.zeros((3 * num_nonzero_coeffs**2, num_nonzero_coeffs**3))
    four_term_mom_rhs = np.zeros(3 * num_nonzero_coeffs**2)

    # moment equation for i3
    i = 0
    for k1_phys in filt_idx:
        for k2_phys in filt_idx:
            for k3_phys in filt_idx:
                col_idx = map_indices(k1_phys, k2_phys, k3_phys)
                four_term_mom_mat[i, col_idx] = k3_phys**i3
            four_term_mom_rhs[i] = math.factorial(i3) * three_term_i1i2[i]
            i += 1

    # moment equation for i2
    i = 0
    for k1_phys in filt_idx:
        for k3_phys in filt_idx:
            for k2_phys in filt_idx:
                col_idx = map_indices(k1_phys, k2_phys, k3_phys)
                four_term_mom_mat[i + num_nonzero_coeffs**2, col_idx] = k2_phys**i2
            four_term_mom_rhs[i + num_nonzero_coeffs**2] = (
                math.factorial(i2) * three_term_i1i3[i]
            )
            i += 1

    # moment equation for i1
    i = 0
    for k2_phys in filt_idx:
        for k3_phys in filt_idx:
            for k1_phys in filt_idx:
                col_idx = map_indices(k1_phys, k2_phys, k3_phys)
                four_term_mom_mat[i + 2 * num_nonzero_coeffs**2, col_idx] = k1_phys**i1
            four_term_mom_rhs[i + 2 * num_nonzero_coeffs**2] = (
                math.factorial(i1) * three_term_i2i3[i]
            )
            i += 1

    # target eigenvalue
    target_lambda = 1 / 2 ** (i1 + i2 + i3 + 1)

    # construct eigenvalue problem
    four_term_ref_mat = four_term_ref - target_lambda * np.eye(num_nonzero_coeffs**3)
    four_term_ref_rhs = np.zeros(num_nonzero_coeffs**3)

    # build the least squares problem
    lhs = np.vstack((four_term_ref_mat, four_term_mom_mat))
    rhs = np.concatenate((four_term_ref_rhs, four_term_mom_rhs))

    # solve least squares approximation
    four_term_coeff, res, rank, s = np.linalg.lstsq(lhs, rhs, rcond=None)

    if residuals:
        # check residuals for refinement and moment equations
        ref_error = np.linalg.norm(
            np.dot(four_term_ref, four_term_coeff) - target_lambda * four_term_coeff
        )
        mom_error = np.linalg.norm(
            np.dot(four_term_mom_mat, four_term_coeff) - four_term_mom_rhs
        )

        four_term_coeff, ref_error, mom_error

        return four_term_coeff, ref_error, mom_error

    return four_term_coeff


def two_term_coeff_solver(
    wvlt: str, deriv_order: int, residuals: bool = False
) -> np.ndarray:
    """Solves for the two-term connection coefficients of a given wavelet.
    Calculations and H matrix construction is vectorized making use of NumPy's
    intrinsic optimizations via broadcasting.

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

    # check that deriv_order is integer
    if not np.issubdtype(type(deriv_order), np.integer):
        raise ValueError("deriv_order must be an integer.")

    # get filter bank of specified wavelet
    rec_lo = np.asarray(pywt.Wavelet(wvlt).filter_bank[2])

    # remove zeros from the reconstruction low-pass filter
    filt = rec_lo[rec_lo != 0]

    # set constants
    filt_len = len(filt)
    filt_idx = np.arange(-filt_len + 2, filt_len - 1)
    num_nonzero_coeffs = 2 * filt_len - 3

    # define flattening mapping function
    def map_indices(j):
        idx = j + filt_len - 2
        return idx

    # create shift luts
    n = np.arange(filt_len)
    n0, n1 = np.meshgrid(n, n, indexing="ij")
    n0, n1 = n0.ravel(), n1.ravel()

    # calculate filter products for all combinations
    filt_prod = filt[n0] * filt[n1]

    # create filter index luts
    k1_out = filt_idx

    # create k_in luts
    k1_in = np.add(2 * k1_out[:, None], n1 - n0)

    # mask for overlap indices
    limit = filt_len - 1
    mask = np.abs(k1_in) < limit

    # get row and column indices of H matrix
    r = map_indices(k1_out[:, None])
    c = map_indices(k1_in)

    # apply mask and flatten to index list
    r_map = np.broadcast_to(r, c.shape)[mask]
    c_map = c[mask]
    filt_map = np.broadcast_to(filt_prod, c.shape)[mask]

    # populate refinement matrix (np.add.at for handling repeated indices)
    two_term_ref = np.zeros((num_nonzero_coeffs, num_nonzero_coeffs))
    np.add.at(two_term_ref, (r_map, c_map), filt_map)

    # build the least squares problem components of the refinement relation
    two_term_ref_mat = two_term_ref - (1 / 2**deriv_order) * np.eye(num_nonzero_coeffs)
    two_term_ref_rhs = np.zeros(num_nonzero_coeffs)

    # build the least squares problem components of the moment equation
    two_term_mom_lhs = np.empty((1, num_nonzero_coeffs))
    two_term_mom_lhs[0, :] = filt_idx**deriv_order
    two_term_mom_rhs = np.empty(1)
    two_term_mom_rhs[0] = math.factorial(deriv_order)  # * (-1) ** deriv_order
    # The (-1)^deriv_order factor very much depends on the convention used for
    # connection coefficients. Here we follow the convention used in our paper.

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
    Calculations and H matrix construction is vectorized making use of NumPy's
    intrinsic optimizations via broadcasting.

    Parameters
    ----------
    wvlt : str
        Name of the wavelet (must be recognized by PyWavelets).
    deriv_orders : np.ndarray
        1D, len=2 int array of derivative orders for which to compute the coefficients.
    residuals : bool, optional
        If True, also return the residuals of the refinement and moment equations.

    Returns
    -------
    np.ndarray
        Array of two-term connection coefficients.
    tuple, optional
        If residuals is True, returns a tuple containing the coefficients and the residuals.
    """

    # check that deriv_orders has length 2
    if len(deriv_orders) != 2:
        raise ValueError("deriv_orders must be a 1D array of length 2.")

    # check that deriv_orders are integers
    if not np.issubdtype(deriv_orders.dtype, np.integer):
        raise ValueError("All entries in deriv_orders must be integers.")

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

    # define flattening mapping function
    def map_indices(j, k):
        idx = (j + filt_len - 2) * num_nonzero_coeffs + (k + filt_len - 2)
        return idx

    # create shift luts
    n = np.arange(filt_len)
    n0, n1, n2 = np.meshgrid(n, n, n, indexing="ij")
    n0, n1, n2 = n0.ravel(), n1.ravel(), n2.ravel()

    # calculate filter products for all combinations
    filt_prod = filt[n0] * filt[n1] * filt[n2]

    # create filter index luts
    k1_out, k2_out = np.meshgrid(filt_idx, filt_idx, indexing="ij")
    k1_out, k2_out = k1_out.ravel(), k2_out.ravel()

    # create k_in luts
    k1_in = np.add(2 * k1_out[:, None], n1 - n0)
    k2_in = np.add(2 * k2_out[:, None], n2 - n0)

    # mask for overlap indices
    limit = filt_len - 1
    mask = (np.abs(k1_in) < limit) & (np.abs(k2_in) < limit)

    # get row and column indices of H matrix
    r = map_indices(k1_out[:, None], k2_out[:, None])
    c = map_indices(k1_in, k2_in)

    # apply mask and flatten to index list
    r_map = np.broadcast_to(r, c.shape)[mask]
    c_map = c[mask]
    filt_map = np.broadcast_to(filt_prod, c.shape)[mask]

    # populate refinement matrix (np.add.at for handling repeated indices)
    three_term_ref = np.zeros((num_nonzero_coeffs**2, num_nonzero_coeffs**2))
    np.add.at(three_term_ref, (r_map, c_map), filt_map)

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


def four_term_coeff_solver(
    wvlt: str, deriv_orders: np.ndarray, residuals: bool = False
) -> np.ndarray:
    """Solves for the two-term connection coefficients of a given wavelet.
    Calculations and H matrix construction is vectorized making use of NumPy's
    intrinsic optimizations via broadcasting.

    Parameters
    ----------
    wvlt : str
        Name of the wavelet (must be recognized by PyWavelets).
    deriv_orders : np.ndarray
        1D, len=3 int array of derivative orders for which to compute the coefficients.
    residuals : bool, optional
        If True, also return the residuals of the refinement and moment equations.

    Returns
    -------
    np.ndarray
        Array of two-term connection coefficients.
    tuple, optional
        If residuals is True, returns a tuple containing the coefficients and the residuals.
    """

    # check that deriv_orders has length 3
    if len(deriv_orders) != 3:
        raise ValueError("deriv_orders must be a 1D array of length 3.")

    # check that deriv_orders are integers
    if not np.issubdtype(deriv_orders.dtype, np.integer):
        raise ValueError("All entries in deriv_orders must be integers.")

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
    i3 = deriv_orders[2]

    # compute three-term coefficients for each derivation order
    three_term_i1i2 = three_term_coeff_solver(wvlt, np.array([i1, i2]))
    three_term_i1i3 = three_term_coeff_solver(wvlt, np.array([i1, i3]))
    three_term_i2i3 = three_term_coeff_solver(wvlt, np.array([i2, i3]))

    # define flattening index mapping function
    def map_indices(i, j, k):
        idx = (
            (i + filt_len - 2) * num_nonzero_coeffs**2
            + (j + filt_len - 2) * num_nonzero_coeffs
            + (k + filt_len - 2)
        )
        return idx

    # create shift luts
    n = np.arange(filt_len)
    n0, n1, n2, n3 = np.meshgrid(n, n, n, n, indexing="ij")
    n0, n1, n2, n3 = n0.ravel(), n1.ravel(), n2.ravel(), n3.ravel()

    # calculate filter products for all combinations
    filt_prod = filt[n0] * filt[n1] * filt[n2] * filt[n3]

    # create filter index luts
    k1_out, k2_out, k3_out = np.meshgrid(filt_idx, filt_idx, filt_idx, indexing="ij")
    k1_out, k2_out, k3_out = k1_out.ravel(), k2_out.ravel(), k3_out.ravel()

    # create k_in luts
    k1_in = np.add(2 * k1_out[:, None], n1 - n0)
    k2_in = np.add(2 * k2_out[:, None], n2 - n0)
    k3_in = np.add(2 * k3_out[:, None], n3 - n0)

    # mask for overlap indices
    limit = filt_len - 1
    mask = (np.abs(k1_in) < limit) & (np.abs(k2_in) < limit) & (np.abs(k3_in) < limit)

    # get row and column indices of H matrix
    r = map_indices(k1_out[:, None], k2_out[:, None], k3_out[:, None])
    c = map_indices(k1_in, k2_in, k3_in)

    # apply mask and flatten to index list
    r_map = np.broadcast_to(r, c.shape)[mask]
    c_map = c[mask]
    filt_map = np.broadcast_to(filt_prod, c.shape)[mask]

    # populate refinement matrix (np.add.at for handling repeated indices)
    four_term_ref = np.zeros((num_nonzero_coeffs**3, num_nonzero_coeffs**3))
    np.add.at(four_term_ref, (r_map, c_map), filt_map)

    # populate the block matrix from the moment equations
    four_term_mom_mat = np.zeros((3 * num_nonzero_coeffs**2, num_nonzero_coeffs**3))
    four_term_mom_rhs = np.zeros(3 * num_nonzero_coeffs**2)

    # moment equation for i3
    i = 0
    for k1_phys in filt_idx:
        for k2_phys in filt_idx:
            for k3_phys in filt_idx:
                col_idx = map_indices(k1_phys, k2_phys, k3_phys)
                four_term_mom_mat[i, col_idx] = k3_phys**i3
            four_term_mom_rhs[i] = math.factorial(i3) * three_term_i1i2[i]
            i += 1

    # moment equation for i2
    i = 0
    for k1_phys in filt_idx:
        for k3_phys in filt_idx:
            for k2_phys in filt_idx:
                col_idx = map_indices(k1_phys, k2_phys, k3_phys)
                four_term_mom_mat[i + num_nonzero_coeffs**2, col_idx] = k2_phys**i2
            four_term_mom_rhs[i + num_nonzero_coeffs**2] = (
                math.factorial(i2) * three_term_i1i3[i]
            )
            i += 1

    # moment equation for i1
    i = 0
    for k2_phys in filt_idx:
        for k3_phys in filt_idx:
            for k1_phys in filt_idx:
                col_idx = map_indices(k1_phys, k2_phys, k3_phys)
                four_term_mom_mat[i + 2 * num_nonzero_coeffs**2, col_idx] = k1_phys**i1
            four_term_mom_rhs[i + 2 * num_nonzero_coeffs**2] = (
                math.factorial(i1) * three_term_i2i3[i]
            )
            i += 1

    # target eigenvalue
    target_lambda = 1 / 2 ** (i1 + i2 + i3 + 1)

    # construct eigenvalue problem
    four_term_ref_mat = four_term_ref - target_lambda * np.eye(num_nonzero_coeffs**3)
    four_term_ref_rhs = np.zeros(num_nonzero_coeffs**3)

    # build the least squares problem
    lhs = np.vstack((four_term_ref_mat, four_term_mom_mat))
    rhs = np.concatenate((four_term_ref_rhs, four_term_mom_rhs))

    # solve least squares approximation
    four_term_coeff, res, rank, s = np.linalg.lstsq(lhs, rhs, rcond=None)

    if residuals:
        # check residuals for refinement and moment equations
        ref_error = np.linalg.norm(
            np.dot(four_term_ref, four_term_coeff) - target_lambda * four_term_coeff
        )
        mom_error = np.linalg.norm(
            np.dot(four_term_mom_mat, four_term_coeff) - four_term_mom_rhs
        )

        four_term_coeff, ref_error, mom_error

        return four_term_coeff, ref_error, mom_error

    return four_term_coeff
