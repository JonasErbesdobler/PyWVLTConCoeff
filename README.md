# PyWVLTConCoeff
[![DOI](https://zenodo.org/badge/1112012796.svg)](https://zenodo.org/badge/latestdoi/1112012796)

A Python package for calculating two-, three-, and four-term wavelet connection coefficients on an infinite domain.

Connection coefficients are defined as integrals of products of shifted scaling functions $\phi$ and their derivatives, e.g., the four-term coefficient

$$
\Lambda^{(i_1),(i_2),(i_3)}_{k_1,k_2,k_3} = \int \phi(t)\,\frac{\partial^{i_1} \phi(t-k_1)}{\partial t^{i_1}}\,\frac{\partial^{i_2} \phi(t-k_2)}{\partial t^{i_2}}\,\frac{\partial^{i_3} \phi(t-k_3)}{\partial t^{i_3}}\,dt .
$$

They allow linear and non-linear operators to be evaluated directly in wavelet space, for example in Wavelet-Galerkin methods or wavelet-based optical flow. Instead of solving the usual eigenvalue problem, this package stacks the refinement relations and the moment constraints into an overdetermined linear system and solves it by singular value decomposition. The coefficients of higher term count are computed recursively from the lower ones.

Any wavelet included in [PyWavelets](https://pywavelets.readthedocs.io) can be used.

## Installation

**Using `python3-venv`**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e . # to install in interactive mode
```
Later, you just need to type `source venv/bin/activate` in the terminal to activate the environment.
If you want to use the code inside Jupyter Notebooks, you may have to run
```bash
pip install ipykernel
```
in the terminal afterward as well.

## Usage
```python
from PyWVLTConCoeff import (
    two_term_coeff_solver,
    three_term_coeff_solver,
    four_term_coeff_solver,
)

wvlt = "db3"  # Daubechies wavelet with filter length N = 6

# two-term coefficients for the first derivative, shifts k = -(N-2), ..., N-2
gamma = two_term_coeff_solver(wvlt, 1)

# three-term coefficients for derivative orders (i1, i2) = (0, 1)
omega = three_term_coeff_solver(wvlt, [0, 1])

# four-term coefficients for derivative orders (i1, i2, i3) = (0, 1, 1)
lam = four_term_coeff_solver(wvlt, [0, 1, 1])
```
The coefficients are returned as flattened arrays in row-major order. With $N$ being the filter length, each shift runs over $2N-3$ values from $-(N-2)$ to $N-2$, so a single coefficient is accessed as
```python
N = 6
lam = lam.reshape(2 * N - 3, 2 * N - 3, 2 * N - 3)
k1, k2, k3 = 0, -1, 0
value = lam[k1 + N - 2, k2 + N - 2, k3 + N - 2]  # -1.9145852962
```
Passing `residuals=True` additionally returns the $L_2$-norm errors in the refinement relations and the moment constraints:
```python
lam, ref_error, mom_error = four_term_coeff_solver(wvlt, [0, 1, 1], residuals=True)
```
A `RuntimeWarning` is raised if the linear system is rank deficient or cannot be fulfilled, in which case the returned values are not connection coefficients.

### Valid derivative orders
The integral above only exists if the scaling function is smooth enough. For a scaling function with $m$ continuous derivatives and a wavelet with $M$ vanishing moments, this is guaranteed for all derivative orders up to $\min(m, M-1)$. The Daubechies $L = 6$ wavelet (`db3`) has $m = 1$ and $M = 3$, so all coefficients with derivative orders up to one are integrals; a single second derivative is covered as well if the other orders are zero, by integration by parts.

For higher orders, the solvers may still return a unique solution of the refinement relations and moment constraints without a warning, e.g., for `db3` with orders $(2, 2, 2)$. Such values extend the definition of the connection coefficients, but the corresponding integral does not exist in the classical sense, and their accuracy decreases with the derivative order. Check that this is what your application needs before using them.

Each solver also has a `_low_mem` variant, e.g., `four_term_coeff_solver_low_mem`, which gives the same result with nested loops that follow the underlying equations closely, at the cost of computational speed.

### Notebooks
- [`notebooks/examples.ipynb`](notebooks/examples.ipynb): introduction to the solvers.
- [`notebooks/paper_figures.ipynb`](notebooks/paper_figures.ipynb): recreates the residual error plots (written to `figures/`).
- [`notebooks/paper_tables.ipynb`](notebooks/paper_tables.ipynb): recreates the tabulated four-term coefficients of the Daubechies $L = 6$ wavelet (written to `data/`).
- [`notebooks/graphical_abstract.ipynb`](notebooks/graphical_abstract.ipynb): recreates the plots of the graphical abstract (written to `figures/`).

The last two notebooks expect the `data/` and `figures/` directories to exist in the repository root.

## Citation
If you use this package in your work, please cite it using the metadata in [`CITATION.cff`](CITATION.cff), or via the "Cite this repository" button on GitHub.

## Contribution
If you wish to contribute, install `pre-commit` first, via the following
```bash
pre-commit install
```
Then, before committing to the repository, run
```bash
pre-commit run -a
```
every time to ensure correct formatting.

## Testing
The tests are run with `pytest` from the repository root:
```bash
pytest
```

## License
This project is licensed under the MIT License, see [`LICENSE`](LICENSE).
