[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/JeppeDruedahl/vfi/master)

# Value Function Iteration with Numba JIT

This package provides a value function iteration algorithm written purely in Python, but optimized with Numba JIT (including parallization).

* The **vfi.linear_interp** module provides a **Numba JIT** compilled **interpolator class** for **linear interpolation** (and extrapolation) of **multiple functions** in **n-dimensions** (showcased in "Fast Linear Interpolation.ipynb")

* The **vfi.optimizer_1d** module provides a **Numba JIT** compilled one-dimensional **optimizer function** (using *golden secant search*) for a user-defined Numba JIT compilled function with abirtrary number of fixed inputs (showcased in "Fast Optimization 1D.pynb").

The overall value function iteration algorithm is showcased in **"Fast VFI.ipynb"**.

# Troubleshooting

If all @numba.njit, @numba.prange and @numba.jitclass decorators are removed, the code runs as pure Python. 
