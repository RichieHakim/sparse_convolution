"""
Sparse 2D convolution via Toeplitz matrix methods.

Provides ``Toeplitz_convolution2d``, a class for convolving sparse 2D arrays
with a dense 2D kernel. Four computation methods are available, each with
multiple backends:

- ``'precomputed'`` + (``'numpy'``, ``'numba'``, ``'torch'``)
- ``'lazy'`` + (``'numpy'``, ``'torch'``)
- ``'gather_scatter'`` + (``'numpy'``, ``'numba'``, ``'torch'``)
- ``'direct'`` + (``'numba'``,)

RH 2022
"""

from typing import Tuple, Optional, Union

import numpy as np
import scipy.sparse

from sparse_convolution._precomputed import (
    build_toeplitz_scipy,
    build_toeplitz_torch,
    compute_precomputed_scipy,
    compute_precomputed_numba,
    compute_precomputed_torch,
)
from sparse_convolution._lazy import (
    compute_lazy_numpy,
    compute_lazy_torch,
)
from sparse_convolution._gather_scatter import (
    compute_gather_scatter,
    HAS_NUMBA,
    HAS_TORCH,
)
from sparse_convolution._direct import compute_direct

## Valid (method, backend) combinations
VALID_BACKENDS = {
    'precomputed': ('numpy', 'numba', 'torch'),
    'lazy': ('numpy', 'torch'),
    'gather_scatter': ('numpy', 'numba', 'torch'),
    'direct': ('numba',),
}


class Toeplitz_convolution2d():
    """
    Convolve a 2D array with a 2D kernel using sparse matrix methods. Four
    computation methods are available, each with selectable backends:

    **Methods:**

    * ``'precomputed'``: Builds a sparse double-block Toeplitz matrix during
      initialization and uses it for fast batched convolution via
      matrix-multiply. Ideal when the same kernel will be applied to many
      inputs (large batch sizes, e.g. 1000+).
    * ``'lazy'``: Uses sparse COO broadcasting (instant init, low memory).
      Every call recomputes from scratch. Best for sparse inputs (density
      < ~0.1) with small batch sizes.
    * ``'gather_scatter'``: Uses spconv-style per-kernel-position
      gather-scatter with a chunked dense accumulator. Instant init, bounded
      memory, and avoids the O(n log n) COO-to-CSR sort that makes
      ``'lazy'`` slow on large batches. Best general-purpose method for
      sparse inputs.
    * ``'direct'``: Two-pass batch-parallel scatter using thread-local dense
      buffers (numba only). Each thread scatters into its own L2-cache-sized
      buffer and extracts CSR directly — zero initialization overhead, no
      global dense buffer. 5-17x faster than ``'precomputed'`` at large
      batch sizes (1000+). Best method for large batches of sparse inputs.

    **Backends:**

    * ``'numpy'``: Uses numpy/scipy operations. Available for
      ``'precomputed'``, ``'lazy'``, and ``'gather_scatter'``.
    * ``'numba'``: Numba JIT-compiled parallel loops. Available for
      ``'precomputed'``, ``'gather_scatter'``, and ``'direct'``. Best CPU
      performance after initial JIT warmup (~35ms first call).
    * ``'torch'``: PyTorch operations with optional GPU acceleration.
      Available for ``'precomputed'``, ``'lazy'``, and ``'gather_scatter'``.
      Automatically uses CUDA if available (or as specified via ``device``).

    Generally, all methods are faster than ``scipy.signal.convolve2d`` when
    convolving multiple sparse arrays with the same kernel.

    RH 2022

    Args:
        x_shape (Tuple[int, int]):
            The shape of the 2D array to be convolved.
        k (np.ndarray):
            2D kernel to convolve with.
        mode (str):
            Convolution mode, either ``'full'``, ``'same'``, or ``'valid'``.
            See ``scipy.signal.convolve2d`` for details.
        dtype (Optional[np.dtype]):
            Data type for computation. If ``None``, uses the kernel's dtype.
        verbose (Union[bool, int]):
            If > 0, prints warnings for large expected Toeplitz matrices.
        method (str):
            Computation method: ``'precomputed'``, ``'lazy'``,
            ``'gather_scatter'``, or ``'direct'``.
        backend (Optional[str]):
            Implementation backend. Valid options depend on ``method``: \\n
            * ``'precomputed'``: ``'numpy'`` (scipy matmul), ``'numba'``
              (parallel CSR matvec), or ``'torch'``
            * ``'lazy'``: ``'numpy'`` (scipy COO) or ``'torch'``
            * ``'gather_scatter'``: ``'numpy'``, ``'numba'``, or ``'torch'``
            * ``'direct'``: ``'numba'`` (only option)
            \\n
            If ``None``, auto-selects the best available backend:
            ``'numba'`` for ``'direct'`` (requires numba), ``'numba'`` for
            ``'gather_scatter'`` when numba is installed, and ``'numpy'``
            otherwise.
        max_buffer_bytes (int):
            Maximum memory (bytes) for the dense accumulator buffer used by
            ``'gather_scatter'``. Controls chunk size for batch processing.
            Ignored by other methods.
        device (Optional[str]):
            Torch device for ``backend='torch'``. E.g. ``'cpu'``,
            ``'cuda'``, ``'cuda:0'``. If ``None``, auto-selects CUDA if
            available. Ignored for non-torch backends.

    Example:
        .. highlight:: python
        .. code-block:: python

            conv = Toeplitz_convolution2d(
                x_shape=(100, 30),
                k=np.random.rand(10, 10),
                mode='same',
                method='direct',
            )
            out = conv(
                x=scipy.sparse.csr_matrix(np.random.rand(5, 3000)),
                batching=True,
            )
    """

    def __init__(
        self,
        x_shape: Tuple[int, int],
        k: np.ndarray,
        mode: str = 'same',
        dtype: Optional[np.dtype] = None,
        verbose: Union[bool, int] = False,
        method: str = 'direct',
        max_buffer_bytes: int = 256 * 1024 * 1024,
        backend: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the convolution object. If ``method='precomputed'``,
        builds and stores the Toeplitz matrix immediately.
        """
        ## Validate x_shape
        assert isinstance(x_shape, (tuple, list)), \
            f"x_shape must be a tuple. Found: {type(x_shape)}"
        assert all(isinstance(s, (int, float, np.integer, np.floating)) for s in x_shape), \
            f"x_shape must be a tuple of integers. Found: {[type(s) for s in x_shape]}"
        x_shape = (int(x_shape[0]), int(x_shape[1]))

        ## Validate kernel
        assert isinstance(k, np.ndarray), "k must be a numpy array"
        assert k.ndim == 2, "k must be a 2D array"

        ## Validate mode
        assert isinstance(mode, str), "mode must be a string"
        assert mode in ('full', 'same', 'valid'), \
            "mode must be 'full', 'same', or 'valid'"

        ## Validate method
        assert isinstance(method, str), "method must be a string"
        assert method in VALID_BACKENDS, \
            f"method must be one of {list(VALID_BACKENDS.keys())}. Got: {method!r}"

        if mode == 'valid':
            assert x_shape[0] >= k.shape[0] and x_shape[1] >= k.shape[1], \
                "x must be larger than k in both dimensions for mode='valid'"

        ## Resolve backend
        if backend is None:
            if method == 'direct':
                assert HAS_NUMBA, (
                    "method='direct' requires numba. Install numba or use "
                    "method='gather_scatter' / method='precomputed'."
                )
                backend = 'numba'
            elif method == 'gather_scatter':
                backend = 'numba' if HAS_NUMBA else 'numpy'
            else:
                backend = 'numpy'
        assert backend in VALID_BACKENDS[method], (
            f"backend={backend!r} is not valid for method={method!r}. "
            f"Valid backends: {VALID_BACKENDS[method]}"
        )
        if backend == 'numba':
            assert HAS_NUMBA, "backend='numba' requires numba to be installed"
        if backend == 'torch':
            assert HAS_TORCH, "backend='torch' requires torch to be installed"

        ## Resolve device for torch
        if device is None and backend == 'torch':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ## Warn or fail if Toeplitz matrix will be very large
        if method == 'precomputed':
            n_nz_expected = x_shape[0] * x_shape[1] * k.shape[0] * k.shape[1]
            if n_nz_expected >= 1e9:
                raise ValueError(
                    f"Expected number of non-zero elements in the Toeplitz matrix "
                    f"is extremely large ({n_nz_expected} non-zero elements).\n"
                    f"This would likely cause a silent Out-Of-Memory (OOM) crash.\n"
                    f"Please use a memory-efficient method like method='direct' or "
                    f"method='gather_scatter' instead."
                )
            elif n_nz_expected >= 1e7:
                import warnings
                warnings.warn(
                    "Expected number of non-zero elements in the "
                    "Toeplitz matrix is large.\n"
                    f"(x_shape[0]*x_shape[1]*k.shape[0]*k.shape[1]) = "
                    f"{n_nz_expected} non-zero elements.\n"
                    "This will likely be slow and have a large memory "
                    "footprint.\n"
                    "Consider using method='direct' or 'gather_scatter' instead.",
                    UserWarning,
                    stacklevel=2,
                )


        ## Store parameters
        self.k = k.copy()
        self.mode = mode
        self.x_shape = x_shape
        self.method = method
        self.dtype = k.dtype if dtype is None else dtype
        self.max_buffer_bytes = max_buffer_bytes
        self.backend = backend
        self.device = device

        ## Pre-build Toeplitz matrix for 'precomputed' method
        self._dt = None
        self._dt_torch = None
        self._so = None

        if self.method == 'precomputed':
            if self.backend == 'torch':
                self._dt_torch, self._so = build_toeplitz_torch(
                    x_shape=self.x_shape, k=self.k,
                    dtype=self.dtype, device=self.device,
                )
            else:
                self._dt, self._so = build_toeplitz_scipy(
                    x_shape=self.x_shape, k=self.k, dtype=self.dtype,
                )

    def __call__(
        self,
        x: Union[np.ndarray, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix],
        batching: bool = True,
        mode: Optional[str] = None,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """
        Convolve the input array with the kernel.

        Args:
            x (Union[np.ndarray, scipy.sparse.csc_matrix,
            scipy.sparse.csr_matrix]):
                Input array(s) to convolve with the kernel. \\n
                * If ``batching==False``: Single 2D array of shape
                  ``(x_shape[0], x_shape[1])``.
                * If ``batching==True``: Multiple flattened 2D arrays.
                  Shape: ``(n_arrays, x_shape[0] * x_shape[1])``.

            batching (bool):
                * ``False``: ``x`` is a single 2D array.
                * ``True``: ``x`` contains multiple flattened 2D arrays.

            mode (Optional[str]):
                Override the convolution mode set in ``__init__``.

        Returns:
            (Union[np.ndarray, scipy.sparse.csr_matrix]):
                out (Union[np.ndarray, scipy.sparse.csr_matrix]):
                    * ``batching==True``: Flattened output rows.
                      Shape: ``(n_arrays, H_out * W_out)``.
                    * ``batching==False``: 2D output array.
                      Shape: ``(H_out, W_out)``.
        """
        if mode is None:
            mode = self.mode

        assert mode in ('full', 'same', 'valid'), \
            f"mode must be 'full', 'same', or 'valid'. Got: {mode!r}"
        if mode == 'valid':
            assert self.x_shape[0] >= self.k.shape[0] and \
                   self.x_shape[1] >= self.k.shape[1], \
                "x must be larger than k in both dimensions for mode='valid'"

        issparse = scipy.sparse.issparse(x)

        ## Validate input dimensions
        if batching:
            expected_dim = self.x_shape[0] * self.x_shape[1]
            assert x.shape[1] == expected_dim, (
                f"When batching=True, x.shape[1] ({x.shape[1]}) must equal "
                f"x_shape[0]*x_shape[1] ({expected_dim})"
            )
        else:
            assert x.shape == (self.x_shape[0], self.x_shape[1]), (
                f"When batching=False, x.shape ({x.shape}) must equal "
                f"x_shape ({self.x_shape})"
            )

        ## Route to the appropriate method + backend
        if self.method == 'precomputed':
            out = self._call_precomputed(x, mode, batching, issparse)
        elif self.method == 'gather_scatter':
            out = self._call_gather_scatter(x, mode, batching)
        elif self.method == 'direct':
            out = self._call_direct(x, mode, batching)
        elif self.method == 'lazy':
            out = self._call_lazy(x, mode, batching)

        ## Ensure output format matches input format
        if not issparse:
            if scipy.sparse.issparse(out):
                out = out.toarray()
        else:
            if not scipy.sparse.issparse(out):
                out = scipy.sparse.csr_matrix(out)

        return out

    def _call_precomputed(self, x, mode, batching, issparse):
        """Route to the appropriate precomputed backend."""
        if self.backend == 'torch':
            return compute_precomputed_torch(
                dt_torch=self._dt_torch, so=self._so,
                x=x, k_shape=self.k.shape, x_shape=self.x_shape,
                mode=mode, batching=batching, issparse=issparse,
                device=self.device,
            )
        elif self.backend == 'numba':
            return compute_precomputed_numba(
                dt=self._dt, so=self._so,
                x=x, k_shape=self.k.shape, x_shape=self.x_shape,
                mode=mode, batching=batching, issparse=issparse,
            )
        else:
            return compute_precomputed_scipy(
                dt=self._dt, so=self._so,
                x=x, k_shape=self.k.shape, x_shape=self.x_shape,
                mode=mode, batching=batching, issparse=issparse,
            )

    def _call_lazy(self, x, mode, batching):
        """Route to the appropriate lazy backend."""
        if self.backend == 'torch':
            out = compute_lazy_torch(
                x=x, k=self.k, x_shape=self.x_shape,
                mode=mode, batching=batching, dtype=self.dtype,
                device=self.device,
            )
        else:
            out = compute_lazy_numpy(
                x=x, k=self.k, x_shape=self.x_shape,
                mode=mode, batching=batching, dtype=self.dtype,
            )
            out.sum_duplicates()
        return out

    def _call_gather_scatter(self, x, mode, batching):
        """Route to the gather_scatter dispatch."""
        return compute_gather_scatter(
            x=x, k=self.k, x_shape=self.x_shape,
            mode=mode, batching=batching, dtype=self.dtype,
            backend=self.backend, max_buffer_bytes=self.max_buffer_bytes,
            device=self.device,
        )

    def _call_direct(self, x, mode, batching):
        """Route to the direct CSR scatter dispatch (numba only)."""
        return compute_direct(
            x=x, k=self.k, x_shape=self.x_shape,
            mode=mode, batching=batching, dtype=self.dtype,
        )
