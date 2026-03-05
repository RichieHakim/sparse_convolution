from typing import Tuple, Optional, Union

import numpy as np
import scipy.sparse

from ._toeplitz_engine import build_toeplitz_matrix, compute_toeplitz
from ._broadcasting_engine import compute_broadcasting

__version__ = '0.1.5'

class Toeplitz_convolution2d:
    """
    Convolve a 2D array with a 2D kernel using the Toeplitz matrix
    multiplication method. This class is ideal when 'x' is very sparse
    (density<0.01), 'x' is small (shape <(1000,1000)), 'k' is small (shape
    <(100,100)), and the batch size is large (e.g. 1000+). Generally, it is
    faster than scipy.signal.convolve2d when convolving multiple arrays with the
    same kernel. It maintains a low memory footprint by storing the toeplitz
    matrix as a sparse matrix.
    RH 2022

    Attributes:
        x_shape (Tuple[int, int]):
            The shape of the 2D array to be convolved.
        k (np.ndarray):
            2D kernel to convolve with.
        mode (str):
            Either ``'full'``, ``'same'``, or ``'valid'``. See
            scipy.signal.convolve2d for details.
        dtype (Optional[np.dtype]):
            The data type to use for the Toeplitz matrix.
            If ``None``, then the data type of the kernel is used.

    Args:
        x_shape (Tuple[int, int]):
            The shape of the 2D array to be convolved.
        k (np.ndarray):
            2D kernel to convolve with.
        mode (str):
            Convolution method to use, either ``'full'``, ``'same'``, or
            ``'valid'``.
            See scipy.signal.convolve2d for details. (Default is 'same')
        dtype (Optional[np.dtype]):
            The data type to use for the Toeplitz matrix. Ideally, this matches
            the data type of the input array. If ``None``, then the data type of
            the kernel is used. (Default is ``None``)

    Example:
        .. highlight:: python
        .. code-block:: python

            # create Toeplitz_convolution2d object
            toeplitz_convolution2d = Toeplitz_convolution2d(
                x_shape=(100,30),
                k=np.random.rand(10,10),
                mode='same',
            )
            toeplitz_convolution2d(
                x=scipy.sparse.csr_matrix(np.random.rand(5,3000)),
                batch_size=True,
            )
    """
    def __init__(
        self,
        x_shape: Tuple[int, int],
        k: np.ndarray,
        mode: str = 'same',
        dtype: Optional[np.dtype] = None,
        verbose: Union[bool, int] = False,
        method: str = 'lazy',
    ):
        """
        Initializes the Toeplitz_convolution2d object.
        """
        ## Type checking
        assert isinstance(x_shape, (tuple, list)), f"x_shape must be a tuple. Found: {type(x_shape)}"
        assert all([isinstance(s, (int, float, np.integer, np.floating)) for s in x_shape]), f"x_shape must be a tuple of integers. Found: {[type(s) for s in x_shape]}"
        x_shape = (int(x_shape[0]), int(x_shape[1]))

        assert isinstance(k, np.ndarray), "k must be a numpy array"
        assert k.ndim == 2, "k must be a 2D array"

        assert isinstance(mode, str), "mode must be a string"
        assert mode in ['full', 'same', 'valid'], "mode must be 'full', 'same', or 'valid'"

        self.k = k.copy()
        self.mode = mode
        self.x_shape = x_shape
        self.dtype = k.dtype if dtype is None else dtype

        if mode == 'valid':
            assert x_shape[0] >= k.shape[0] and x_shape[1] >= k.shape[1], "x must be larger than k in both dimensions for mode='valid'"
        
        self.method = method
        assert method in ['lazy', 'precomputed'], "method must be 'lazy' or 'precomputed'"
        
        if self.method == 'precomputed':
            self.dt, self.so = build_toeplitz_matrix(self.x_shape, self.k, self.dtype)
    
    def __call__(
        self,
        x: Union[np.ndarray, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix],
        batching: bool = True,
        mode: Optional[str] = None,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """
        Convolve the input array with the kernel.
        """
        if mode is None:
            mode = self.mode  ## use the mode that was set in the init if not specified
        
        issparse = scipy.sparse.issparse(x)

        if batching:
            expected_dim = self.x_shape[0] * self.x_shape[1]
            if x.shape[1] != expected_dim:
                raise ValueError(f"When batching=True, x.shape[1] ({x.shape[1]}) must match x_shape[0]*x_shape[1] ({expected_dim})")

        if getattr(x, 'dtype', type(x)) != self.dtype:
            x = x.astype(self.dtype)

        if self.method == 'precomputed':
            out = compute_toeplitz(
                x=x, dt=self.dt, so=self.so, k_shape=self.k.shape,
                x_shape=self.x_shape, mode=mode, batching=batching, issparse=issparse
            )
        else:
            out = compute_broadcasting(
                x=x, k=self.k, x_shape=self.x_shape, mode=mode,
                batching=batching, dtype=self.dtype
            )

        if hasattr(out, 'sum_duplicates'):
            out.sum_duplicates()

        # Ensure return type matches dense arrays if input is a dense array
        if not issparse:
            out = out.toarray()

        return out