from typing import Tuple, Optional, Union

import scipy.sparse
import numpy as np


class Toeplitz_convolution2d():
    """
    Convolve a 2D array with a 2D kernel using sparse matrix methods. Two
    computation methods are available:

    * ``'precomputed'``: Builds a sparse double-block Toeplitz matrix during
      initialization and uses it for fast batched convolution via
      matrix-multiply. Ideal when the same kernel will be applied to many inputs
      (large batch sizes, e.g. 1000+).
    * ``'lazy'``: Uses sparse COO broadcasting (instant init, low memory).
      Every call uses broadcasting. Best for sparse inputs (density < ~0.1).

    Generally, both methods are faster than scipy.signal.convolve2d when
    convolving multiple sparse arrays with the same kernel. The ``'precomputed'``
    method amortizes its initialization cost over many calls, while
    ``'lazy'`` is better for one-off convolutions or very large/sparse inputs.
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
            The data type to use for computation.
            If ``None``, then the data type of the kernel is used.
        method (str):
            Either ``'precomputed'`` or ``'lazy'``.

    Args:
        x_shape (Tuple[int, int]):
            The shape of the 2D array to be convolved.
        k (np.ndarray):
            2D kernel to convolve with.
        mode (str):
            Convolution mode, either ``'full'``, ``'same'``, or ``'valid'``.
            See scipy.signal.convolve2d for details. (Default is ``'same'``)
        dtype (Optional[np.dtype]):
            The data type to use for the Toeplitz matrix. Ideally, this matches
            the data type of the input array. If ``None``, then the data type of
            the kernel is used. (Default is ``None``)
        verbose (Union[bool, int]):
            If > 0, prints warnings for large expected Toeplitz matrices.
            (Default is ``False``)
        method (str):
            * ``'precomputed'``: Builds the Toeplitz matrix upfront. Faster for
              large batch sizes but uses more memory and has slower
              initialization. \n
            * ``'lazy'``: Uses sparse COO broadcasting (instant init, low
              memory). Every call uses broadcasting. Best for sparse inputs
              (density < ~0.1). \n
            (Default is ``'lazy'``)

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
        method: str = 'lazy',
    ):
        """
        Initializes the Toeplitz_convolution2d object. If method is
        ``'precomputed'``, builds and stores the Toeplitz matrix.
        """
        ## Type checking
        assert isinstance(x_shape, (tuple, list)), f"x_shape must be a tuple. Found: {type(x_shape)}"
        assert all([isinstance(s, (int, float, np.integer, np.floating)) for s in x_shape]), f"x_shape must be a tuple of integers. Found: {[type(s) for s in x_shape]}"
        x_shape = (int(x_shape[0]), int(x_shape[1]))

        assert isinstance(k, np.ndarray), "k must be a numpy array"
        assert k.ndim == 2, "k must be a 2D array"

        assert isinstance(mode, str), "mode must be a string"
        assert mode in ['full', 'same', 'valid'], "mode must be 'full', 'same', or 'valid'"

        assert isinstance(method, str), "method must be a string"
        assert method in ['precomputed', 'lazy'], "method must be 'precomputed' or 'lazy'"

        if mode == 'valid':
            assert x_shape[0] >= k.shape[0] and x_shape[1] >= k.shape[1], "x must be larger than k in both dimensions for mode='valid'"

        ## Warn if Toeplitz matrix will be very large
        if verbose > 0 and method == 'precomputed':
            n_nz_elements_expected = x_shape[0] * x_shape[1] * k.shape[0] * k.shape[1]
            if n_nz_elements_expected >= 1e8:
                print(
                    "Warning: Expected number of non-zero elements in the Toeplitz matrix is large. \n"
                    f"(x_shape[0]*x_shape[1]*k.shape[0]*k.shape[1]) = {n_nz_elements_expected} non-zero elements. \n"
                    "This will likely be slow and have a large memory footprint. \n"
                    "Consider using method='lazy' or breaking the `x` array into smaller chunks."
                )

        self.k = k.copy()
        self.mode = mode
        self.x_shape = x_shape
        self.method = method
        self.dtype = k.dtype if dtype is None else dtype

        self._dt = None
        self._so = None

        if self.method == 'precomputed':
            self._dt, self._so = self._build_toeplitz_matrix(self.x_shape, self.k, self.dtype)

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
                Input array(s) (i.e. image(s)) to convolve with the kernel. \n
                * If ``batching==False``: Single 2D array to convolve with the
                  kernel. Shape: *(self.x_shape[0], self.x_shape[1])*
                * If ``batching==True``: Multiple 2D arrays that have been
                  flattened into row vectors (with order='C'). \n
                Shape: *(n_arrays, self.x_shape[0]*self.x_shape[1])*

            batching (bool):
                * ``False``: x is a single 2D array.
                * ``True``: x is a 2D array where each row is a flattened 2D
                  array. \n
                (Default is ``True``)

            mode (Optional[str]):
                Defines the mode of the convolution. Options are ``'full'``,
                ``'same'``, or ``'valid'``. See ``scipy.signal.convolve2d`` for
                details. Overrides the mode set in __init__. (Default is
                ``None``, which uses the mode from __init__)

        Returns:
            (Union[np.ndarray, scipy.sparse.csr_matrix]):
                out (Union[np.ndarray, scipy.sparse.csr_matrix]):
                    * ``batching==True``: Multiple convolved 2D arrays that have
                      been flattened into row vectors (with order='C'). Shape:
                      *(n_arrays, height*width)*
                    * ``batching==False``: Single convolved 2D array of shape
                      *(height, width)*
        """
        if mode is None:
            mode = self.mode

        issparse = scipy.sparse.issparse(x)

        ## Validate input dimensions for batching mode
        if batching:
            expected_dim = self.x_shape[0] * self.x_shape[1]
            assert x.shape[1] == expected_dim, (
                f"When batching=True, x.shape[1] ({x.shape[1]}) must equal "
                f"x_shape[0]*x_shape[1] ({expected_dim})"
            )

        ## Precomputed path: use pre-built Toeplitz matrix
        if self._dt is not None:
            return self._compute_toeplitz(x=x, mode=mode, batching=batching, issparse=issparse)

        ## Lazy path: use COO broadcasting
        out = self._compute_broadcasting(x=x, mode=mode, batching=batching)
        out.sum_duplicates()
        if not issparse:
            out = out.toarray()
        return out

    def _compute_toeplitz(self, x, mode, batching, issparse):
        """
        Compute convolution using the pre-built double-block Toeplitz matrix.
        """
        if batching:
            x_v = x.T  ## transpose into column vectors
        else:
            x_v = x.reshape(-1, 1)  ## reshape 2D array into a column vector

        if issparse:
            x_v = x_v.tocsc()

        out_v = self._dt @ x_v  ## if sparse, then out_v will be a csc matrix

        ## Compute crop indices based on convolution mode
        so = self._so
        k_shape = self.k.shape
        if mode == 'full':
            t = 0
            b = so[0] + 1
            l = 0
            r = so[1] + 1
        if mode == 'same':
            t = (k_shape[0] - 1) // 2
            b = -(k_shape[0] - 1) // 2
            l = (k_shape[1] - 1) // 2
            r = -(k_shape[1] - 1) // 2
            b = self.x_shape[0] + 1 if b == 0 else b
            r = self.x_shape[1] + 1 if r == 0 else r
        if mode == 'valid':
            t = (k_shape[0] - 1)
            b = -(k_shape[0] - 1)
            l = (k_shape[1] - 1)
            r = -(k_shape[1] - 1)
            b = self.x_shape[0] + 1 if b == 0 else b
            r = self.x_shape[1] + 1 if r == 0 else r

        ## Crop the output to the correct size
        if batching:
            idx_crop = np.zeros(so, dtype=np.bool_)
            idx_crop[t:b, l:r] = True
            idx_crop = idx_crop.reshape(-1)
            out = out_v[idx_crop, :].T
        else:
            if issparse:
                out = out_v.reshape(so).tocsc()[t:b, l:r]
            else:
                out = out_v.reshape(so)[t:b, l:r]
        return out

    def _compute_broadcasting(self, x, mode, batching):
        """
        Compute convolution using sparse COO broadcasting. For each nonzero
        element in x and each nonzero kernel element, directly compute the
        output contribution via scatter-add into a sparse output matrix.

        This avoids building the full Toeplitz matrix, making initialization
        instant and memory usage proportional to nnz(x) * nnz(k).
        """
        k = self.k
        x_shape = self.x_shape
        dtype = self.dtype

        ## Compute output dimensions and spatial offsets based on mode
        if mode == 'full':
            H_out = x_shape[0] + k.shape[0] - 1
            W_out = x_shape[1] + k.shape[1] - 1
            t = l = 0
        elif mode == 'same':
            H_out = x_shape[0]
            W_out = x_shape[1]
            t = (k.shape[0] - 1) // 2
            l = (k.shape[1] - 1) // 2
        elif mode == 'valid':
            H_out = x_shape[0] - k.shape[0] + 1
            W_out = x_shape[1] - k.shape[1] + 1
            t = k.shape[0] - 1
            l = k.shape[1] - 1

        ## Get kernel nonzero elements in COO format
        k_coo = scipy.sparse.coo_matrix(k)
        if k_coo.dtype != dtype:
            k_coo = k_coo.astype(dtype)
        k_r, k_c, k_d = k_coo.row, k_coo.col, k_coo.data  ## shape: (n_k_nnz,) each

        ## Get input nonzero elements in COO format
        x_coo = scipy.sparse.coo_matrix(x)
        if batching:
            n_batch = x.shape[0]
            batch_idx = x_coo.row               ## batch index per nonzero element
            x_flat_idx = x_coo.col              ## flattened spatial index
            x_r = x_flat_idx // x_shape[1]      ## 2D row
            x_c = x_flat_idx % x_shape[1]       ## 2D col
        else:
            batch_idx = np.zeros_like(x_coo.row)
            x_r = x_coo.row
            x_c = x_coo.col

        ## Broadcast: outer product of input nonzeros with kernel nonzeros
        ## Convolution: output[xr + kr, xc + kc] += x_val * k_val
        ## Offsets (t, l) shift coordinates for 'same' and 'valid' modes
        out_r = (x_r[:, None] + k_r[None, :] - t).ravel()  ## shape: (n_x_nnz * n_k_nnz,)
        out_c = (x_c[:, None] + k_c[None, :] - l).ravel()
        out_b = np.repeat(batch_idx[:, None], len(k_d), axis=1).ravel()
        out_d = (x_coo.data[:, None] * k_d[None, :]).ravel()

        ## Filter out-of-bounds contributions
        valid = (out_r >= 0) & (out_r < H_out) & (out_c >= 0) & (out_c < W_out)
        out_r, out_c, out_b, out_d = out_r[valid], out_c[valid], out_b[valid], out_d[valid]

        ## Assemble sparse output via scatter-add (duplicate indices are summed)
        if batching:
            out = scipy.sparse.csr_matrix(
                (out_d, (out_b, out_r * W_out + out_c)),
                shape=(n_batch, H_out * W_out),
                dtype=dtype,
            )
        else:
            out = scipy.sparse.csr_matrix(
                (out_d, (out_r, out_c)),
                shape=(H_out, W_out),
                dtype=dtype,
            )
        return out

    @staticmethod
    def _build_toeplitz_matrix(x_shape, k, dtype):
        """
        Build the double-block Toeplitz matrix for 2D convolution. The matrix
        encodes the full convolution operation as a single sparse
        matrix-multiply.

        Args:
            x_shape (Tuple[int, int]):
                Spatial dimensions (height, width) of the input.
            k (np.ndarray):
                2D convolution kernel.
            dtype (np.dtype):
                Data type for the sparse matrix entries.

        Returns:
            (Tuple[scipy.sparse.csr_matrix, Tuple[int, int]]):
                dt (scipy.sparse.csr_matrix):
                    The double-block Toeplitz matrix.
                so (Tuple[int, int]):
                    The full output shape (before mode-based cropping).
        """
        so = (k.shape[0] + x_shape[0] - 1, k.shape[1] + x_shape[1] - 1)

        ## Build row-Toeplitz blocks for each kernel row
        t = [scipy.sparse.diags(
            diagonals=np.ones((k.shape[1], x_shape[1]), dtype=dtype) * k_i[::-1][:, None],
            offsets=np.arange(-k.shape[1] + 1, 1),
            shape=(so[1], x_shape[1]),
            dtype=dtype,
        ) for k_i in k]

        ## Stack blocks vertically with zero padding for the input rows
        tc = scipy.sparse.vstack(
            t + [scipy.sparse.dia_matrix((t[0].shape), dtype=dtype)] * (x_shape[0] - 1)
        )

        ## Assemble the doubly-blocked Toeplitz matrix by shifting block columns
        dt = scipy.sparse.hstack([
            Toeplitz_convolution2d._roll_sparse(
                x=tc,
                shift=(ii > 0) * ii * so[1],
            ) for ii in range(x_shape[0])
        ]).tocsr()

        return dt, so

    @staticmethod
    def _roll_sparse(x, shift):
        """
        Roll rows of a sparse COO matrix down by ``shift`` positions.
        """
        out = x.copy()
        out.row += shift
        return out
