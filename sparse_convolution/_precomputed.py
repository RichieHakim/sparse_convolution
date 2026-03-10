"""
Precomputed Toeplitz matrix convolution backends.

Builds a sparse double-block Toeplitz matrix at initialization time, then
applies convolution via sparse matrix-vector multiply. Three backends:

- **numpy** (scipy): CSR matmul via scipy's optimized C implementation.
  Supports both sparse and dense input/output.
- **numba**: Parallel CSR matvec via numba prange over rows. 2-4x faster
  than scipy for batched inputs (batch >= 2). Uses the same scipy-built
  CSR matrix.
- **torch**: Converts the Toeplitz matrix to a torch sparse CSR tensor for
  matmul via ``torch.sparse.mm``. Supports CPU and GPU.
"""

import numpy as np
import scipy.sparse

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


## ---------------------------------------------------------------------------
## Toeplitz matrix building (shared by all backends)
## ---------------------------------------------------------------------------

def build_toeplitz_scipy(x_shape, k, dtype):
    """
    Build the double-block Toeplitz matrix for 2D convolution as a scipy
    sparse CSR matrix.

    The Toeplitz matrix encodes the full convolution operation as a single
    sparse matrix-multiply: ``y = T @ x.ravel()``, where ``y`` has the
    shape of a ``'full'`` convolution output (flattened).

    Args:
        x_shape (Tuple[int, int]):
            Spatial dimensions ``(H, W)`` of the input.
        k (np.ndarray):
            2D convolution kernel.
        dtype (np.dtype):
            Data type for the sparse matrix entries.

    Returns:
        (Tuple[scipy.sparse.csr_matrix, Tuple[int, int]]):
            dt (scipy.sparse.csr_matrix):
                The double-block Toeplitz matrix.
            so (Tuple[int, int]):
                The full output shape ``(H_full, W_full)`` before mode-based
                cropping.
    """
    so = (k.shape[0] + x_shape[0] - 1, k.shape[1] + x_shape[1] - 1)

    ## Build row-Toeplitz blocks: one per kernel row. Each block is a banded
    ## matrix of shape (W_full, W_in) with the reversed kernel row on the
    ## diagonals.
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
        _roll_sparse(
            x=tc,
            shift=(ii > 0) * ii * so[1],
        ) for ii in range(x_shape[0])
    ]).tocsr()

    return dt, so


def _roll_sparse(x, shift):
    """
    Roll rows of a sparse COO matrix down by ``shift`` positions.
    """
    out = x.copy()
    out.row += shift
    return out


## ---------------------------------------------------------------------------
## Cropping utilities
## ---------------------------------------------------------------------------

def _crop_indices(so, k_shape, x_shape, mode):
    """
    Compute crop slice boundaries ``(t, b, l, r)`` for the given convolution
    mode, applied to the full output of shape ``so``.

    Returns:
        (Tuple[int, int, int, int]):
            t, b, l, r: Top, bottom, left, right crop boundaries for
            ``output[t:b, l:r]``.
    """
    if mode == 'full':
        t = 0
        b = so[0] + 1
        l = 0
        r = so[1] + 1
    elif mode == 'same':
        t = (k_shape[0] - 1) // 2
        b = -(k_shape[0] - 1) // 2
        l = (k_shape[1] - 1) // 2
        r = -(k_shape[1] - 1) // 2
        b = x_shape[0] + 1 if b == 0 else b
        r = x_shape[1] + 1 if r == 0 else r
    elif mode == 'valid':
        t = (k_shape[0] - 1)
        b = -(k_shape[0] - 1)
        l = (k_shape[1] - 1)
        r = -(k_shape[1] - 1)
        b = x_shape[0] + 1 if b == 0 else b
        r = x_shape[1] + 1 if r == 0 else r
    return t, b, l, r


## ---------------------------------------------------------------------------
## Scipy backend
## ---------------------------------------------------------------------------

def compute_precomputed_scipy(dt, so, x, k_shape, x_shape, mode, batching, issparse):
    """
    Compute convolution using the pre-built scipy Toeplitz matrix.

    Args:
        dt (scipy.sparse.csr_matrix):
            Pre-built Toeplitz matrix.
        so (Tuple[int, int]):
            Full output shape before cropping.
        x (Union[np.ndarray, scipy.sparse.spmatrix]):
            Input array.
        k_shape (Tuple[int, int]):
            Kernel shape.
        x_shape (Tuple[int, int]):
            Input spatial shape.
        mode (str):
            Convolution mode.
        batching (bool):
            Whether x is batched.
        issparse (bool):
            Whether input is sparse.

    Returns:
        (Union[np.ndarray, scipy.sparse.csr_matrix]):
            Convolution output (sparse if input was sparse).
    """
    if batching:
        x_v = x.T  ## (H*W, n_batch) column vectors
    else:
        x_v = x.reshape(-1, 1)

    if issparse:
        x_v = x_v.tocsc()

    out_v = dt @ x_v  ## sparse @ sparse → sparse; sparse @ dense → dense

    ## Crop to requested mode
    t, b, l, r = _crop_indices(so, k_shape, x_shape, mode)

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


## ---------------------------------------------------------------------------
## Numba backend
## ---------------------------------------------------------------------------

if HAS_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def _csr_matvecs_parallel(indptr, indices, data, X, Y):
        """
        Parallel CSR matrix-dense multiply: ``Y += A @ X``.

        Uses numba prange over matrix rows. Each thread handles a chunk of
        rows across all batch columns. The CSR arrays (indptr, indices, data)
        are read once total — much more cache-friendly than parallelizing
        over batch columns (which would re-read the CSR structure B times).

        Args:
            indptr (np.ndarray):
                CSR row pointer, shape ``(n_rows + 1,)``.
            indices (np.ndarray):
                CSR column indices, shape ``(nnz,)``.
            data (np.ndarray):
                CSR values, shape ``(nnz,)``.
            X (np.ndarray):
                Dense input, shape ``(n_cols, n_batch)``. C-contiguous.
            Y (np.ndarray):
                Dense output, shape ``(n_rows, n_batch)``.
                Modified in-place (must be pre-zeroed).
        """
        n_rows = len(indptr) - 1
        n_vecs = X.shape[1]
        for i in numba.prange(n_rows):
            for jj in range(indptr[i], indptr[i + 1]):
                a = data[jj]
                j = indices[jj]
                for b in range(n_vecs):
                    Y[i, b] += a * X[j, b]


def compute_precomputed_numba(dt, so, x, k_shape, x_shape, mode, batching, issparse):
    """
    Compute convolution using the pre-built scipy Toeplitz CSR matrix with
    numba-accelerated parallel CSR matvec (prange over rows).

    Best for **dense** batched inputs (batch >= 2), where it achieves 2-4x
    speedup over scipy's single-threaded C matmul. Also ~1.5x faster for
    single images (batch=1) regardless of sparsity. For **sparse** batched
    inputs (batch >= 5), ``backend='numpy'`` (scipy's CSR × CSC path) is
    faster because it skips structural zeros — this backend densifies the
    input first via ``x.toarray()``.

    Args:
        dt (scipy.sparse.csr_matrix):
            Pre-built Toeplitz matrix.
        so (Tuple[int, int]):
            Full output shape before cropping.
        x (Union[np.ndarray, scipy.sparse.spmatrix]):
            Input array.
        k_shape (Tuple[int, int]):
            Kernel shape.
        x_shape (Tuple[int, int]):
            Input spatial shape.
        mode (str):
            Convolution mode.
        batching (bool):
            Whether x is batched.
        issparse (bool):
            Whether input is sparse.

    Returns:
        (Union[np.ndarray, scipy.sparse.csr_matrix]):
            Convolution output.
    """
    assert HAS_NUMBA, "backend='numba' requires numba to be installed"

    ## Convert input to dense column vectors: (n_pixels, n_batch)
    if issparse:
        x_dense = x.toarray()
    else:
        x_dense = np.asarray(x)

    if batching:
        X = np.ascontiguousarray(x_dense.T, dtype=dt.dtype)
    else:
        X = np.ascontiguousarray(x_dense.reshape(-1, 1), dtype=dt.dtype)

    ## Parallel CSR matvec: Y = T @ X
    n_full = so[0] * so[1]
    Y = np.zeros((n_full, X.shape[1]), dtype=dt.dtype)
    _csr_matvecs_parallel(dt.indptr, dt.indices, dt.data, X, Y)

    ## Crop to requested mode via reshape+slice
    ## Y: (H_full * W_full, n_batch_or_1) → (H_full, W_full, n_batch_or_1)
    t, b, l, r = _crop_indices(so, k_shape, x_shape, mode)
    Y_3d = Y.reshape(so[0], so[1], -1)
    out_cropped = Y_3d[t:b, l:r, :]  ## (H_out, W_out, n_batch_or_1)

    if batching:
        ## (H_out, W_out, n_batch) → (n_batch, H_out * W_out)
        out = out_cropped.reshape(-1, out_cropped.shape[2]).T
    else:
        out = out_cropped[:, :, 0]

    return out


## ---------------------------------------------------------------------------
## Torch backend
## ---------------------------------------------------------------------------

def build_toeplitz_torch(x_shape, k, dtype, device):
    """
    Build the Toeplitz matrix as a torch sparse CSR tensor.

    Uses scipy to build the matrix structure (one-time cost), then converts
    to ``torch.sparse_csr_tensor`` on the specified device.

    Args:
        x_shape (Tuple[int, int]):
            Input spatial dimensions.
        k (np.ndarray):
            2D kernel.
        dtype (np.dtype):
            NumPy dtype for values.
        device (str):
            Torch device string (e.g. ``'cpu'``, ``'cuda'``).

    Returns:
        (Tuple[torch.Tensor, Tuple[int, int]]):
            dt_torch (torch.Tensor):
                Sparse CSR Toeplitz matrix on ``device``.
            so (Tuple[int, int]):
                Full output shape before cropping.
    """
    assert HAS_TORCH, "torch is required for backend='torch'"

    dt_scipy, so = build_toeplitz_scipy(x_shape, k, dtype)

    ## Sort column indices (required by torch.sparse_csr_tensor; scipy ops
    ## can leave indices unsorted after vstack/hstack)
    dt_scipy.sort_indices()

    ## Use float32 on GPU to avoid cuSPARSE float64 issues.
    ## Use int32 indices to save memory (CSR preserves int32 unlike COO).
    is_gpu = device is not None and 'cuda' in str(device)
    if is_gpu and dtype == np.float64:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float64 if dtype == np.float64 else torch.float32

    ## Convert scipy CSR → torch sparse CSR
    dt_torch = torch.sparse_csr_tensor(
        crow_indices=torch.as_tensor(dt_scipy.indptr, dtype=torch.int32),
        col_indices=torch.as_tensor(dt_scipy.indices, dtype=torch.int32),
        values=torch.as_tensor(dt_scipy.data, dtype=torch_dtype),
        size=dt_scipy.shape,
        device=device,
    )

    return dt_torch, so


def compute_precomputed_torch(dt_torch, so, x, k_shape, x_shape, mode, batching, issparse, device):
    """
    Compute convolution using the pre-built torch Toeplitz matrix.
    Supports GPU acceleration when ``device='cuda'``.

    The computation is: ``out = T @ x_columns``, followed by mode-based
    cropping. The output is always converted back to numpy/scipy for
    consistency with the public API.

    Args:
        dt_torch (torch.Tensor):
            Sparse CSR Toeplitz matrix on device.
        so (Tuple[int, int]):
            Full output shape before cropping.
        x (Union[np.ndarray, scipy.sparse.spmatrix]):
            Input array.
        k_shape (Tuple[int, int]):
            Kernel shape.
        x_shape (Tuple[int, int]):
            Input spatial shape.
        mode (str):
            Convolution mode.
        batching (bool):
            Whether x is batched.
        issparse (bool):
            Whether input is sparse.
        device (str):
            Torch device.

    Returns:
        (Union[np.ndarray, scipy.sparse.csr_matrix]):
            Convolution output (sparse CSR if input was sparse, dense
            otherwise).
    """
    torch_dtype = dt_torch.dtype

    ## Convert input to dense numpy (torch.sparse.mm requires dense rhs)
    if issparse:
        x_dense = x.toarray()
    else:
        x_dense = np.asarray(x)

    ## Arrange as column vectors: (n_pixels, n_batch)
    if batching:
        x_cols = x_dense.T.copy()  ## (H*W, n_batch), C-contiguous copy
    else:
        x_cols = x_dense.reshape(-1, 1).copy()

    x_torch = torch.as_tensor(x_cols, dtype=torch_dtype, device=device)

    ## Sparse matmul: (n_full_pixels, n_pixels) @ (n_pixels, n_batch) → (n_full_pixels, n_batch)
    out_v = torch.sparse.mm(dt_torch, x_torch)

    ## Crop to requested mode
    t, b, l, r = _crop_indices(so, k_shape, x_shape, mode)

    ## Reshape to (H_full, W_full, n_batch), crop, flatten back
    out_full = out_v.reshape(so[0], so[1], -1)  ## (H_full, W_full, n_batch_or_1)
    out_cropped = out_full[t:b, l:r, :]

    ## Convert back to numpy
    H_out = out_cropped.shape[0]
    W_out = out_cropped.shape[1]

    if batching:
        ## (H_out, W_out, n_batch) → (n_batch, H_out*W_out)
        result_np = out_cropped.reshape(-1, out_cropped.shape[2]).T.cpu().numpy()
        if issparse:
            return scipy.sparse.csr_matrix(result_np)
        return result_np
    else:
        ## (H_out, W_out, 1) → (H_out, W_out)
        result_np = out_cropped[:, :, 0].cpu().numpy()
        if issparse:
            return scipy.sparse.csr_matrix(result_np)
        return result_np
