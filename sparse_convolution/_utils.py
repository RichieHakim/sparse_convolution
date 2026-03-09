"""
Shared utilities for sparse convolution backends.
"""

import numpy as np
import scipy.sparse


def compute_output_dims(x_shape, k_shape, mode):
    """
    Compute output spatial dimensions and coordinate offsets for a given
    convolution mode.

    Args:
        x_shape (Tuple[int, int]):
            Input spatial dimensions ``(H, W)``.
        k_shape (Tuple[int, int]):
            Kernel dimensions ``(kH, kW)``.
        mode (str):
            Convolution mode: ``'full'``, ``'same'``, or ``'valid'``.

    Returns:
        (Tuple[int, int, int, int]):
            H_out (int): Output height.
            W_out (int): Output width.
            t (int): Row offset (top crop / padding).
            l (int): Column offset (left crop / padding).
    """
    if mode == 'full':
        H_out = x_shape[0] + k_shape[0] - 1
        W_out = x_shape[1] + k_shape[1] - 1
        t = l = 0
    elif mode == 'same':
        H_out = x_shape[0]
        W_out = x_shape[1]
        t = (k_shape[0] - 1) // 2
        l = (k_shape[1] - 1) // 2
    elif mode == 'valid':
        H_out = x_shape[0] - k_shape[0] + 1
        W_out = x_shape[1] - k_shape[1] + 1
        t = k_shape[0] - 1
        l = k_shape[1] - 1
    else:
        raise ValueError(f"mode must be 'full', 'same', or 'valid'. Got: {mode!r}")
    return H_out, W_out, t, l


def extract_kernel_coo(k, dtype):
    """
    Extract kernel nonzero positions and values.

    Args:
        k (np.ndarray):
            2D kernel array.
        dtype (np.dtype):
            Data type for the kernel values.

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray]):
            k_r (np.ndarray): Row indices of nonzeros. Shape ``(nnz_k,)``.
            k_c (np.ndarray): Col indices of nonzeros. Shape ``(nnz_k,)``.
            k_d (np.ndarray): Nonzero values. Shape ``(nnz_k,)``.
    """
    k_nz = np.nonzero(k)
    k_r = k_nz[0].astype(np.int64)
    k_c = k_nz[1].astype(np.int64)
    k_d = k[k_nz].astype(dtype)
    return k_r, k_c, k_d


def extract_input_coo(x, x_shape, batching, dtype):
    """
    Extract input nonzero elements in COO format with 2D spatial coordinates.

    Args:
        x (Union[np.ndarray, scipy.sparse.spmatrix]):
            Input array. If batching: shape ``(n_batch, H*W)``.
            Otherwise: shape ``(H, W)``.
        x_shape (Tuple[int, int]):
            Spatial dimensions ``(H, W)``.
        batching (bool):
            Whether ``x`` contains multiple flattened images.
        dtype (np.dtype):
            Data type for values.

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]):
            x_data (np.ndarray): Nonzero values. Shape ``(nnz,)``.
            x_r (np.ndarray): 2D row coordinates. Shape ``(nnz,)``.
            x_c (np.ndarray): 2D col coordinates. Shape ``(nnz,)``.
            batch_idx (np.ndarray): Batch index per nonzero. Shape ``(nnz,)``.
            n_batch (int): Number of images in the batch.
    """
    x_coo = scipy.sparse.coo_matrix(x)
    x_data = x_coo.data.astype(dtype)

    if batching:
        n_batch = x.shape[0]
        batch_idx = x_coo.row.astype(np.int64)
        x_flat = x_coo.col.astype(np.int64)
        x_r = x_flat // x_shape[1]
        x_c = x_flat % x_shape[1]
    else:
        n_batch = 1
        batch_idx = np.zeros(x_coo.nnz, dtype=np.int64)
        x_r = x_coo.row.astype(np.int64)
        x_c = x_coo.col.astype(np.int64)

    return x_data, x_r, x_c, batch_idx, n_batch
