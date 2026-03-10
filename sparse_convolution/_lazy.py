"""
Lazy COO broadcasting convolution backends.

Computes convolution by broadcasting every input nonzero against every kernel
nonzero to produce output COO triplets, then assembling a sparse output matrix.
No matrix is precomputed at init time. Two backends:

- **numpy** (scipy): Materializes the full ``(nnz_x, nnz_k)`` broadcast arrays
  using numpy, assembles output via ``scipy.sparse.csr_matrix`` (which sums
  duplicate indices). Best for very sparse inputs with small batches.
- **torch**: Same broadcast pattern but using torch tensors and
  ``scatter_add_`` for accumulation. Avoids the scipy COO→CSR sort bottleneck.
  Supports GPU.
"""

import numpy as np
import scipy.sparse

from sparse_convolution._utils import compute_output_dims, extract_kernel_coo

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def compute_lazy_numpy(x, k, x_shape, mode, batching, dtype):
    """
    Lazy convolution using numpy broadcasting and scipy COO assembly.

    For each input nonzero at ``(r, c)`` and each kernel nonzero at
    ``(kr, kc)`` with value ``kd``, contributes ``x_val * kd`` to output
    position ``(r + kr - t, c + kc - l)``, where ``(t, l)`` are the mode
    offsets. All contributions are broadcast at once and assembled into a
    sparse CSR matrix.

    Args:
        x (Union[np.ndarray, scipy.sparse.spmatrix]):
            Input array. Batched: ``(n_batch, H*W)``. Single: ``(H, W)``.
        k (np.ndarray):
            2D kernel.
        x_shape (Tuple[int, int]):
            Spatial dimensions ``(H, W)``.
        mode (str):
            Convolution mode.
        batching (bool):
            Whether ``x`` is batched.
        dtype (np.dtype):
            Output data type.

    Returns:
        (scipy.sparse.csr_matrix):
            Convolution output in CSR format (with duplicates summed).
    """
    H_out, W_out, t, l = compute_output_dims(x_shape, k.shape, mode)

    ## Kernel COO
    k_coo = scipy.sparse.coo_matrix(k)
    if k_coo.dtype != dtype:
        k_coo = k_coo.astype(dtype)
    k_r, k_c, k_d = k_coo.row, k_coo.col, k_coo.data  ## (nnz_k,)

    ## Input COO
    x_coo = scipy.sparse.coo_matrix(x)
    if batching:
        n_batch = x.shape[0]
        batch_idx = x_coo.row
        x_flat_idx = x_coo.col
        x_r = x_flat_idx // x_shape[1]
        x_c = x_flat_idx % x_shape[1]
    else:
        batch_idx = np.zeros_like(x_coo.row)
        x_r = x_coo.row
        x_c = x_coo.col

    ## Broadcast: outer product of input nonzeros × kernel nonzeros
    ## out_r[i, j] = x_r[i] + k_r[j] - t, shape: (nnz_x * nnz_k,)
    out_r = (x_r[:, None] + k_r[None, :] - t).ravel()
    out_c = (x_c[:, None] + k_c[None, :] - l).ravel()
    out_b = np.broadcast_to(batch_idx[:, None], (len(batch_idx), len(k_d))).ravel().copy()
    out_d = (x_coo.data[:, None] * k_d[None, :]).ravel()

    ## Filter out-of-bounds
    valid = (out_r >= 0) & (out_r < H_out) & (out_c >= 0) & (out_c < W_out)
    out_r, out_c, out_b, out_d = out_r[valid], out_c[valid], out_b[valid], out_d[valid]

    ## Assemble sparse output (duplicate indices are summed by scipy)
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


def compute_lazy_torch(x, k, x_shape, mode, batching, dtype, device):
    """
    Lazy convolution using torch broadcasting and ``scatter_add_``.

    Same algorithmic pattern as the numpy backend: broadcast all input
    nonzeros against all kernel nonzeros, filter out-of-bounds, scatter-add
    into a dense buffer. Key advantages over numpy:

    - Uses ``torch.scatter_add_`` which is a single fused operation (no
      COO→CSR sort overhead).
    - Runs on GPU when ``device='cuda'``.

    Args:
        x (Union[np.ndarray, scipy.sparse.spmatrix]):
            Input array.
        k (np.ndarray):
            2D kernel.
        x_shape (Tuple[int, int]):
            Spatial dimensions ``(H, W)``.
        mode (str):
            Convolution mode.
        batching (bool):
            Whether ``x`` is batched.
        dtype (np.dtype):
            Output data type.
        device (str):
            Torch device (``'cpu'``, ``'cuda'``, etc.).

    Returns:
        (scipy.sparse.csr_matrix):
            Convolution output in CSR format.
    """
    assert HAS_TORCH, "torch is required for backend='torch'"

    H_out, W_out, t, l = compute_output_dims(x_shape, k.shape, mode)
    out_pixels = H_out * W_out
    torch_dtype = torch.float64 if dtype == np.float64 else torch.float32

    ## Kernel COO → torch tensors
    k_r_np, k_c_np, k_d_np = extract_kernel_coo(k, dtype)
    k_r = torch.as_tensor(k_r_np, dtype=torch.long, device=device)
    k_c = torch.as_tensor(k_c_np, dtype=torch.long, device=device)
    k_d = torch.as_tensor(k_d_np, dtype=torch_dtype, device=device)
    n_k = len(k_d)

    ## Input COO → torch tensors
    x_coo = scipy.sparse.coo_matrix(x)
    if batching:
        n_batch = x.shape[0]
        batch_idx = torch.as_tensor(x_coo.row, dtype=torch.long, device=device)
        x_flat = x_coo.col
        x_r = torch.as_tensor(x_flat // x_shape[1], dtype=torch.long, device=device)
        x_c = torch.as_tensor(x_flat % x_shape[1], dtype=torch.long, device=device)
    else:
        n_batch = 1
        batch_idx = torch.zeros(x_coo.nnz, dtype=torch.long, device=device)
        x_r = torch.as_tensor(x_coo.row, dtype=torch.long, device=device)
        x_c = torch.as_tensor(x_coo.col, dtype=torch.long, device=device)
    x_data = torch.as_tensor(x_coo.data.astype(dtype), dtype=torch_dtype, device=device)

    ## Broadcast: (nnz_x, 1) + (1, nnz_k) → (nnz_x, nnz_k) → ravel
    out_r = (x_r[:, None] + k_r[None, :] - t).ravel()
    out_c = (x_c[:, None] + k_c[None, :] - l).ravel()
    out_b = batch_idx[:, None].expand(-1, n_k).reshape(-1)
    out_d = (x_data[:, None] * k_d[None, :]).ravel()

    ## Filter out-of-bounds
    valid = (out_r >= 0) & (out_r < H_out) & (out_c >= 0) & (out_c < W_out)
    out_r = out_r[valid]
    out_c = out_c[valid]
    out_b = out_b[valid]
    out_d = out_d[valid]

    ## Scatter-add into dense buffer: (n_batch, out_pixels)
    buf = torch.zeros(n_batch, out_pixels, dtype=torch_dtype, device=device)
    linear_idx = out_b * out_pixels + out_r * W_out + out_c
    buf.view(-1).scatter_add_(0, linear_idx, out_d)

    ## Convert to scipy sparse
    result_np = buf.cpu().numpy()
    out = scipy.sparse.csr_matrix(result_np)

    if not batching and out.shape == (1, out_pixels):
        out = out.reshape((H_out, W_out))
    return out
