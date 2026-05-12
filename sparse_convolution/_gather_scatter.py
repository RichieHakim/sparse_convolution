"""
Gather-scatter convolution backends.

Inspired by spconv's Gather-GEMM-Scatter CUDA algorithm, adapted for CPU
(numpy/numba) and CPU/GPU (torch). For each kernel nonzero position, gathers
input values at shifted coordinates and scatters weighted contributions into
an output accumulator.

Three backends:

- **numpy**: Per-position vectorized ops with ``np.add.at`` or ``np.bincount``.
- **numba**: JIT-compiled batch-parallel scatter with prange over batch
  images. Uses a chunked dense accumulator for bounded memory.
- **torch**: PyTorch ``scatter_add_`` with automatic GPU support.
"""

import numpy as np
import scipy.sparse

from sparse_convolution._utils import compute_output_dims, extract_kernel_coo, extract_input_coo

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
## Numpy backend kernels
## ---------------------------------------------------------------------------

def _scatter_numpy(buf, batch_idx, x_r, x_c, x_data, k_r, k_c, k_d, H_out, W_out, t, l):
    """
    Pure numpy scatter into dense buffer. Auto-selects between two strategies:

    - **Broadcast + bincount** (default): Broadcasts all kernel positions at
      once into (nnz_x * nnz_k) arrays, then uses ``np.bincount`` for
      scatter-add. ~2-3x faster than ``np.add.at`` but requires O(nnz_x *
      nnz_k) temporary memory.
    - **Per-position loop** (fallback): Loops over kernel positions, using
      ``np.add.at`` per position. Lower memory for very large inputs.

    Threshold: uses broadcast+bincount when nnz_x * nnz_k < 5M elements
    (corresponding to ~200MB of temporary arrays).

    Args:
        buf (np.ndarray):
            Dense accumulator, shape ``(chunk_size, H_out * W_out)``.
            Modified in-place.
        batch_idx (np.ndarray):
            Local batch index per input nonzero. Shape ``(nnz_chunk,)``.
        x_r, x_c (np.ndarray):
            Input row/col coordinates. Shape ``(nnz_chunk,)``.
        x_data (np.ndarray):
            Input nonzero values. Shape ``(nnz_chunk,)``.
        k_r, k_c (np.ndarray):
            Kernel nonzero row/col positions. Shape ``(nnz_k,)``.
        k_d (np.ndarray):
            Kernel nonzero values. Shape ``(nnz_k,)``.
        H_out, W_out (int):
            Output spatial dimensions.
        t, l (int):
            Row and column offsets for the convolution mode.
    """
    n_x = len(x_data)
    n_k = len(k_d)
    out_pixels = H_out * W_out
    n_batch = buf.shape[0]

    ## For large products, the broadcast creates huge temporary arrays.
    ## Fall back to per-position np.add.at loop to bound memory.
    ## Threshold: 5M elements ≈ 200MB of temporaries (5 arrays × 8 bytes each)
    if n_x * n_k > 5_000_000:
        for ki in range(n_k):
            kr, kc, kd = k_r[ki], k_c[ki], k_d[ki]
            out_r = x_r + kr - t
            out_c = x_c + kc - l
            valid = (out_r >= 0) & (out_r < H_out) & (out_c >= 0) & (out_c < W_out)
            linear_out = out_r[valid] * W_out + out_c[valid]
            np.add.at(buf, (batch_idx[valid], linear_out), x_data[valid] * kd)
        return

    ## Broadcast all kernel positions at once: (nnz_x, nnz_k) → ravel
    out_r = (x_r[:, None] + k_r[None, :] - t).ravel()
    out_c = (x_c[:, None] + k_c[None, :] - l).ravel()

    ## Expand batch and data arrays to match broadcast shape
    out_b = np.broadcast_to(batch_idx[:, None], (n_x, n_k)).ravel()
    out_d = (x_data[:, None] * k_d[None, :]).ravel()

    ## Filter out-of-bounds
    valid = (out_r >= 0) & (out_r < H_out) & (out_c >= 0) & (out_c < W_out)
    out_r = out_r[valid]
    out_c = out_c[valid]
    out_b = out_b[valid]
    out_d = out_d[valid]

    ## Scatter-add via bincount (much faster than np.add.at)
    flat_idx = out_b * out_pixels + out_r * W_out + out_c
    counts = np.bincount(flat_idx, weights=out_d, minlength=n_batch * out_pixels)
    buf += counts.reshape(n_batch, out_pixels)


def _coo_numpy(batch_idx, x_r, x_c, x_data, k_r, k_c, k_d, H_out, W_out, t, l):
    """
    Pure numpy COO builder. Per-kernel-position vectorized operations.
    Fallback when numba is unavailable and the dense buffer strategy is
    infeasible.

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray]):
            out_rows, out_cols, out_data: COO triplets for the output matrix.
    """
    all_rows = []
    all_cols = []
    all_data = []
    n_k = len(k_d)
    if n_k == 0:
        empty_index = np.array([], dtype=np.int64)
        empty_data = np.array([], dtype=x_data.dtype)
        return empty_index, empty_index, empty_data
    for ki in range(n_k):
        kr, kc, kd = k_r[ki], k_c[ki], k_d[ki]
        out_r = x_r + kr - t
        out_c = x_c + kc - l
        valid = (out_r >= 0) & (out_r < H_out) & (out_c >= 0) & (out_c < W_out)
        all_rows.append(batch_idx[valid])
        all_cols.append(out_r[valid] * W_out + out_c[valid])
        all_data.append(x_data[valid] * kd)
    return np.concatenate(all_rows), np.concatenate(all_cols), np.concatenate(all_data)


## ---------------------------------------------------------------------------
## Numba backend kernels (conditionally defined)
## ---------------------------------------------------------------------------

if HAS_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def _scatter_numba(buf, batch_starts, x_r, x_c, x_data, k_r, k_c, k_d, H_out, W_out, t, l):
        """
        Batch-parallel numba scatter kernel. Uses prange over batch images
        with xi-outer inner loop for good output cache locality.

        Each thread handles one batch image — zero write conflicts since
        different threads write to different rows of ``buf``.

        Args:
            buf (np.ndarray):
                Dense accumulator, shape ``(n_batch, H_out * W_out)``.
                Modified in-place (must be pre-zeroed).
            batch_starts (np.ndarray):
                Cumulative offsets into the sorted input arrays, shape
                ``(n_batch + 1,)``. ``batch_starts[b]`` is the first index
                in (x_r, x_c, x_data) belonging to batch image ``b``.
            x_r, x_c (np.ndarray):
                Input row/col coordinates, sorted by batch. Shape ``(nnz,)``.
            x_data (np.ndarray):
                Input nonzero values, sorted by batch. Shape ``(nnz,)``.
            k_r, k_c (np.ndarray):
                Kernel nonzero row/col positions. Shape ``(nnz_k,)``.
            k_d (np.ndarray):
                Kernel nonzero values. Shape ``(nnz_k,)``.
            H_out, W_out (int):
                Output spatial dimensions.
            t, l (int):
                Row and column offsets for the convolution mode.
        """
        n_k = len(k_d)
        n_batch = len(batch_starts) - 1
        for b in numba.prange(n_batch):
            start = batch_starts[b]
            end = batch_starts[b + 1]
            for xi in range(start, end):
                xr = x_r[xi]
                xc = x_c[xi]
                xd = x_data[xi]
                for ki in range(n_k):
                    out_r = xr + k_r[ki] - t
                    out_c = xc + k_c[ki] - l
                    if 0 <= out_r < H_out and 0 <= out_c < W_out:
                        buf[b, out_r * W_out + out_c] += xd * k_d[ki]

    @numba.njit(cache=True)
    def _coo_numba(batch_idx, x_r, x_c, x_data, k_r, k_c, k_d, H_out, W_out, t, l):
        """
        Numba JIT COO builder. Same double loop as the dense version but
        writes output triplets directly instead of into a dense buffer.
        Pre-allocates max_out arrays and returns sliced views.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]):
                out_rows, out_cols, out_data.
        """
        n_k = len(k_d)
        n_x = len(x_data)
        max_out = n_x * n_k
        out_rows = np.empty(max_out, dtype=np.int64)
        out_cols = np.empty(max_out, dtype=np.int64)
        out_data = np.empty(max_out, dtype=np.float64)

        idx = 0
        for ki in range(n_k):
            kr = k_r[ki]
            kc = k_c[ki]
            kd = k_d[ki]
            for xi in range(n_x):
                out_r = x_r[xi] + kr - t
                out_c = x_c[xi] + kc - l
                if 0 <= out_r < H_out and 0 <= out_c < W_out:
                    out_rows[idx] = batch_idx[xi]
                    out_cols[idx] = out_r * W_out + out_c
                    out_data[idx] = x_data[xi] * kd
                    idx += 1

        return out_rows[:idx], out_cols[:idx], out_data[:idx]


## ---------------------------------------------------------------------------
## Torch backend kernel
## ---------------------------------------------------------------------------

def _scatter_torch(batch_idx, x_r, x_c, x_data, k_r, k_c, k_d,
                   H_out, W_out, t, l, n_batch, out_pixels, dtype_np, device):
    """
    PyTorch scatter kernel. Broadcasts all kernel positions at once (fully
    vectorized), then uses a single ``scatter_add_`` call. More efficient
    than per-position looping because it reduces Python overhead and allows
    better GPU utilization.

    Args:
        batch_idx, x_r, x_c, x_data (np.ndarray):
            Input COO arrays.
        k_r, k_c, k_d (np.ndarray):
            Kernel nonzero positions and values.
        H_out, W_out, t, l (int):
            Output dimensions and mode offsets.
        n_batch (int):
            Number of images.
        out_pixels (int):
            ``H_out * W_out``.
        dtype_np (np.dtype):
            NumPy dtype for output.
        device (str):
            Torch device string.

    Returns:
        (np.ndarray): Dense output buffer, shape ``(n_batch, out_pixels)``.
    """
    torch_dtype = torch.float64 if dtype_np == np.float64 else torch.float32

    ## Convert to torch tensors on device
    t_batch = torch.as_tensor(batch_idx, dtype=torch.long, device=device)
    t_xr = torch.as_tensor(x_r, dtype=torch.long, device=device)
    t_xc = torch.as_tensor(x_c, dtype=torch.long, device=device)
    t_xdata = torch.as_tensor(x_data, dtype=torch_dtype, device=device)
    t_kr = torch.as_tensor(k_r, dtype=torch.long, device=device)
    t_kc = torch.as_tensor(k_c, dtype=torch.long, device=device)
    t_kd = torch.as_tensor(k_d, dtype=torch_dtype, device=device)

    n_k = len(k_d)

    ## Broadcast all kernel positions at once: (nnz_x, nnz_k)
    ## out_r[i, j] = x_r[i] + k_r[j] - t
    out_r = (t_xr[:, None] + t_kr[None, :] - t).ravel()
    out_c = (t_xc[:, None] + t_kc[None, :] - l).ravel()
    out_b = t_batch[:, None].expand(-1, n_k).reshape(-1)
    out_d = (t_xdata[:, None] * t_kd[None, :]).ravel()

    ## Filter out-of-bounds
    valid = (out_r >= 0) & (out_r < H_out) & (out_c >= 0) & (out_c < W_out)

    ## Single scatter-add call
    flat_idx = out_b[valid] * out_pixels + out_r[valid] * W_out + out_c[valid]
    buf = torch.zeros(n_batch * out_pixels, dtype=torch_dtype, device=device)
    buf.scatter_add_(0, flat_idx, out_d[valid])

    return buf.reshape(n_batch, out_pixels).cpu().numpy()


## ---------------------------------------------------------------------------
## Public dispatch: strategy selection + backend routing
## ---------------------------------------------------------------------------

def compute_gather_scatter(x, k, x_shape, mode, batching, dtype,
                           backend, max_buffer_bytes, device):
    """
    Compute convolution using spconv-style per-kernel-position
    gather-scatter with automatic strategy selection.

    Automatically selects between dense accumulator and COO construction
    strategies based on a heuristic comparing the dense buffer size to the
    expected output nnz.

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
        backend (str):
            One of ``'numpy'``, ``'numba'``, or ``'torch'``.
        max_buffer_bytes (int):
            Maximum memory for the dense accumulator buffer.
        device (str):
            Torch device (only used when ``backend='torch'``).

    Returns:
        (scipy.sparse.csr_matrix):
            Convolution output in CSR format.
    """
    H_out, W_out, t, l = compute_output_dims(x_shape, k.shape, mode)
    out_pixels = H_out * W_out

    ## Extract kernel COO (shared by all paths)
    k_r, k_c, k_d = extract_kernel_coo(k, dtype)
    n_k = len(k_d)

    ## Extract input COO, then route to backend
    x_data, x_r, x_c, batch_idx, n_batch = extract_input_coo(x, x_shape, batching, dtype)

    ## Strategy selection heuristic:
    ## Dense buffer for a single image = out_pixels * itemsize bytes.
    ## If the expected output nnz per image is much less than out_pixels,
    ## the dense buffer wastes time scanning zeros during CSR conversion.
    bytes_per_row = out_pixels * np.dtype(dtype).itemsize
    avg_nnz_per_image = max(1, len(x_data) / max(1, n_batch))
    expected_out_nnz_per_image = avg_nnz_per_image * n_k  ## upper bound
    use_dense = (out_pixels <= 10 * expected_out_nnz_per_image) or (bytes_per_row <= 8192)

    if use_dense:
        return _dispatch_dense(
            x_data=x_data, x_r=x_r, x_c=x_c,
            batch_idx=batch_idx, n_batch=n_batch,
            k_r=k_r, k_c=k_c, k_d=k_d,
            H_out=H_out, W_out=W_out, out_pixels=out_pixels,
            t=t, l=l, dtype=dtype, batching=batching,
            backend=backend, max_buffer_bytes=max_buffer_bytes, device=device,
        )
    else:
        return _dispatch_coo(
            x_data=x_data, x_r=x_r, x_c=x_c,
            batch_idx=batch_idx, n_batch=n_batch,
            k_r=k_r, k_c=k_c, k_d=k_d,
            H_out=H_out, W_out=W_out, out_pixels=out_pixels,
            t=t, l=l, dtype=dtype, batching=batching,
            backend=backend, device=device,
        )


def _dispatch_dense(x_data, x_r, x_c, batch_idx, n_batch,
                    k_r, k_c, k_d, H_out, W_out, out_pixels, t, l,
                    dtype, batching, backend, max_buffer_bytes, device):
    """
    Dense accumulator strategy: scatter-add into a dense buffer per chunk,
    then convert to CSR.
    """
    bytes_per_row = out_pixels * np.dtype(dtype).itemsize
    chunk_size = max(1, min(n_batch, max_buffer_bytes // bytes_per_row))

    ## Torch backend: process all images in one call
    if backend == 'torch':
        buf = _scatter_torch(
            batch_idx, x_r, x_c, x_data,
            k_r, k_c, k_d,
            H_out, W_out, t, l,
            n_batch, out_pixels, dtype, device,
        )
        out = scipy.sparse.csr_matrix(buf)
        if not batching and out.shape == (1, out_pixels):
            out = out.reshape((H_out, W_out))
        return out

    ## Numpy/numba backend: chunked processing with dense buffers
    use_numba = backend == 'numba'
    if use_numba:
        assert HAS_NUMBA, "numba is required for backend='numba'"

    result_chunks = []
    for chunk_start in range(0, n_batch, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_batch)
        actual_chunk = chunk_end - chunk_start

        ## Select nonzeros belonging to this chunk
        mask = (batch_idx >= chunk_start) & (batch_idx < chunk_end)
        c_batch = batch_idx[mask] - chunk_start
        c_xr = x_r[mask]
        c_xc = x_c[mask]
        c_data = x_data[mask]

        ## Dense accumulator: shape (actual_chunk, out_pixels)
        buf = np.zeros((actual_chunk, out_pixels), dtype=dtype)

        if use_numba:
            ## Ensure sorted by batch (already true for CSR/dense input;
            ## needed for CSC input where COO row order is column-major)
            if len(c_batch) > 1 and np.any(c_batch[1:] < c_batch[:-1]):
                sort_idx = np.argsort(c_batch, kind='stable')
                c_batch = c_batch[sort_idx]
                c_xr = c_xr[sort_idx]
                c_xc = c_xc[sort_idx]
                c_data = c_data[sort_idx]
            ## Compute batch offsets for parallel scatter
            batch_starts = np.searchsorted(
                c_batch, np.arange(actual_chunk + 1),
            ).astype(np.int64)
            _scatter_numba(
                buf, batch_starts, c_xr, c_xc, c_data,
                k_r, k_c, k_d,
                H_out, W_out, t, l,
            )
        else:
            _scatter_numpy(
                buf, c_batch, c_xr, c_xc, c_data,
                k_r, k_c, k_d,
                H_out, W_out, t, l,
            )

        result_chunks.append(scipy.sparse.csr_matrix(buf))

    ## Assemble output
    if batching:
        out = scipy.sparse.vstack(result_chunks, format='csr')
    else:
        out = result_chunks[0]
        if out.shape == (1, out_pixels):
            out = out.reshape((H_out, W_out))
    return out


def _dispatch_coo(x_data, x_r, x_c, batch_idx, n_batch,
                  k_r, k_c, k_d, H_out, W_out, out_pixels, t, l,
                  dtype, batching, backend, device):
    """
    COO construction strategy: build output COO triplets directly, then
    construct CSR. Avoids the massive dense buffer when output is very sparse.
    """
    ## Torch backend: use scatter into dense buffer regardless (torch has
    ## no efficient COO→CSR path, and scatter_add_ is fast)
    if backend == 'torch':
        buf = _scatter_torch(
            batch_idx, x_r, x_c, x_data,
            k_r, k_c, k_d,
            H_out, W_out, t, l,
            n_batch, out_pixels, dtype, device,
        )
        out = scipy.sparse.csr_matrix(buf)
        if not batching and out.shape == (1, out_pixels):
            out = out.reshape((H_out, W_out))
        return out

    ## Numba or numpy COO builder
    if backend == 'numba' and HAS_NUMBA:
        out_rows, out_cols, out_data = _coo_numba(
            batch_idx, x_r, x_c, x_data,
            k_r, k_c, k_d,
            H_out, W_out, t, l,
        )
    else:
        out_rows, out_cols, out_data = _coo_numpy(
            batch_idx, x_r, x_c, x_data,
            k_r, k_c, k_d,
            H_out, W_out, t, l,
        )

    ## Build CSR from COO triplets (scipy sums duplicates automatically)
    if batching:
        out = scipy.sparse.csr_matrix(
            (out_data, (out_rows, out_cols)),
            shape=(n_batch, out_pixels),
            dtype=dtype,
        )
    else:
        out_r_2d = out_cols // W_out
        out_c_2d = out_cols % W_out
        out = scipy.sparse.csr_matrix(
            (out_data, (out_r_2d, out_c_2d)),
            shape=(H_out, W_out),
            dtype=dtype,
        )

    out.sum_duplicates()
    return out
