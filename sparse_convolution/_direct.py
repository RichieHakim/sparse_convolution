"""
Direct CSR scatter convolution backend (numba only).

Two-pass batch-parallel scatter using thread-local dense buffers
(L2-cache-sized, ~80KB for 100x100 images). Each numba thread handles
one batch image — zero write conflicts.

- **Pass 1**: scatter kernel-weighted input values into a local dense
  buffer, count output nonzeros. Two adaptive variants:

  - *Scan* (``_scatter_count_scan``): sequential scan of the full buffer
    after scatter. O(nnz_i * n_k + out_pixels) per image. Best when
    output is dense (many scatter ops relative to out_pixels).
  - *Tracked* (``_scatter_count_tracked``): inline nnz counter that
    tracks zero-to-nonzero transitions during scatter. O(nnz_i * n_k)
    per image with ~2-3 cycles overhead per scatter op. Best when
    output is sparse.

- **Pass 2** (``_scatter_fill_csr``): re-scatter and extract nonzeros
  via sequential scan into pre-allocated CSR arrays. Naturally sorted
  column indices — no sort needed.

Complexity (per image, tracked path):
    Work: O(nnz_i * n_k)
    Memory: O(out_pixels) per thread + O(total_nnz_output) total

Performance: 5-17x faster than precomputed+scipy at batch=50k across
all density regimes (0.001-0.01), because thread-local buffers fit in
L2 cache and avoid the O(N * out_pixels) global dense buffer.
"""

import numpy as np
import scipy.sparse

from sparse_convolution._utils import compute_output_dims, extract_kernel_coo

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


## ---------------------------------------------------------------------------
## Numba kernels (conditionally defined)
## ---------------------------------------------------------------------------

if HAS_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def _scatter_count_scan(x_indptr, x_indices, x_data, k_vals, k_rows, k_cols,
                            W_in, H_out, W_out, t, l, counts):
        """
        Pass 1 (scan variant): scatter into thread-local dense buffer, then
        scan the full buffer to count nonzeros. O(nnz_i * n_k + out_pixels)
        per image. Best when scatter ops >= out_pixels / 2 (dense output).

        Args:
            x_indptr (np.ndarray):
                CSR row pointers, shape ``(n_batch + 1,)``.
            x_indices (np.ndarray):
                CSR column indices (flat pixel positions), shape ``(nnz,)``.
            x_data (np.ndarray):
                CSR values, shape ``(nnz,)``.
            k_vals, k_rows, k_cols (np.ndarray):
                Kernel nonzero values and positions, shape ``(Nk,)``.
            W_in (int):
                Input width.
            H_out, W_out (int):
                Output spatial dimensions.
            t, l (int):
                Mode offsets.
            counts (np.ndarray):
                Output array for per-image nnz counts, shape ``(n_batch,)``.
        """
        Nk = len(k_vals)
        n_batch = len(x_indptr) - 1
        out_pixels = H_out * W_out
        for b in numba.prange(n_batch):
            buf = np.zeros(out_pixels, dtype=x_data.dtype)
            for jj in range(x_indptr[b], x_indptr[b + 1]):
                x_flat = x_indices[jj]
                x_val = x_data[jj]
                x_r = x_flat // W_in
                x_c = x_flat % W_in
                for ki in range(Nk):
                    out_r = x_r + k_rows[ki] - t
                    out_c = x_c + k_cols[ki] - l
                    if 0 <= out_r < H_out and 0 <= out_c < W_out:
                        buf[out_r * W_out + out_c] += x_val * k_vals[ki]
            c = 0
            for j in range(out_pixels):
                if buf[j] != 0.0:
                    c += 1
            counts[b] = c

    @numba.njit(parallel=True, cache=True)
    def _scatter_count_tracked(x_indptr, x_indices, x_data, k_vals, k_rows, k_cols,
                               W_in, H_out, W_out, t, l, counts):
        """
        Pass 1 (tracked variant): scatter with inline nnz counter. Tracks
        transitions to/from zero during scatter, avoiding the O(out_pixels)
        buffer scan. O(nnz_i * n_k) per image — matches the theoretical
        lower bound. Best when scatter ops << out_pixels (sparse output).

        Adds ~2-3 cycles overhead per scatter op (one extra float load for
        ``old_val``, two comparisons, one conditional increment). Only used
        when expected scatter ops < out_pixels * 0.3.

        Args:
            x_indptr, x_indices, x_data (np.ndarray):
                CSR input arrays.
            k_vals, k_rows, k_cols (np.ndarray):
                Kernel nonzero values and positions.
            W_in, H_out, W_out, t, l (int):
                Geometry parameters.
            counts (np.ndarray):
                Output array for per-image nnz counts, shape ``(n_batch,)``.
        """
        Nk = len(k_vals)
        n_batch = len(x_indptr) - 1
        out_pixels = H_out * W_out
        for b in numba.prange(n_batch):
            buf = np.zeros(out_pixels, dtype=x_data.dtype)
            n_nonzero = 0
            for jj in range(x_indptr[b], x_indptr[b + 1]):
                x_flat = x_indices[jj]
                x_val = x_data[jj]
                x_r = x_flat // W_in
                x_c = x_flat % W_in
                for ki in range(Nk):
                    out_r = x_r + k_rows[ki] - t
                    out_c = x_c + k_cols[ki] - l
                    if 0 <= out_r < H_out and 0 <= out_c < W_out:
                        idx = out_r * W_out + out_c
                        old = buf[idx]
                        new = old + x_val * k_vals[ki]
                        buf[idx] = new
                        ## Track transitions: 0->nonzero or nonzero->0
                        if old == 0.0:
                            if new != 0.0:
                                n_nonzero += 1
                        elif new == 0.0:
                            n_nonzero -= 1
            counts[b] = n_nonzero

    @numba.njit(parallel=True, cache=True)
    def _scatter_fill_csr(x_indptr, x_indices, x_data, k_vals, k_rows, k_cols,
                          W_in, H_out, W_out, t, l,
                          out_indptr, out_indices, out_data):
        """
        Pass 2: scatter into thread-local buffer, extract nonzeros via
        sequential scan (produces sorted CSR indices naturally).

        Args:
            x_indptr, x_indices, x_data (np.ndarray):
                CSR input arrays.
            k_vals, k_rows, k_cols (np.ndarray):
                Kernel nonzero values and positions.
            W_in, H_out, W_out, t, l (int):
                Geometry parameters.
            out_indptr (np.ndarray):
                CSR row pointers for output, shape ``(n_batch + 1,)``.
            out_indices (np.ndarray):
                CSR column indices to fill, shape ``(total_nnz,)``.
            out_data (np.ndarray):
                CSR values to fill, shape ``(total_nnz,)``.
        """
        Nk = len(k_vals)
        n_batch = len(x_indptr) - 1
        out_pixels = H_out * W_out
        for b in numba.prange(n_batch):
            buf = np.zeros(out_pixels, dtype=out_data.dtype)
            for jj in range(x_indptr[b], x_indptr[b + 1]):
                x_flat = x_indices[jj]
                x_val = x_data[jj]
                x_r = x_flat // W_in
                x_c = x_flat % W_in
                for ki in range(Nk):
                    out_r = x_r + k_rows[ki] - t
                    out_c = x_c + k_cols[ki] - l
                    if 0 <= out_r < H_out and 0 <= out_c < W_out:
                        buf[out_r * W_out + out_c] += x_val * k_vals[ki]
            pos = out_indptr[b]
            for j in range(out_pixels):
                v = buf[j]
                if v != 0.0:
                    out_indices[pos] = j
                    out_data[pos] = v
                    pos += 1


## ---------------------------------------------------------------------------
## Public dispatch
## ---------------------------------------------------------------------------

def compute_direct(x, k, x_shape, mode, batching, dtype):
    """
    Compute convolution using two-pass direct CSR scatter (numba only).

    Requires numba. Operates directly on CSR input — no intermediate
    dense buffer, no COO construction, no Toeplitz matrix. Each thread
    uses a local buffer of size ``out_pixels`` that fits in L2 cache.

    For ``batching=False``, wraps the single image as a 1-row CSR matrix
    internally.

    Args:
        x (Union[np.ndarray, scipy.sparse.spmatrix]):
            Input array. If dense, converted to CSR internally.
        k (np.ndarray):
            2D kernel.
        x_shape (Tuple[int, int]):
            Spatial dimensions ``(H, W)``.
        mode (str):
            Convolution mode: ``'full'``, ``'same'``, or ``'valid'``.
        batching (bool):
            Whether ``x`` is batched.
        dtype (np.dtype):
            Output data type.

    Returns:
        (scipy.sparse.csr_matrix):
            Convolution output in CSR format. Shape
            ``(n_batch, H_out * W_out)`` if batching, else
            ``(H_out, W_out)``.
    """
    H_out, W_out, t, l = compute_output_dims(x_shape, k.shape, mode)
    out_pixels = H_out * W_out

    ## Extract kernel COO
    k_r, k_c, k_d = extract_kernel_coo(k, dtype)
    n_k = len(k_d)

    ## Handle batching=False: wrap as 1-row CSR, run, reshape
    if not batching:
        if not scipy.sparse.issparse(x):
            x_flat = scipy.sparse.csr_matrix(x.reshape(1, -1))
        else:
            x_flat = scipy.sparse.csr_matrix(x.reshape(1, -1))
        out = _run_two_pass(
            x_csr=x_flat, k_r=k_r, k_c=k_c, k_d=k_d, n_k=n_k,
            x_shape=x_shape, H_out=H_out, W_out=W_out,
            out_pixels=out_pixels, t=t, l=l, dtype=dtype,
        )
        ## Reshape from (1, out_pixels) to (H_out, W_out)
        return out.reshape(H_out, W_out)

    ## Batched path: ensure CSR format
    if not scipy.sparse.issparse(x):
        x_csr = scipy.sparse.csr_matrix(x)
    elif not isinstance(x, scipy.sparse.csr_matrix):
        x_csr = x.tocsr()
    else:
        x_csr = x

    return _run_two_pass(
        x_csr=x_csr, k_r=k_r, k_c=k_c, k_d=k_d, n_k=n_k,
        x_shape=x_shape, H_out=H_out, W_out=W_out,
        out_pixels=out_pixels, t=t, l=l, dtype=dtype,
    )


def _run_two_pass(x_csr, k_r, k_c, k_d, n_k, x_shape, H_out, W_out,
                  out_pixels, t, l, dtype):
    """
    Execute the two-pass scatter → CSR construction.

    Args:
        x_csr (scipy.sparse.csr_matrix):
            Input in CSR format, shape ``(n_batch, H * W)``.
        k_r, k_c, k_d (np.ndarray):
            Kernel COO arrays.
        n_k (int):
            Number of kernel nonzeros.
        x_shape (Tuple[int, int]):
            Spatial dimensions ``(H, W)``.
        H_out, W_out (int):
            Output spatial dimensions.
        out_pixels (int):
            ``H_out * W_out``.
        t, l (int):
            Mode offsets.
        dtype (np.dtype):
            Output dtype.

    Returns:
        (scipy.sparse.csr_matrix):
            Output CSR matrix, shape ``(n_batch, out_pixels)``.
    """
    n_batch = x_csr.shape[0]

    ## Cast CSR arrays to numba-compatible types once
    csr_indptr = x_csr.indptr.astype(np.int64)
    csr_indices = x_csr.indices.astype(np.int64)
    csr_data = x_csr.data.astype(dtype)

    ## Pass 1: scatter + count output nnz per image.
    ## Adaptive kernel selection: tracked counter avoids the O(out_pixels)
    ## buffer scan but adds ~2-3 cycles per scatter op. Crossover at
    ## scatter_ops ~ 0.3 * out_pixels (empirically tuned).
    avg_nnz = x_csr.nnz / max(1, n_batch)
    scatter_ops_per_image = avg_nnz * n_k
    use_tracked = scatter_ops_per_image < out_pixels * 0.3

    counts = np.empty(n_batch, dtype=np.int64)
    count_fn = _scatter_count_tracked if use_tracked else _scatter_count_scan
    count_fn(
        csr_indptr, csr_indices, csr_data,
        k_d, k_r, k_c,
        x_shape[1], H_out, W_out, t, l,
        counts,
    )

    ## Build CSR indptr from counts
    indptr = np.empty(n_batch + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    total_nnz = int(indptr[-1])

    ## Pass 2: scatter + extract to CSR arrays
    indices = np.empty(total_nnz, dtype=np.int32)
    data = np.empty(total_nnz, dtype=dtype)
    if total_nnz > 0:
        _scatter_fill_csr(
            csr_indptr, csr_indices, csr_data,
            k_d, k_r, k_c,
            x_shape[1], H_out, W_out, t, l,
            indptr, indices, data,
        )

    return scipy.sparse.csr_matrix(
        (data, indices, indptr),
        shape=(n_batch, out_pixels), copy=False,
    )
