"""
Direct CSR scatter convolution backend (numba only).

Adaptive batch-parallel scatter using thread-local dense buffers
(L2-cache-sized, ~80KB for 100x100 images). Each numba thread handles
one batch image — zero write conflicts.

**Architecture:**

1. **Precompute** kernel delta table (``k_deltas[ki] = k_rows[ki] * W_out
   + k_cols[ki]``) and interior pixel bounds. Eliminates per-scatter-op
   2D index arithmetic and bounds checking for ~92-100% of input pixels.

2. **Adaptive dispatch** based on expected output density
   (``mean_nnz * n_k / out_pixels``):

   - **Sparse output** (ratio < 1): two-phase count → scatter.
     Phase 1 uses lightweight 1-byte flags (~5× cheaper than scatter)
     to get exact nnz. Phase 2 writes directly to exact-size arrays.
     No over-allocation, no compaction.

   - **Dense output** (ratio >= 1): single-pass over-allocate → scatter.
     Upper bound equals ``out_pixels`` (exact for dense output), so
     over-allocation waste is negligible. Skips the counting pass
     entirely. Optional compaction if upper bounds weren't exact.

Complexity (per image):
    Work: O(nnz_i * n_k + out_pixels)
    Memory: O(out_pixels) per thread + O(total_nnz_output) total
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
    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _count_nnz(x_indptr, x_indices,
                   k_rows, k_cols, k_deltas,
                   W_in, H_out, W_out, out_pixels, t, l, offset,
                   r_lo, r_hi, c_lo, c_hi,
                   Nk, counts):
        """
        Lightweight boolean counting pass: compute exact output nnz per image.

        Uses a 1-byte flag array instead of an 8-byte float buffer. No float
        multiply — just sets flags for which output positions are touched.
        """
        n_batch = len(x_indptr) - 1
        for b in numba.prange(n_batch):
            flags = np.zeros(out_pixels, dtype=numba.uint8)

            for jj in range(x_indptr[b], x_indptr[b + 1]):
                x_flat = x_indices[jj]
                x_r = x_flat // W_in
                x_c = x_flat % W_in

                if r_lo <= x_r < r_hi and c_lo <= x_c < c_hi:
                    x_base = x_r * W_out + x_c - offset
                    for ki in range(Nk):
                        flags[x_base + k_deltas[ki]] = 1
                else:
                    for ki in range(Nk):
                        out_r = x_r + k_rows[ki] - t
                        out_c = x_c + k_cols[ki] - l
                        if 0 <= out_r < H_out and 0 <= out_c < W_out:
                            flags[out_r * W_out + out_c] = 1

            count = np.int64(0)
            for j in range(out_pixels):
                count += flags[j]
            counts[b] = count

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _scatter_extract(x_indptr, x_indices, x_data,
                         k_vals, k_rows, k_cols, k_deltas,
                         W_in, H_out, W_out, out_pixels, t, l, offset,
                         r_lo, r_hi, c_lo, c_hi,
                         out_indptr, out_indices, out_data):
        """
        Scatter + sequential-scan extraction into exact-size CSR arrays.

        Writes directly to pre-allocated exact-size output arrays.
        No over-allocation, no compaction step.
        """
        Nk = len(k_vals)
        n_batch = len(x_indptr) - 1
        for b in numba.prange(n_batch):
            buf = np.zeros(out_pixels, dtype=out_data.dtype)

            for jj in range(x_indptr[b], x_indptr[b + 1]):
                x_flat = x_indices[jj]
                x_val = x_data[jj]
                x_r = x_flat // W_in
                x_c = x_flat % W_in

                if r_lo <= x_r < r_hi and c_lo <= x_c < c_hi:
                    x_base = x_r * W_out + x_c - offset
                    for ki in range(Nk):
                        buf[x_base + k_deltas[ki]] += x_val * k_vals[ki]
                else:
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

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _scatter_extract_counted(x_indptr, x_indices, x_data,
                                 k_vals, k_rows, k_cols, k_deltas,
                                 W_in, H_out, W_out, out_pixels, t, l, offset,
                                 r_lo, r_hi, c_lo, c_hi,
                                 ub_indptr, out_indices, out_data,
                                 actual_counts):
        """
        Scatter + extraction into over-allocated CSR arrays, recording
        actual nnz per image. Used for the dense-output path where the
        counting pass is skipped.
        """
        Nk = len(k_vals)
        n_batch = len(x_indptr) - 1
        for b in numba.prange(n_batch):
            buf = np.zeros(out_pixels, dtype=out_data.dtype)

            for jj in range(x_indptr[b], x_indptr[b + 1]):
                x_flat = x_indices[jj]
                x_val = x_data[jj]
                x_r = x_flat // W_in
                x_c = x_flat % W_in

                if r_lo <= x_r < r_hi and c_lo <= x_c < c_hi:
                    x_base = x_r * W_out + x_c - offset
                    for ki in range(Nk):
                        buf[x_base + k_deltas[ki]] += x_val * k_vals[ki]
                else:
                    for ki in range(Nk):
                        out_r = x_r + k_rows[ki] - t
                        out_c = x_c + k_cols[ki] - l
                        if 0 <= out_r < H_out and 0 <= out_c < W_out:
                            buf[out_r * W_out + out_c] += x_val * k_vals[ki]

            pos = ub_indptr[b]
            count = np.int64(0)
            for j in range(out_pixels):
                v = buf[j]
                if v != 0.0:
                    out_indices[pos] = j
                    out_data[pos] = v
                    pos += 1
                    count += 1
            actual_counts[b] = count

    @numba.njit(parallel=True, cache=True)
    def _compact_csr(ub_indptr, actual_indptr, actual_counts,
                     src_indices, src_data, dst_indices, dst_data):
        """
        Parallel compaction of over-allocated CSR arrays into tightly-packed
        output.
        """
        n_batch = len(actual_counts)
        for b in numba.prange(n_batch):
            src_start = ub_indptr[b]
            dst_start = actual_indptr[b]
            n = actual_counts[b]
            for i in range(n):
                dst_indices[dst_start + i] = src_indices[src_start + i]
                dst_data[dst_start + i] = src_data[src_start + i]


## ---------------------------------------------------------------------------
## Public dispatch
## ---------------------------------------------------------------------------

def compute_direct(x, k, x_shape, mode, batching, dtype):
    """
    Compute convolution using adaptive direct CSR scatter (numba only).

    Requires numba. Operates directly on CSR input — no intermediate
    Toeplitz matrix, no COO construction. Each thread uses a local buffer
    of size ``out_pixels`` that fits in L2 cache.

    Adaptively chooses between two strategies based on expected output
    density:

    - **Sparse output** (scatter_ops < out_pixels): two-phase count →
      scatter. Phase 1 uses 1-byte flags to get exact nnz. Phase 2
      writes directly to exact-size arrays. No compaction.
    - **Dense output** (scatter_ops >= out_pixels): single-pass scatter
      with upper-bound allocation. Skips the counting pass (which would
      be redundant since output is nearly dense). Optional compaction.

    Also uses interior/boundary split and precomputed delta tables for
    index arithmetic reduction.

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

    ## Precompute delta table
    k_deltas = (k_r * W_out + k_c).astype(np.int64)
    offset = np.int64(t * W_out + l)

    ## Precompute interior pixel bounds
    if n_k > 0:
        r_lo = np.int64(t - int(k_r.min()))
        r_hi = np.int64(H_out + t - int(k_r.max()))
        c_lo = np.int64(l - int(k_c.min()))
        c_hi = np.int64(W_out + l - int(k_c.max()))
    else:
        r_lo = r_hi = c_lo = c_hi = np.int64(0)

    ## Handle batching=False: wrap as 1-row CSR, run, reshape
    if not batching:
        if not scipy.sparse.issparse(x):
            x_flat = scipy.sparse.csr_matrix(x.reshape(1, -1))
        else:
            x_flat = scipy.sparse.csr_matrix(x.reshape(1, -1))
        out = _run_adaptive(
            x_csr=x_flat, k_r=k_r, k_c=k_c, k_d=k_d, k_deltas=k_deltas,
            n_k=n_k, x_shape=x_shape, H_out=H_out, W_out=W_out,
            out_pixels=out_pixels, t=t, l=l, offset=offset,
            r_lo=r_lo, r_hi=r_hi, c_lo=c_lo, c_hi=c_hi, dtype=dtype,
        )
        return out.reshape(H_out, W_out)

    ## Batched path: ensure CSR format
    if not scipy.sparse.issparse(x):
        x_csr = scipy.sparse.csr_matrix(x)
    elif not isinstance(x, scipy.sparse.csr_matrix):
        x_csr = x.tocsr()
    else:
        x_csr = x

    return _run_adaptive(
        x_csr=x_csr, k_r=k_r, k_c=k_c, k_d=k_d, k_deltas=k_deltas,
        n_k=n_k, x_shape=x_shape, H_out=H_out, W_out=W_out,
        out_pixels=out_pixels, t=t, l=l, offset=offset,
        r_lo=r_lo, r_hi=r_hi, c_lo=c_lo, c_hi=c_hi, dtype=dtype,
    )


def _run_adaptive(x_csr, k_r, k_c, k_d, k_deltas, n_k, x_shape,
                   H_out, W_out, out_pixels, t, l, offset,
                   r_lo, r_hi, c_lo, c_hi, dtype):
    """
    Adaptive dispatch: choose between counted (two-phase) and direct
    (single-pass over-allocate) strategies based on expected output density.

    When mean scatter ops per pixel < 1, the output is sparse and the
    counting pass saves more (avoiding compaction) than it costs.
    When >= 1, the output is nearly dense, upper bounds are tight, and
    the counting pass is wasted work.
    """
    n_batch = x_csr.shape[0]

    ## Cast CSR arrays to numba-compatible types once
    csr_indptr = x_csr.indptr.astype(np.int64)
    csr_indices = x_csr.indices.astype(np.int64)
    csr_data = x_csr.data.astype(dtype)

    ## Decide strategy based on expected output density
    per_image_nnz = np.diff(csr_indptr)
    mean_scatter_ops = np.mean(per_image_nnz) * n_k if n_batch > 0 else 0

    if mean_scatter_ops < out_pixels:
        ## Sparse output: two-phase (count → exact alloc → scatter)
        return _run_counted(
            csr_indptr, csr_indices, csr_data,
            k_r, k_c, k_d, k_deltas, n_k, x_shape,
            H_out, W_out, out_pixels, t, l, offset,
            r_lo, r_hi, c_lo, c_hi, dtype, n_batch,
        )
    else:
        ## Dense output: single-pass (over-alloc → scatter → compact)
        return _run_overalloc(
            csr_indptr, csr_indices, csr_data,
            per_image_nnz, k_r, k_c, k_d, k_deltas, n_k, x_shape,
            H_out, W_out, out_pixels, t, l, offset,
            r_lo, r_hi, c_lo, c_hi, dtype, n_batch,
        )


def _run_counted(csr_indptr, csr_indices, csr_data,
                  k_r, k_c, k_d, k_deltas, n_k, x_shape,
                  H_out, W_out, out_pixels, t, l, offset,
                  r_lo, r_hi, c_lo, c_hi, dtype, n_batch):
    """Two-phase: count pass → exact-alloc scatter. Best for sparse output."""
    ## Phase 1: lightweight boolean count
    exact_counts = np.empty(n_batch, dtype=np.int64)
    _count_nnz(
        csr_indptr, csr_indices,
        k_r, k_c, k_deltas,
        np.int64(x_shape[1]), np.int64(H_out), np.int64(W_out),
        np.int64(out_pixels), np.int64(t), np.int64(l), offset,
        r_lo, r_hi, c_lo, c_hi,
        np.int64(n_k), exact_counts,
    )

    ## Build exact indptr
    indptr = np.empty(n_batch + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(exact_counts, out=indptr[1:])
    total_nnz = int(indptr[-1])

    if total_nnz == 0:
        return scipy.sparse.csr_matrix(
            (np.empty(0, dtype=dtype), np.empty(0, dtype=np.int32), indptr),
            shape=(n_batch, out_pixels), copy=False,
        )

    ## Phase 2: scatter directly into exact-size arrays
    out_indices = np.empty(total_nnz, dtype=np.int32)
    out_data = np.empty(total_nnz, dtype=dtype)

    _scatter_extract(
        csr_indptr, csr_indices, csr_data,
        k_d, k_r, k_c, k_deltas,
        np.int64(x_shape[1]), np.int64(H_out), np.int64(W_out),
        np.int64(out_pixels), np.int64(t), np.int64(l), offset,
        r_lo, r_hi, c_lo, c_hi,
        indptr, out_indices, out_data,
    )

    return scipy.sparse.csr_matrix(
        (out_data, out_indices, indptr),
        shape=(n_batch, out_pixels), copy=False,
    )


def _run_overalloc(csr_indptr, csr_indices, csr_data,
                    per_image_nnz, k_r, k_c, k_d, k_deltas, n_k, x_shape,
                    H_out, W_out, out_pixels, t, l, offset,
                    r_lo, r_hi, c_lo, c_hi, dtype, n_batch):
    """Single-pass: over-allocate with upper bounds, scatter, compact if needed.
    Best for dense output where upper bound ≈ exact."""
    ## Upper bound per image: min(nnz_i * n_k, out_pixels)
    ub_counts = np.minimum(per_image_nnz * n_k, out_pixels)

    ub_indptr = np.empty(n_batch + 1, dtype=np.int64)
    ub_indptr[0] = 0
    np.cumsum(ub_counts, out=ub_indptr[1:])
    total_ub = int(ub_indptr[-1])

    if total_ub == 0:
        indptr = np.zeros(n_batch + 1, dtype=np.int32)
        return scipy.sparse.csr_matrix(
            (np.empty(0, dtype=dtype), np.empty(0, dtype=np.int32), indptr),
            shape=(n_batch, out_pixels), copy=False,
        )

    ## Allocate over-sized output arrays
    ub_indices = np.empty(total_ub, dtype=np.int32)
    ub_data = np.empty(total_ub, dtype=dtype)
    actual_counts = np.empty(n_batch, dtype=np.int64)

    ## Single-pass scatter + extract + count
    _scatter_extract_counted(
        csr_indptr, csr_indices, csr_data,
        k_d, k_r, k_c, k_deltas,
        np.int64(x_shape[1]), np.int64(H_out), np.int64(W_out),
        np.int64(out_pixels), np.int64(t), np.int64(l), offset,
        r_lo, r_hi, c_lo, c_hi,
        ub_indptr, ub_indices, ub_data, actual_counts,
    )

    ## Build actual indptr
    indptr = np.empty(n_batch + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(actual_counts, out=indptr[1:])
    total_nnz = int(indptr[-1])

    if total_nnz == 0:
        return scipy.sparse.csr_matrix(
            (np.empty(0, dtype=dtype), np.empty(0, dtype=np.int32), indptr),
            shape=(n_batch, out_pixels), copy=False,
        )

    ## Compact only if needed (skip when upper bounds were exact)
    if total_nnz == total_ub:
        indices = ub_indices
        data = ub_data
    else:
        indices = np.empty(total_nnz, dtype=np.int32)
        data = np.empty(total_nnz, dtype=dtype)
        _compact_csr(
            ub_indptr, indptr, actual_counts,
            ub_indices, ub_data, indices, data,
        )

    return scipy.sparse.csr_matrix(
        (data, indices, indptr),
        shape=(n_batch, out_pixels), copy=False,
    )
