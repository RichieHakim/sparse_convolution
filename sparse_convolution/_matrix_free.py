"""
Matrix-free 2D convolution operator and backends.

Provides ``MatrixFreeToeplitzConvolution2D``, a ``scipy.sparse.linalg.LinearOperator``
subclass for convolving 2D arrays without materializing the explicit Toeplitz matrix.
Supports both dense inputs (via spectral 2D FFT) and sparse inputs (via coordinate
shift-and-accumulate with 64-bit indices to prevent integer overflow).
"""

from typing import Tuple, Optional, Union
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.signal

from sparse_convolution._utils import (
    compute_output_dims,
    extract_kernel_coo,
    extract_input_coo,
)


def compute_matrix_free_sparse(
    x: Union[scipy.sparse.spmatrix, np.ndarray],
    k: np.ndarray,
    x_shape: Tuple[int, int],
    mode: str,
    batching: bool,
    dtype: np.dtype,
) -> scipy.sparse.csr_matrix:
    """
    Perform matrix-free 2D convolution for sparse inputs via coordinate
    shift-and-accumulate.

    For each nonzero entry in the input (r_in, c_in, v_in) and each nonzero
    in the kernel (dr, dc, w), shifts coordinate to:
        r_out = r_in + dr - t
        c_out = c_in + dc - l
    and accumulates value v_in * w. Uses 64-bit integer arithmetic to prevent
    overflow when NNZ > 2^31 - 1.

    Args:
        x: Sparse input matrix (single 2D or batched).
        k: 2D kernel array.
        x_shape: Spatial dimensions (H_in, W_in).
        mode: 'full', 'same', or 'valid'.
        batching: Whether x contains multiple flattened images.
        dtype: Computation data type.

    Returns:
        scipy.sparse.csr_matrix containing convolved output.
    """
    H_out, W_out, t, l = compute_output_dims(x_shape, k.shape, mode)
    x_data, x_r, x_c, batch_idx, n_batch = extract_input_coo(x, x_shape, batching, dtype)

    if len(x_data) == 0:
        if batching:
            return scipy.sparse.csr_matrix((n_batch, H_out * W_out), dtype=dtype)
        else:
            return scipy.sparse.csr_matrix((H_out, W_out), dtype=dtype)

    kr, kc, kv = extract_kernel_coo(k, dtype)

    min_dr = int(np.min(kr)) - t
    max_dr = int(np.max(kr)) - t
    min_dc = int(np.min(kc)) - l
    max_dc = int(np.max(kc)) - l

    # Interior mask: points guaranteed to remain within [0, H_out) and [0, W_out)
    is_interior = (
        (x_r >= -min_dr) & (x_r < H_out - max_dr) &
        (x_c >= -min_dc) & (x_c < W_out - max_dc)
    )

    r_in = x_r[is_interior]
    c_in = x_c[is_interior]
    d_in = x_data[is_interior]
    b_in = batch_idx[is_interior] if batching else None

    r_bd = x_r[~is_interior]
    c_bd = x_c[~is_interior]
    d_bd = x_data[~is_interior]
    b_bd = batch_idx[~is_interior] if batching else None

    # If input has many nonzeros, chunk to guarantee bounded RAM usage (< 1 GB)
    chunk_size = 25000
    if len(x_data) > chunk_size:
        sub_csrs = []
        for start_idx in range(0, len(x_data), chunk_size):
            end_idx = min(start_idx + chunk_size, len(x_data))
            chunk_r = x_r[start_idx:end_idx]
            chunk_c = x_c[start_idx:end_idx]
            chunk_d = x_data[start_idx:end_idx]
            chunk_b = batch_idx[start_idx:end_idx] if batching else None

            is_int = (
                (chunk_r >= -min_dr) & (chunk_r < H_out - max_dr) &
                (chunk_c >= -min_dc) & (chunk_c < W_out - max_dc)
            )
            r_in_c = chunk_r[is_int]
            c_in_c = chunk_c[is_int]
            d_in_c = chunk_d[is_int]
            b_in_c = chunk_b[is_int] if batching else None

            r_bd_c = chunk_r[~is_int]
            c_bd_c = chunk_c[~is_int]
            d_bd_c = chunk_d[~is_int]
            b_bd_c = chunk_b[~is_int] if batching else None

            all_r_c = []
            all_c_c = []
            all_v_c = []
            all_b_c = [] if batching else None

            for ki in range(len(kv)):
                dr = kr[ki] - t
                dc = kc[ki] - l
                w = kv[ki]

                if len(r_in_c) > 0:
                    all_r_c.append(r_in_c + dr)
                    all_c_c.append(c_in_c + dc)
                    all_v_c.append(d_in_c * w)
                    if batching:
                        all_b_c.append(b_in_c)

                if len(r_bd_c) > 0:
                    rb_s = r_bd_c + dr
                    cb_s = c_bd_c + dc
                    valid = (rb_s >= 0) & (rb_s < H_out) & (cb_s >= 0) & (cb_s < W_out)
                    if np.any(valid):
                        all_r_c.append(rb_s[valid])
                        all_c_c.append(cb_s[valid])
                        all_v_c.append(d_bd_c[valid] * w)
                        if batching:
                            all_b_c.append(b_bd_c[valid])

            if all_r_c:
                out_r_c = np.concatenate(all_r_c)
                out_c_c = np.concatenate(all_c_c)
                out_v_c = np.concatenate(all_v_c)
                del all_r_c, all_c_c, all_v_c

                if batching:
                    out_b_c = np.concatenate(all_b_c)
                    del all_b_c
                    flat_col = out_r_c.astype(np.int64) * W_out + out_c_c.astype(np.int64)
                    sub_csr = scipy.sparse.csr_matrix(
                        (out_v_c, (out_b_c, flat_col)),
                        shape=(n_batch, H_out * W_out),
                        dtype=dtype,
                    )
                else:
                    sub_csr = scipy.sparse.csr_matrix(
                        (out_v_c, (out_r_c, out_c_c)),
                        shape=(H_out, W_out),
                        dtype=dtype,
                    )
                sub_csrs.append(sub_csr)
                del out_r_c, out_c_c, out_v_c

        if not sub_csrs:
            if batching:
                return scipy.sparse.csr_matrix((n_batch, H_out * W_out), dtype=dtype)
            else:
                return scipy.sparse.csr_matrix((H_out, W_out), dtype=dtype)

        res = sub_csrs[0]
        for sub in sub_csrs[1:]:
            res = res + sub
        return res

    all_r = []
    all_c = []
    all_v = []
    all_b = [] if batching else None

    # Stream over kernel nonzeros to bound peak memory usage
    for ki in range(len(kv)):
        dr = kr[ki] - t
        dc = kc[ki] - l
        w = kv[ki]

        # Interior points: always valid, zero bounds-checking overhead
        if len(r_in) > 0:
            all_r.append(r_in + dr)
            all_c.append(c_in + dc)
            all_v.append(d_in * w)
            if batching:
                all_b.append(b_in)

        # Boundary points: check bounds
        if len(r_bd) > 0:
            rb_s = r_bd + dr
            cb_s = c_bd + dc
            valid = (rb_s >= 0) & (rb_s < H_out) & (cb_s >= 0) & (cb_s < W_out)
            if np.any(valid):
                all_r.append(rb_s[valid])
                all_c.append(cb_s[valid])
                all_v.append(d_bd[valid] * w)
                if batching:
                    all_b.append(b_bd[valid])

    if not all_r:
        if batching:
            return scipy.sparse.csr_matrix((n_batch, H_out * W_out), dtype=dtype)
        else:
            return scipy.sparse.csr_matrix((H_out, W_out), dtype=dtype)

    out_r = np.concatenate(all_r)
    out_c = np.concatenate(all_c)
    out_v = np.concatenate(all_v)
    del all_r, all_c, all_v

    if batching:
        out_b = np.concatenate(all_b)
        del all_b
        # Use int64 for flat column index to prevent overflow on large matrices
        flat_col = out_r.astype(np.int64) * W_out + out_c.astype(np.int64)
        out = scipy.sparse.csr_matrix(
            (out_v, (out_b, flat_col)),
            shape=(n_batch, H_out * W_out),
            dtype=dtype,
        )
    else:
        out = scipy.sparse.csr_matrix(
            (out_v, (out_r, out_c)),
            shape=(H_out, W_out),
            dtype=dtype,
        )

    out.sum_duplicates()
    return out


def compute_matrix_free_dense(
    x: np.ndarray,
    k: np.ndarray,
    x_shape: Tuple[int, int],
    mode: str,
    batching: bool,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Perform matrix-free 2D convolution for dense inputs via 2D FFT convolution.
    """
    H_out, W_out, _, _ = compute_output_dims(x_shape, k.shape, mode)

    if batching:
        n_batch = x.shape[0]
        out = np.empty((n_batch, H_out * W_out), dtype=dtype)
        for i in range(n_batch):
            img = x[i].reshape(x_shape)
            res = scipy.signal.fftconvolve(img, k, mode=mode)
            out[i] = res.astype(dtype).ravel()
        return out
    else:
        img = x.reshape(x_shape)
        res = scipy.signal.fftconvolve(img, k, mode=mode)
        return res.astype(dtype)


def compute_matrix_free_adjoint_sparse(
    y: Union[scipy.sparse.spmatrix, np.ndarray],
    k: np.ndarray,
    x_shape: Tuple[int, int],
    mode: str,
    batching: bool,
    dtype: np.dtype,
) -> scipy.sparse.csr_matrix:
    """
    Perform adjoint (transpose) convolution for sparse inputs via coordinate shift.
    """
    H_out, W_out, t, l = compute_output_dims(x_shape, k.shape, mode)
    y_data, y_r, y_c, batch_idx, n_batch = extract_input_coo(y, (H_out, W_out), batching, dtype)

    if len(y_data) == 0:
        if batching:
            return scipy.sparse.csr_matrix((n_batch, x_shape[0] * x_shape[1]), dtype=dtype)
        else:
            return scipy.sparse.csr_matrix(x_shape, dtype=dtype)

    kr, kc, kv = extract_kernel_coo(k, dtype)

    min_dr = int(np.min(kr)) - t
    max_dr = int(np.max(kr)) - t
    min_dc = int(np.min(kc)) - l
    max_dc = int(np.max(kc)) - l

    # Interior mask for adjoint: r_in = y_r - dr in [0, H_in), c_in = y_c - dc in [0, W_in)
    is_interior = (
        (y_r >= max_dr) & (y_r < x_shape[0] + min_dr) &
        (y_c >= max_dc) & (y_c < x_shape[1] + min_dc)
    )

    r_in = y_r[is_interior]
    c_in = y_c[is_interior]
    d_in = y_data[is_interior]
    b_in = batch_idx[is_interior] if batching else None

    r_bd = y_r[~is_interior]
    c_bd = y_c[~is_interior]
    d_bd = y_data[~is_interior]
    b_bd = batch_idx[~is_interior] if batching else None

    all_r = []
    all_c = []
    all_v = []
    all_b = [] if batching else None

    # Adjoint relation: r_in = r_out - dr, c_in = c_out - dc
    for ki in range(len(kv)):
        dr = kr[ki] - t
        dc = kc[ki] - l
        w = kv[ki]

        # Interior points: zero bounds check
        if len(r_in) > 0:
            all_r.append(r_in - dr)
            all_c.append(c_in - dc)
            all_v.append(d_in * w)
            if batching:
                all_b.append(b_in)

        # Boundary points
        if len(r_bd) > 0:
            rb_in = r_bd - dr
            cb_in = c_bd - dc
            valid = (rb_in >= 0) & (rb_in < x_shape[0]) & (cb_in >= 0) & (cb_in < x_shape[1])
            if np.any(valid):
                all_r.append(rb_in[valid])
                all_c.append(cb_in[valid])
                all_v.append(d_bd[valid] * w)
                if batching:
                    all_b.append(b_bd[valid])

    if not all_r:
        if batching:
            return scipy.sparse.csr_matrix((n_batch, x_shape[0] * x_shape[1]), dtype=dtype)
        else:
            return scipy.sparse.csr_matrix(x_shape, dtype=dtype)

    out_r = np.concatenate(all_r)
    out_c = np.concatenate(all_c)
    out_v = np.concatenate(all_v)
    del all_r, all_c, all_v

    if batching:
        out_b = np.concatenate(all_b)
        del all_b
        flat_col = out_r.astype(np.int64) * x_shape[1] + out_c.astype(np.int64)
        out = scipy.sparse.csr_matrix(
            (out_v, (out_b, flat_col)),
            shape=(n_batch, x_shape[0] * x_shape[1]),
            dtype=dtype,
        )
    else:
        out = scipy.sparse.csr_matrix(
            (out_v, (out_r, out_c)),
            shape=x_shape,
            dtype=dtype,
        )

    out.sum_duplicates()
    return out


def compute_matrix_free_adjoint_dense(
    y: np.ndarray,
    k: np.ndarray,
    x_shape: Tuple[int, int],
    mode: str,
    batching: bool,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Perform adjoint (transpose) convolution for dense inputs.
    """
    H_out, W_out, t, l = compute_output_dims(x_shape, k.shape, mode)
    kh, kw = k.shape
    k_rot = k[::-1, ::-1]

    def _adjoint_single(img_y):
        if mode == 'full':
            res = scipy.signal.fftconvolve(img_y, k_rot, mode='valid')
        elif mode == 'valid':
            res = scipy.signal.fftconvolve(img_y, k_rot, mode='full')
        elif mode == 'same':
            # Embed y into full output space, then convolve with k_rot in 'valid' mode
            y_embed = np.zeros((x_shape[0] + kh - 1, x_shape[1] + kw - 1), dtype=dtype)
            y_embed[t : t + H_out, l : l + W_out] = img_y
            res = scipy.signal.fftconvolve(y_embed, k_rot, mode='valid')
        return res.astype(dtype)

    if batching:
        n_batch = y.shape[0]
        out = np.empty((n_batch, x_shape[0] * x_shape[1]), dtype=dtype)
        for i in range(n_batch):
            img = y[i].reshape((H_out, W_out))
            out[i] = _adjoint_single(img).ravel()
        return out
    else:
        img = y.reshape((H_out, W_out))
        return _adjoint_single(img)


class MatrixFreeToeplitzConvolution2D(scipy.sparse.linalg.LinearOperator):
    """
    Matrix-Free 2D Convolution LinearOperator for scipy.sparse.linalg.

    Represents the 2D convolution with kernel ``k`` as a linear mapping
    A : R^{N_in} -> R^{M_out}, where N_in = H_in * W_in and M_out = H_out * W_out,
    without materializing the gigantic Toeplitz matrix in memory.

    Can be passed directly to iterative solvers (e.g. ``scipy.sparse.linalg.cg``,
    ``gmres``, ``lsqr``, ``eigs``).

    Args:
        x_shape (Tuple[int, int]):
            Spatial dimensions ``(H, W)`` of the input.
        k (np.ndarray):
            2D convolution kernel.
        mode (str):
            Convolution mode: ``'full'``, ``'same'``, or ``'valid'``.
        dtype (Optional[np.dtype]):
            Data type for computation. Defaults to kernel's dtype.
    """

    def __init__(
        self,
        x_shape: Tuple[int, int],
        k: np.ndarray,
        mode: str = 'same',
        dtype: Optional[np.dtype] = None,
    ):
        assert isinstance(x_shape, (tuple, list)) and len(x_shape) == 2, \
            f"x_shape must be a tuple of length 2. Found: {x_shape}"
        self.x_shape = (int(x_shape[0]), int(x_shape[1]))

        assert isinstance(k, np.ndarray) and k.ndim == 2, "k must be a 2D numpy array"
        self.k = k.copy()

        assert mode in ('full', 'same', 'valid'), \
            f"mode must be 'full', 'same', or 'valid'. Got: {mode!r}"
        if mode == 'valid':
            assert self.x_shape[0] >= self.k.shape[0] and self.x_shape[1] >= self.k.shape[1], \
                "x must be larger than k in both dimensions for mode='valid'"
        self.mode = mode

        dt = k.dtype if dtype is None else np.dtype(dtype)
        H_out, W_out, t, l = compute_output_dims(self.x_shape, self.k.shape, self.mode)
        self.out_shape = (H_out, W_out)
        self.t = t
        self.l = l

        M_out = H_out * W_out
        N_in = self.x_shape[0] * self.x_shape[1]

        super().__init__(dtype=dt, shape=(M_out, N_in))

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Matrix-vector multiplication A @ v for a 1D vector v in R^{N_in}.
        """
        v_2d = v.reshape(self.x_shape)
        res = scipy.signal.fftconvolve(v_2d, self.k, mode=self.mode)
        return res.astype(self.dtype).ravel()

    def _rmatvec(self, v: np.ndarray) -> np.ndarray:
        """
        Adjoint matrix-vector multiplication A^T @ v for a 1D vector v in R^{M_out}.
        """
        v_2d = v.reshape(self.out_shape)
        res = compute_matrix_free_adjoint_dense(
            v_2d, self.k, self.x_shape, self.mode, batching=False, dtype=self.dtype
        )
        return res.ravel()

    def _matmat(self, V: np.ndarray) -> np.ndarray:
        """
        Multi-vector multiplication A @ V for matrix V in R^{N_in x B}.
        """
        B = V.shape[1]
        out = np.empty((self.shape[0], B), dtype=self.dtype)
        for j in range(B):
            out[:, j] = self._matvec(V[:, j])
        return out

    def _rmatmat(self, V: np.ndarray) -> np.ndarray:
        """
        Adjoint multi-vector multiplication A^T @ V for matrix V in R^{M_out x B}.
        """
        B = V.shape[1]
        out = np.empty((self.shape[1], B), dtype=self.dtype)
        for j in range(B):
            out[:, j] = self._rmatvec(V[:, j])
        return out

    def __call__(
        self,
        x: Union[np.ndarray, scipy.sparse.spmatrix],
        batching: bool = False,
        mode: Optional[str] = None,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """
        Convolve input array (dense or sparse, 2D or batched).
        """
        if mode is None:
            mode = self.mode

        issparse = scipy.sparse.issparse(x)
        if issparse:
            return compute_matrix_free_sparse(
                x=x, k=self.k, x_shape=self.x_shape,
                mode=mode, batching=batching, dtype=self.dtype
            )
        else:
            return compute_matrix_free_dense(
                x=np.asarray(x), k=self.k, x_shape=self.x_shape,
                mode=mode, batching=batching, dtype=self.dtype
            )
