import numpy as np
import scipy.sparse

def compute_broadcasting(x, k, x_shape, mode, batching, dtype):
    """
    Computes convolution using optimized COO broadcasting.
    """
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

    k_coo = scipy.sparse.coo_matrix(k)
    if k_coo.dtype != dtype:
        k_coo = k_coo.astype(dtype)
    k_r = k_coo.row
    k_c = k_coo.col
    k_d = k_coo.data

    x_coo = scipy.sparse.coo_matrix(x)
    
    if batching:
        B = x.shape[0]
        batch_idx = x_coo.row
        x_idx = x_coo.col
        x_r = x_idx // x_shape[1]
        x_c = x_idx % x_shape[1]
    else:
        batch_idx = np.zeros_like(x_coo.row)
        x_r = x_coo.row
        x_c = x_coo.col
    
    # Broadcasting logic
    new_r = x_r[:, None] + k_r[None, :] - t
    new_c = x_c[:, None] + k_c[None, :] - l
    new_b = np.repeat(batch_idx[:, None], len(k_d), axis=1)
    new_d = x_coo.data[:, None] * k_d[None, :]

    # Ravel for valid mask
    new_r = new_r.ravel()
    new_c = new_c.ravel()
    new_b = new_b.ravel()
    new_d = new_d.ravel()

    # Filter out-of-bounds bounds
    valid_mask = (new_r >= 0) & (new_r < H_out) & (new_c >= 0) & (new_c < W_out)
    
    new_r = new_r[valid_mask]
    new_c = new_c[valid_mask]
    new_b = new_b[valid_mask]
    new_d = new_d[valid_mask]

    if batching:
        out_idx = new_r * W_out + new_c
        out = scipy.sparse.csr_matrix(
            (new_d, (new_b, out_idx)),
            shape=(B, H_out * W_out),
            dtype=dtype
        )
    else:
        out = scipy.sparse.csr_matrix(
            (new_d, (new_r, new_c)),
            shape=(H_out, W_out),
            dtype=dtype
        )

    return out
