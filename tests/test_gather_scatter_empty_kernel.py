import numpy as np
import scipy.sparse

import sparse_convolution as sc


def test_gather_scatter_numpy_all_zero_kernel_returns_empty_sparse_output():
    x = scipy.sparse.csr_matrix(
        ([1.0], ([10], [10])),
        shape=(100, 100),
    )
    k = np.zeros((5, 5), dtype=float)

    conv = sc.Toeplitz_convolution2d(
        x_shape=x.shape,
        k=k,
        mode="same",
        method="gather_scatter",
        backend="numpy",
    )
    out = conv(x=x, batching=False)

    assert scipy.sparse.isspmatrix_csr(out)
    assert out.shape == x.shape
    assert out.nnz == 0
