"""
Tests for parity between the 'lazy' and 'precomputed' convolution methods.
Validates that both methods produce the same output across modes, densities,
and batching configurations.
"""

import pytest
import numpy as np
import scipy.sparse

from sparse_convolution import Toeplitz_convolution2d


@pytest.fixture(params=[0.001, 0.01, 0.1])
def density(request):
    return request.param

@pytest.fixture(params=['same', 'full', 'valid'])
def mode(request):
    return request.param

@pytest.fixture(params=[True, False])
def batching(request):
    return request.param


def test_lazy_vs_precomputed_parity(density, mode, batching):
    """
    Compare 'lazy' and 'precomputed' methods for identical output across
    various sparsity levels, convolution modes, and batching settings.
    """
    x_shape = (50, 50)
    k_shape = (5, 5)

    np.random.seed(42)
    kernel = np.random.rand(*k_shape).astype(np.float64)

    ## Generate sparse input
    if batching:
        batch_size = 10
        total_elements = batch_size * x_shape[0] * x_shape[1]
        num_non_zero = max(1, int(total_elements * density))
        x_dense = np.zeros((batch_size, x_shape[0] * x_shape[1]), dtype=np.float64)
        indices = np.random.choice(total_elements, num_non_zero, replace=False)
        x_dense.flat[indices] = np.random.rand(num_non_zero)
        x_sparse = scipy.sparse.csr_matrix(x_dense)
    else:
        total_elements = x_shape[0] * x_shape[1]
        num_non_zero = max(1, int(total_elements * density))
        x_dense = np.zeros(x_shape, dtype=np.float64)
        indices = np.random.choice(total_elements, num_non_zero, replace=False)
        x_dense.flat[indices] = np.random.rand(num_non_zero)
        x_sparse = scipy.sparse.csr_matrix(x_dense)

    ## Run both methods
    conv_lazy = Toeplitz_convolution2d(x_shape=x_shape, k=kernel, mode=mode, method='lazy')
    conv_precomputed = Toeplitz_convolution2d(x_shape=x_shape, k=kernel, mode=mode, method='precomputed')

    out_lazy = conv_lazy(x=x_sparse, batching=batching)
    out_precomputed = conv_precomputed(x=x_sparse, batching=batching)

    ## Convert to dense for comparison
    if scipy.sparse.issparse(out_lazy):
        out_lazy = out_lazy.toarray()
    if scipy.sparse.issparse(out_precomputed):
        out_precomputed = out_precomputed.toarray()

    diff = np.abs(out_lazy - out_precomputed).max()
    assert diff < 1e-10, (
        f"Methods disagree: max diff={diff:.2e} "
        f"(mode={mode}, density={density}, batching={batching})"
    )
