"""
Tests for parity between all convolution methods ('lazy', 'precomputed',
'gather_scatter') and backends ('numpy', 'numba', 'torch'). Validates that
all method+backend combinations produce the same output across modes,
densities, and batching configurations.
"""

import pytest
import numpy as np
import scipy.sparse

from sparse_convolution import Toeplitz_convolution2d


## ---------------------------------------------------------------------------
## Fixtures
## ---------------------------------------------------------------------------

@pytest.fixture(params=[0.001, 0.01, 0.1])
def density(request):
    return request.param

@pytest.fixture(params=['same', 'full', 'valid'])
def mode(request):
    return request.param

@pytest.fixture(params=[True, False])
def batching(request):
    return request.param


## ---------------------------------------------------------------------------
## Helpers
## ---------------------------------------------------------------------------

def _make_sparse_input(x_shape, density, batching, seed=42):
    """
    Generate a sparse input array for testing.

    Returns:
        (scipy.sparse.csr_matrix): Sparse input with the given density.
    """
    np.random.seed(seed)
    if batching:
        batch_size = 10
        total_elements = batch_size * x_shape[0] * x_shape[1]
        num_non_zero = max(1, int(total_elements * density))
        x_dense = np.zeros((batch_size, x_shape[0] * x_shape[1]), dtype=np.float64)
        indices = np.random.choice(total_elements, num_non_zero, replace=False)
        x_dense.flat[indices] = np.random.rand(num_non_zero)
    else:
        total_elements = x_shape[0] * x_shape[1]
        num_non_zero = max(1, int(total_elements * density))
        x_dense = np.zeros(x_shape, dtype=np.float64)
        indices = np.random.choice(total_elements, num_non_zero, replace=False)
        x_dense.flat[indices] = np.random.rand(num_non_zero)
    return scipy.sparse.csr_matrix(x_dense)


def _compare_outputs(out_a, out_b, label_a, label_b, mode, density, batching, tol=1e-10):
    """
    Convert outputs to dense and compare element-wise.
    """
    if scipy.sparse.issparse(out_a):
        out_a = out_a.toarray()
    if scipy.sparse.issparse(out_b):
        out_b = out_b.toarray()

    diff = np.abs(out_a - out_b).max()
    assert diff < tol, (
        f"{label_a} vs {label_b} disagree: max diff={diff:.2e} "
        f"(mode={mode}, density={density}, batching={batching})"
    )


## ---------------------------------------------------------------------------
## Method parity tests (numpy backend, baseline)
## ---------------------------------------------------------------------------

def test_lazy_vs_precomputed_parity(density, mode, batching):
    """
    Compare 'lazy' and 'precomputed' methods (both numpy backend) for
    identical output.
    """
    x_shape = (50, 50)
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_sparse = _make_sparse_input(x_shape, density, batching)

    conv_lazy = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='lazy', backend='numpy',
    )
    conv_precomputed = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='precomputed', backend='numpy',
    )

    out_lazy = conv_lazy(x=x_sparse, batching=batching)
    out_precomputed = conv_precomputed(x=x_sparse, batching=batching)
    _compare_outputs(out_lazy, out_precomputed, 'lazy', 'precomputed', mode, density, batching)


def test_gather_scatter_vs_precomputed_parity(density, mode, batching):
    """
    Compare 'gather_scatter' and 'precomputed' methods for identical output.
    """
    x_shape = (50, 50)
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_sparse = _make_sparse_input(x_shape, density, batching)

    conv_gs = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='gather_scatter',
    )
    conv_precomputed = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='precomputed',
    )

    out_gs = conv_gs(x=x_sparse, batching=batching)
    out_precomputed = conv_precomputed(x=x_sparse, batching=batching)
    _compare_outputs(out_gs, out_precomputed, 'gather_scatter', 'precomputed', mode, density, batching)


## ---------------------------------------------------------------------------
## Torch backend parity tests (torch vs numpy for each method)
## ---------------------------------------------------------------------------

@pytest.fixture(params=['same', 'full', 'valid'])
def mode_torch(request):
    return request.param


@pytest.fixture(params=[True, False])
def batching_torch(request):
    return request.param


def _has_torch():
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_precomputed_torch_vs_numpy(mode_torch, batching_torch):
    """
    Compare precomputed+torch vs precomputed+numpy for identical output.
    Uses density=0.05 as a representative case.
    """
    x_shape = (40, 40)
    density = 0.05
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_sparse = _make_sparse_input(x_shape, density, batching_torch)

    conv_numpy = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode_torch,
        method='precomputed', backend='numpy',
    )
    conv_torch = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode_torch,
        method='precomputed', backend='torch', device='cpu',
    )

    out_numpy = conv_numpy(x=x_sparse, batching=batching_torch)
    out_torch = conv_torch(x=x_sparse, batching=batching_torch)
    _compare_outputs(
        out_numpy, out_torch,
        'precomputed+numpy', 'precomputed+torch',
        mode_torch, density, batching_torch,
    )


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_lazy_torch_vs_numpy(mode_torch, batching_torch):
    """
    Compare lazy+torch vs lazy+numpy for identical output.
    """
    x_shape = (40, 40)
    density = 0.05
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_sparse = _make_sparse_input(x_shape, density, batching_torch)

    conv_numpy = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode_torch,
        method='lazy', backend='numpy',
    )
    conv_torch = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode_torch,
        method='lazy', backend='torch', device='cpu',
    )

    out_numpy = conv_numpy(x=x_sparse, batching=batching_torch)
    out_torch = conv_torch(x=x_sparse, batching=batching_torch)
    _compare_outputs(
        out_numpy, out_torch,
        'lazy+numpy', 'lazy+torch',
        mode_torch, density, batching_torch,
    )


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_gather_scatter_torch_vs_numpy(mode_torch, batching_torch):
    """
    Compare gather_scatter+torch vs gather_scatter+numpy for identical output.
    """
    x_shape = (40, 40)
    density = 0.05
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_sparse = _make_sparse_input(x_shape, density, batching_torch)

    conv_numpy = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode_torch,
        method='gather_scatter', backend='numpy',
    )
    conv_torch = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode_torch,
        method='gather_scatter', backend='torch', device='cpu',
    )

    out_numpy = conv_numpy(x=x_sparse, batching=batching_torch)
    out_torch = conv_torch(x=x_sparse, batching=batching_torch)
    _compare_outputs(
        out_numpy, out_torch,
        'gather_scatter+numpy', 'gather_scatter+torch',
        mode_torch, density, batching_torch,
    )


## ---------------------------------------------------------------------------
## Backend-specific tests
## ---------------------------------------------------------------------------

def _has_numba():
    try:
        import numba
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
def test_precomputed_numba_vs_numpy(mode, batching):
    """
    Compare precomputed+numba (parallel CSR matvec) vs precomputed+numpy
    (scipy C matmul) for identical output.
    """
    x_shape = (50, 50)
    density = 0.05
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_sparse = _make_sparse_input(x_shape, density, batching)

    conv_numpy = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode,
        method='precomputed', backend='numpy',
    )
    conv_numba = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode,
        method='precomputed', backend='numba',
    )

    out_numpy = conv_numpy(x=x_sparse, batching=batching)
    out_numba = conv_numba(x=x_sparse, batching=batching)
    _compare_outputs(
        out_numpy, out_numba,
        'precomputed+numpy', 'precomputed+numba',
        mode, density, batching,
    )


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
def test_gather_scatter_numba_vs_numpy(mode, batching):
    """
    Compare gather_scatter+numba vs gather_scatter+numpy for identical output.
    """
    x_shape = (40, 40)
    density = 0.05
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_sparse = _make_sparse_input(x_shape, density, batching)

    conv_numpy = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode,
        method='gather_scatter', backend='numpy',
    )
    conv_numba = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode,
        method='gather_scatter', backend='numba',
    )

    out_numpy = conv_numpy(x=x_sparse, batching=batching)
    out_numba = conv_numba(x=x_sparse, batching=batching)
    _compare_outputs(
        out_numpy, out_numba,
        'gather_scatter+numpy', 'gather_scatter+numba',
        mode, density, batching,
    )


## ---------------------------------------------------------------------------
## Dense input tests (non-sparse inputs)
## ---------------------------------------------------------------------------

def test_dense_input_precomputed(mode):
    """
    Verify precomputed method works correctly with dense (non-sparse) input.
    """
    x_shape = (20, 20)
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_dense = np.random.rand(*x_shape)

    conv = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='precomputed',
    )

    out = conv(x=x_dense, batching=False)
    assert isinstance(out, np.ndarray), f"Expected np.ndarray, got {type(out)}"

    ## Compare with scipy.signal.convolve2d
    import scipy.signal
    out_ref = scipy.signal.convolve2d(x_dense, kernel, mode=mode)
    diff = np.abs(out - out_ref).max()
    assert diff < 1e-10, f"Dense precomputed vs scipy.signal: max diff={diff:.2e}"


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_dense_input_precomputed_torch(mode):
    """
    Verify precomputed+torch works correctly with dense input.
    """
    x_shape = (20, 20)
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_dense = np.random.rand(*x_shape)

    conv = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode,
        method='precomputed', backend='torch', device='cpu',
    )

    out = conv(x=x_dense, batching=False)
    assert isinstance(out, np.ndarray), f"Expected np.ndarray, got {type(out)}"

    import scipy.signal
    out_ref = scipy.signal.convolve2d(x_dense, kernel, mode=mode)
    diff = np.abs(out - out_ref).max()
    assert diff < 1e-10, f"Dense precomputed+torch vs scipy.signal: max diff={diff:.2e}"
