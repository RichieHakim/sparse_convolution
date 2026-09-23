"""
Tests for parity between all convolution methods ('lazy', 'precomputed',
'gather_scatter') and backends ('numpy', 'numba', 'torch'). Validates that
all method+backend combinations produce the same output across modes,
densities, and batching configurations.
"""

import pytest
import numpy as np
import scipy.sparse
import scipy.signal

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
## Direct method parity tests (numba only)
## ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
def test_direct_vs_precomputed_parity(density, mode, batching):
    """
    Compare 'direct' and 'precomputed' methods for identical output.
    """
    x_shape = (50, 50)
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_sparse = _make_sparse_input(x_shape, density, batching)

    conv_direct = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='direct',
    )
    conv_precomputed = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='precomputed', backend='numpy',
    )

    out_direct = conv_direct(x=x_sparse, batching=batching)
    out_precomputed = conv_precomputed(x=x_sparse, batching=batching)
    _compare_outputs(out_direct, out_precomputed, 'direct', 'precomputed', mode, density, batching)


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
def test_direct_dense_input(mode):
    """
    Verify 'direct' method works correctly with dense (non-sparse) input.
    """
    x_shape = (20, 20)
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_dense = np.random.rand(*x_shape)

    conv = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='direct',
    )
    out = conv(x=x_dense, batching=False)
    assert isinstance(out, np.ndarray), f"Expected np.ndarray, got {type(out)}"


    out_ref = scipy.signal.convolve2d(x_dense, kernel, mode=mode)
    diff = np.abs(out - out_ref).max()
    assert diff < 1e-10, f"Dense direct vs scipy.signal: max diff={diff:.2e}"


## ---------------------------------------------------------------------------
## Direct method: exhaustive shape + edge case tests
## ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
@pytest.mark.parametrize("x_shape,k_shape", [
    ## Non-square images
    ((80, 20), (5, 5)),
    ((20, 80), (5, 5)),
    ((100, 10), (3, 3)),
    ((10, 100), (3, 3)),
    ## Non-square kernels
    ((50, 50), (1, 5)),
    ((50, 50), (5, 1)),
    ((50, 50), (3, 7)),
    ((50, 50), (7, 3)),
    ## Tiny kernel
    ((50, 50), (1, 1)),
    ((30, 40), (1, 1)),
    ## Large kernel
    ((50, 50), (15, 15)),
    ((60, 60), (11, 13)),
    ## Kernel same size as image (valid mode produces 1×1)
    ((10, 10), (10, 10)),
    ## Minimal image
    ((5, 5), (3, 3)),
    ((3, 3), (1, 1)),
    ((2, 2), (2, 2)),
])
def test_direct_vs_scipy_shapes(x_shape, k_shape, mode, batching):
    """
    Compare direct+numba against scipy.signal.convolve2d across a variety of
    non-square image/kernel shapes, all modes, and batching states.
    """
    if mode == 'valid' and (x_shape[0] < k_shape[0] or x_shape[1] < k_shape[1]):
        pytest.skip("x must be >= k for mode='valid'")

    np.random.seed(42)
    kernel = np.random.rand(*k_shape).astype(np.float64)

    if batching:
        batch_size = 5
        x_dense_batch = np.random.rand(batch_size, x_shape[0] * x_shape[1]) * 0.1
        ## Make it sparse (~10% nonzero)
        mask = np.random.rand(*x_dense_batch.shape) > 0.1
        x_dense_batch[mask] = 0.0
        x_sparse = scipy.sparse.csr_matrix(x_dense_batch)

        conv = Toeplitz_convolution2d(
            x_shape=x_shape, k=kernel, mode=mode, method='direct',
        )
        out = conv(x=x_sparse, batching=True)
        if scipy.sparse.issparse(out):
            out = out.toarray()

    
        for i in range(batch_size):
            ref = scipy.signal.convolve2d(
                x_dense_batch[i].reshape(x_shape), kernel, mode=mode,
            )
            diff = np.abs(out[i].reshape(ref.shape) - ref).max()
            assert diff < 1e-10, (
                f"direct vs scipy.signal: max diff={diff:.2e} "
                f"(shape={x_shape}, k={k_shape}, mode={mode}, batch_idx={i})"
            )
    else:
        x_dense = np.random.rand(*x_shape) * 0.1
        mask = np.random.rand(*x_dense.shape) > 0.1
        x_dense[mask] = 0.0

        conv = Toeplitz_convolution2d(
            x_shape=x_shape, k=kernel, mode=mode, method='direct',
        )
        out = conv(x=x_dense, batching=False)
        if scipy.sparse.issparse(out):
            out = out.toarray()

    
        ref = scipy.signal.convolve2d(x_dense, kernel, mode=mode)
        diff = np.abs(out - ref).max()
        assert diff < 1e-10, (
            f"direct vs scipy.signal: max diff={diff:.2e} "
            f"(shape={x_shape}, k={k_shape}, mode={mode})"
        )


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
@pytest.mark.parametrize("density", [0.0001, 0.3, 0.5, 0.9])
def test_direct_vs_precomputed_extreme_densities(density, mode, batching):
    """
    Test direct method at extreme densities (very sparse and very dense)
    against precomputed baseline.
    """
    x_shape = (40, 40)
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_sparse = _make_sparse_input(x_shape, density, batching)

    conv_direct = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='direct',
    )
    conv_precomputed = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='precomputed', backend='numpy',
    )

    out_direct = conv_direct(x=x_sparse, batching=batching)
    out_precomputed = conv_precomputed(x=x_sparse, batching=batching)
    _compare_outputs(
        out_direct, out_precomputed, 'direct', 'precomputed',
        mode, density, batching,
    )


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
@pytest.mark.parametrize("batch_size", [1, 2, 50, 200])
def test_direct_various_batch_sizes(batch_size, mode):
    """
    Test direct method with various batch sizes against scipy.signal.
    """
    x_shape = (30, 30)
    density = 0.05
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)

    total_elements = batch_size * x_shape[0] * x_shape[1]
    num_non_zero = max(1, int(total_elements * density))
    x_dense = np.zeros((batch_size, x_shape[0] * x_shape[1]), dtype=np.float64)
    indices = np.random.choice(total_elements, num_non_zero, replace=False)
    x_dense.flat[indices] = np.random.rand(num_non_zero)
    x_sparse = scipy.sparse.csr_matrix(x_dense)

    conv = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='direct',
    )
    out = conv(x=x_sparse, batching=True)
    if scipy.sparse.issparse(out):
        out = out.toarray()


    for i in range(batch_size):
        ref = scipy.signal.convolve2d(
            x_dense[i].reshape(x_shape), kernel, mode=mode,
        )
        diff = np.abs(out[i].reshape(ref.shape) - ref).max()
        assert diff < 1e-10, (
            f"direct vs scipy.signal: max diff={diff:.2e} "
            f"(batch_size={batch_size}, mode={mode}, batch_idx={i})"
        )


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
def test_direct_csc_input(mode, batching):
    """
    Verify direct method handles CSC format input (auto-converts to CSR).
    """
    x_shape = (40, 40)
    density = 0.05
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_csr = _make_sparse_input(x_shape, density, batching)
    x_csc = x_csr.tocsc()

    conv = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='direct',
    )
    conv_ref = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='precomputed', backend='numpy',
    )

    out_direct = conv(x=x_csc, batching=batching)
    out_ref = conv_ref(x=x_csr, batching=batching)
    _compare_outputs(
        out_direct, out_ref, 'direct(csc)', 'precomputed(csr)',
        mode, density, batching,
    )


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
def test_direct_float32(mode, batching):
    """
    Verify direct method produces correct results with float32 dtype.
    """
    x_shape = (40, 40)
    density = 0.05
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float32)
    x_sparse = _make_sparse_input(x_shape, density, batching)

    conv_direct = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='direct',
    )
    conv_ref = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='precomputed', backend='numpy',
    )

    out_direct = conv_direct(x=x_sparse, batching=batching)
    out_ref = conv_ref(x=x_sparse, batching=batching)
    _compare_outputs(
        out_direct, out_ref, 'direct(f32)', 'precomputed(f64)',
        mode, density, batching, tol=1e-5,
    )


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
def test_direct_single_nonzero_pixel(mode):
    """
    Edge case: single nonzero pixel in the input.
    """
    x_shape = (20, 20)
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_dense = np.zeros(x_shape, dtype=np.float64)
    x_dense[10, 10] = 1.0

    conv = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='direct',
    )
    out = conv(x=x_dense, batching=False)
    if scipy.sparse.issparse(out):
        out = out.toarray()


    ref = scipy.signal.convolve2d(x_dense, kernel, mode=mode)
    diff = np.abs(out - ref).max()
    assert diff < 1e-10, f"Single-pixel direct vs scipy.signal: max diff={diff:.2e}"


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
def test_direct_all_zeros_input(mode):
    """
    Edge case: all-zero input should produce all-zero output.
    """
    x_shape = (20, 20)
    np.random.seed(42)
    kernel = np.random.rand(5, 5).astype(np.float64)
    x_dense = np.zeros(x_shape, dtype=np.float64)

    conv = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode=mode, method='direct',
    )
    out = conv(x=x_dense, batching=False)
    if scipy.sparse.issparse(out):
        out = out.toarray()

    assert np.all(out == 0.0), f"All-zero input should give all-zero output"


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
def test_direct_exhaustive_small_shapes():
    """
    Exhaustive test of all x_shape (1-6)×(1-6) and k_shape (1-6)×(1-6)
    against scipy.signal.convolve2d. Tests batching=False and batching=True.
    """


    np.random.seed(42)
    shapes = np.meshgrid(
        np.arange(1, 7), np.arange(1, 7),
        np.arange(1, 7), np.arange(1, 7),
    )
    shapes = [s.reshape(-1) for s in shapes]

    for mode in ['full', 'same', 'valid']:
        for ii in range(len(shapes[0])):
            xh, xw, kh, kw = shapes[0][ii], shapes[1][ii], shapes[2][ii], shapes[3][ii]

            if mode == 'valid' and (xh < kh or xw < kw):
                continue

            x = np.random.rand(xh, xw)
            k = np.random.rand(kh, kw)
            ref = scipy.signal.convolve2d(x, k, mode=mode)

            ## batching=False
            conv = Toeplitz_convolution2d(
                x_shape=(xh, xw), k=k, mode=mode, method='direct',
            )
            out = conv(x=x, batching=False)
            if scipy.sparse.issparse(out):
                out = out.toarray()
            assert np.allclose(out, ref), (
                f"direct batching=False: x=({xh},{xw}), k=({kh},{kw}), mode={mode}"
            )

            ## batching=True (3 images)
            x_batch = np.stack([np.random.rand(xh, xw).reshape(-1) for _ in range(3)])
            refs = np.stack([
                scipy.signal.convolve2d(x_batch[j].reshape(xh, xw), k, mode=mode).reshape(-1)
                for j in range(3)
            ])
            out_batch = conv(x=x_batch, batching=True)
            if scipy.sparse.issparse(out_batch):
                out_batch = out_batch.toarray()
            assert np.allclose(out_batch, refs), (
                f"direct batching=True: x=({xh},{xw}), k=({kh},{kw}), mode={mode}"
            )


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
@pytest.mark.parametrize("k_shape,k_offset_support", [((5, 5), False), ((7, 4), True)])
@pytest.mark.parametrize("n_dense", [0, 4])
def test_direct_localized_blobs(mode, k_shape, k_offset_support, n_dense):
    """
    One small blob per image at random positions, the frame corners, plus
    empty images. Exercises the per-image output bounding box: clipping at
    the frame edges and kernels whose nonzero support is offset. ``n_dense``
    fully dense images push the batch onto the over-allocate path.
    """
    x_shape = (60, 45)
    rng = np.random.default_rng(42)
    kernel = rng.random(k_shape)
    if k_offset_support:
        kernel[:2] = 0  ## support starts at row 2
        kernel[:, -1] = 0  ## and ends one column early

    h, w = x_shape[0] - 3, x_shape[1] - 3
    corners = [(0, 0), (h, 0), (0, w), (h, w)]
    corners += [tuple(rng.integers(0, (h, w))) for _ in range(20)]
    x_dense = np.zeros((len(corners) + 2 + n_dense, *x_shape))  ## (n_images, H, W)
    for i, (r, c) in enumerate(corners):
        x_dense[i, r:r + 3, c:c + 3] = rng.random((3, 3)) * (rng.random((3, 3)) > 0.3)
    x_dense[len(corners) + 2:] = rng.random((n_dense, *x_shape))  ## after 2 empty
    x = scipy.sparse.csr_matrix(x_dense.reshape(len(x_dense), -1))

    conv = Toeplitz_convolution2d(x_shape=x_shape, k=kernel, mode=mode, method='direct')
    out = conv(x=x, batching=True)
    ref = np.stack([
        scipy.signal.convolve2d(x_i, kernel, mode=mode).reshape(-1) for x_i in x_dense
    ])
    is_overalloc = x.getnnz(axis=1).mean() * np.count_nonzero(kernel) >= out.shape[1]
    assert is_overalloc == (n_dense > 0), "batch is on the wrong dispatch path"
    assert out.has_sorted_indices
    diff = np.abs(out.toarray() - ref).max()
    assert diff < 1e-10, f"max diff={diff:.2e} (k={k_shape}, n_dense={n_dense})"


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
@pytest.mark.parametrize(
    "x_shape,dtype_idx", [((500, 700), np.int32), ((50_000, 50_000), np.int64)],
)
def test_direct_index_dtype(x_shape, dtype_idx):
    """
    Frames with more than 2**31 pixels need int64 output indices. Small
    blobs keep the bounding-box buffers (and this test) cheap.
    """
    rng = np.random.default_rng(42)
    kernel = rng.random((5, 5))
    blob = rng.random((4, 6))
    t = l = 2  ## 'same' mode offsets for a 5x5 kernel
    H, W = x_shape
    corners = [(0, 0), (H // 2, W // 3), (H - 4, W - 6)]

    ## Build the input without a dense frame
    rr, cc = np.meshgrid(np.arange(4), np.arange(6), indexing='ij')  ## shape: (4, 6)
    idx_row = np.repeat(np.arange(len(corners)), blob.size)
    idx_col = np.concatenate([((rr + r) * W + cc + c).ravel() for r, c in corners])
    x = scipy.sparse.csr_matrix(
        (np.tile(blob.ravel(), len(corners)), (idx_row, idx_col)),
        shape=(len(corners), H * W),
    )

    conv = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode='same', method='direct',
    )
    out = conv(x=x, batching=True)
    assert out.indices.dtype == dtype_idx and out.indptr.dtype == dtype_idx

    ## Reference: the blob's full convolution placed at (r - t, c - l), clipped
    full = scipy.signal.convolve2d(blob, kernel, mode='full')  ## shape: (8, 10)
    pp, qq = np.meshgrid(*[np.arange(n) for n in full.shape], indexing='ij')
    for i, (r, c) in enumerate(corners):
        out_r, out_c = (pp + r - t).ravel(), (qq + c - l).ravel()
        inside = (out_r >= 0) & (out_r < H) & (out_c >= 0) & (out_c < W)
        row = out[i]
        assert np.array_equal(row.indices, out_r[inside] * W + out_c[inside])
        assert np.allclose(row.data, full.ravel()[inside], atol=1e-12)


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
@pytest.mark.parametrize("data,indices", [
    ([0.0], [55]),  ## explicitly stored zero
    ([1.0, -1.0], [55, 56]),  ## outputs that cancel to exactly zero
])
def test_direct_exact_zeros(data, indices):
    """
    Outputs that are exactly zero are dropped without leaving unfilled
    slots. Regression: the counting pass reserved slots for them, leaving
    garbage indices in the returned CSR.
    """
    x = scipy.sparse.csr_matrix(
        (np.array(data), np.array(indices), np.array([0, len(data)])), shape=(1, 100),
    )
    conv = Toeplitz_convolution2d(
        x_shape=(10, 10), k=np.ones((3, 3)), mode='same', method='direct',
    )
    out = conv(x=x, batching=True)
    out.check_format(full_check=True)
    assert np.all(out.data != 0)
    ref = scipy.signal.convolve2d(
        x.toarray().reshape(10, 10), np.ones((3, 3)), mode='same',
    )
    assert np.abs(out.toarray().reshape(10, 10) - ref).max() < 1e-12


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
def test_direct_fuzz_vs_scipy():
    """
    Random small frames, modes and kernels (including single-tap
    displacements, offset supports and kernels larger than the image) with
    sparse inputs that include explicit zeros, duplicates and exact
    cancellations. Covers both dispatch paths and the bounding-box path.
    """
    rng = np.random.default_rng(0)
    values = np.array([-1.0, 0.0, 0.5, 1.0])  ## few distinct values: outputs can cancel
    n_counted = n_overalloc = 0
    for _ in range(1000):
        H, W = (int(v) for v in rng.integers(1, 13, size=2))
        k_shape = tuple(int(v) for v in rng.integers(1, 25, size=2))
        mode = str(rng.choice(['full', 'same', 'valid']))
        if mode == 'valid' and (k_shape[0] > H or k_shape[1] > W):
            continue
        kernel = rng.choice(values, size=k_shape) * (rng.random(k_shape) < rng.random())
        batch = int(rng.integers(1, 6))
        n = int(rng.integers(0, 4 * batch))
        ## COO -> CSR sums duplicates and keeps explicit zeros
        idx_row, idx_col = rng.integers(0, batch, n), rng.integers(0, H * W, n)
        x = scipy.sparse.csr_matrix(
            (rng.choice(values, size=n), (idx_row, idx_col)), shape=(batch, H * W),
        )

        conv = Toeplitz_convolution2d(
            x_shape=(H, W), k=kernel, mode=mode, method='direct',
        )
        out = conv(x=x, batching=True)
        ref = np.stack([
            scipy.signal.convolve2d(x_i.reshape(H, W), kernel, mode=mode).reshape(-1)
            for x_i in x.toarray()
        ])  ## shape: (batch, H_out * W_out)
        out.check_format(full_check=True)
        assert out.has_sorted_indices and np.all(out.data != 0)
        assert np.abs(out.toarray() - ref).max() < 1e-12, (H, W, k_shape, mode)

        is_counted = x.getnnz(axis=1).mean() * np.count_nonzero(kernel) < ref.shape[1]
        n_counted += int(is_counted)
        n_overalloc += int(not is_counted)
    assert n_counted > 100 and n_overalloc > 100, (n_counted, n_overalloc)


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


    out_ref = scipy.signal.convolve2d(x_dense, kernel, mode=mode)
    diff = np.abs(out - out_ref).max()
    assert diff < 1e-10, f"Dense precomputed+torch vs scipy.signal: max diff={diff:.2e}"
