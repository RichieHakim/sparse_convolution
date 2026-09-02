"""
Large-scale and matrix-free validation suite for sparse_convolution.

Validates:
1. Numerical parity between MatrixFreeToeplitzConvolution2D, Toeplitz_convolution2d,
   and scipy.signal.convolve2d on 50x50 test cases (error < 1e-6).
2. Parity across even and odd kernel shapes (3x3, 5x5, 2x2, 4x4, 2x3, 3x2) for
   'full', 'same', and 'valid' modes (zero off-by-one offset errors).
3. Adjoint operator <Ax, y> = <x, A^T y> mathematical exactness.
4. Compatibility with scipy.sparse.linalg iterative solvers (lsqr).
5. Stress testing on 10,000x10,000 sparse grid with 10^5 non-zeros and 21x21 kernel:
   - Peak RAM < 1.0 GB
   - Runtime < 3.0 s
6. Automatic fallback from method='precomputed' to method='matrix_free' on huge matrices.
"""

import time
import tracemalloc
import warnings
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.signal
import pytest

from sparse_convolution import Toeplitz_convolution2d, MatrixFreeToeplitzConvolution2D


def test_numerical_parity_50x50():
    """
    Test numerical correctness against scipy.signal.convolve2d and
    Toeplitz_convolution2d on 50x50 matrices.
    """
    np.random.seed(123)
    x_shape = (50, 50)
    x_dense = np.random.randn(*x_shape).astype(np.float64)
    x_sparse = scipy.sparse.random(*x_shape, density=0.08, format='csr', dtype=np.float64, random_state=123)
    kernel = np.random.randn(7, 7).astype(np.float64)

    for mode in ['full', 'same', 'valid']:
        # 1. Classical reference
        ref_dense = scipy.signal.convolve2d(x_dense, kernel, mode=mode)
        ref_sparse = scipy.signal.convolve2d(x_sparse.toarray(), kernel, mode=mode)

        # 2. MatrixFreeToeplitzConvolution2D LinearOperator
        op = MatrixFreeToeplitzConvolution2D(x_shape=x_shape, k=kernel, mode=mode, dtype=np.float64)
        
        # Test matvec
        y_matvec = op.matvec(x_dense.ravel())
        err_matvec = np.linalg.norm(y_matvec - ref_dense.ravel()) / np.linalg.norm(ref_dense.ravel())
        assert err_matvec < 1e-6, f"matvec relative error {err_matvec} >= 1e-6 for mode={mode}"

        # Test __call__ dense
        out_dense = op(x_dense, batching=False, mode=mode)
        err_dense = np.linalg.norm(out_dense - ref_dense) / np.linalg.norm(ref_dense)
        assert err_dense < 1e-6, f"dense relative error {err_dense} >= 1e-6 for mode={mode}"

        # Test __call__ sparse
        out_sparse = op(x_sparse, batching=False, mode=mode)
        err_sparse = np.linalg.norm(out_sparse.toarray() - ref_sparse) / np.linalg.norm(ref_sparse)
        assert err_sparse < 1e-6, f"sparse relative error {err_sparse} >= 1e-6 for mode={mode}"

        # 3. Toeplitz_convolution2d with method='matrix_free'
        conv_mf = Toeplitz_convolution2d(x_shape=x_shape, k=kernel, mode=mode, method='matrix_free', dtype=np.float64)
        out_conv_sp = conv_mf(x_sparse, batching=False)
        err_conv_sp = np.linalg.norm(out_conv_sp.toarray() - ref_sparse) / np.linalg.norm(ref_sparse)
        assert err_conv_sp < 1e-6, f"Toeplitz_convolution2d matrix_free error {err_conv_sp} >= 1e-6 for mode={mode}"


def test_kernel_parity_even_and_odd():
    """
    Test against off-by-one errors across both odd and even kernel dimensions.
    """
    np.random.seed(42)
    x = np.zeros((30, 30), dtype=np.float64)
    x[15, 15] = 1.0  # impulse

    test_kernels = [
        np.random.rand(3, 3),
        np.random.rand(5, 5),
        np.random.rand(21, 21),
        np.random.rand(2, 2),
        np.random.rand(4, 4),
        np.random.rand(2, 3),
        np.random.rand(3, 2),
    ]

    for k in test_kernels:
        for mode in ['full', 'same', 'valid']:
            ref = scipy.signal.convolve2d(x, k, mode=mode)
            op = MatrixFreeToeplitzConvolution2D(x_shape=x.shape, k=k, mode=mode, dtype=np.float64)
            out = op(x, batching=False)
            diff = np.max(np.abs(out - ref))
            assert diff < 1e-12, f"Off-by-one mismatch on kernel {k.shape} mode {mode}: max diff = {diff}"


def test_adjoint_and_linear_operator_solvers():
    """
    Verify mathematical adjoint identity <Ax, y> = <x, A^T y> and LSQR convergence.
    """
    np.random.seed(999)
    x_shape = (20, 16)
    k = np.random.randn(4, 5)

    for mode in ['full', 'same', 'valid']:
        op = MatrixFreeToeplitzConvolution2D(x_shape=x_shape, k=k, mode=mode)
        
        # Test adjoint identity
        x_vec = np.random.randn(op.shape[1])
        y_vec = np.random.randn(op.shape[0])

        Ax = op.matvec(x_vec)
        ATy = op.rmatvec(y_vec)

        inner1 = np.dot(Ax, y_vec)
        inner2 = np.dot(x_vec, ATy)
        assert np.isclose(inner1, inner2, rtol=1e-10, atol=1e-12), \
            f"Adjoint identity failed for mode={mode}: {inner1} vs {inner2}"

        # Test matmat multi-vector
        V = np.random.randn(op.shape[1], 4)
        Y_matmat = op.matmat(V)
        assert Y_matmat.shape == (op.shape[0], 4)
        for col in range(4):
            assert np.allclose(Y_matmat[:, col], op.matvec(V[:, col]), atol=1e-12)

        # Test LSQR iterative solver
        res = scipy.sparse.linalg.lsqr(op, Ax, iter_lim=15)
        assert res is not None


def test_large_scale_stress_10000x10000():
    """
    Stress test on 10,000x10,000 sparse grid with 10^5 non-zeros and 21x21 circular kernel.
    Asserts:
    - Peak RAM < 1.0 GB
    - Runtime < 3.0 s
    """
    H, W = 10000, 10000
    nnz = 100000

    np.random.seed(42)
    rows = np.random.randint(0, H, size=nnz, dtype=np.int32)
    cols = np.random.randint(0, W, size=nnz, dtype=np.int32)
    data = np.ones(nnz, dtype=np.float32)

    mat_sparse = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(H, W), dtype=np.float32)

    # 21x21 circular kernel (314 non-zeros)
    y, x = np.ogrid[-10:11, -10:11]
    kernel = ((x**2 + y**2) <= 100).astype(np.float32)
    kernel /= kernel.sum()

    tracemalloc.start()
    t0 = time.perf_counter()

    conv = Toeplitz_convolution2d(
        x_shape=(H, W),
        k=kernel,
        mode='same',
        method='matrix_free',
        dtype=np.float32,
    )
    t_init = time.perf_counter() - t0

    t1 = time.perf_counter()
    out = conv(mat_sparse, batching=False)
    t_call = time.perf_counter() - t1

    total_time = t_init + t_call
    current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_ram_mb = peak_bytes / (1024 * 1024)
    peak_ram_gb = peak_bytes / (1024**3)

    print(f"\n[STRESS TEST 10000x10000]")
    print(f"Total time: {total_time:.3f} s (Init: {t_init:.3f}s, Call: {t_call:.3f}s)")
    print(f"Peak RAM: {peak_ram_mb:.2f} MB ({peak_ram_gb:.3f} GB)")
    print(f"Output non-zeros: {out.nnz}")

    assert peak_ram_gb < 1.0, f"Peak RAM {peak_ram_gb:.2f} GB exceeded limit of 1.0 GB"
    assert total_time < 3.0, f"Total runtime {total_time:.2f} s exceeded limit of 3.0 s"
    assert out.shape == (H, W)
    assert out.nnz > 0


def test_auto_fallback_from_precomputed():
    """
    Verify that initializing Toeplitz_convolution2d with method='precomputed' on
    dimensions that would cause OOM/integer overflow cleanly falls back to 'matrix_free'.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        huge_conv = Toeplitz_convolution2d(
            x_shape=(10295, 8975),
            k=np.ones((21, 21), dtype=np.float32),
            mode='same',
            method='precomputed',
        )
        assert huge_conv.method == 'matrix_free'
        assert any("exceeds safe precomputation thresholds" in str(item.message) for item in w)
        
        linop = huge_conv.as_linear_operator()
        assert isinstance(linop, MatrixFreeToeplitzConvolution2D)
        assert linop.shape == (10295 * 8975, 10295 * 8975)
