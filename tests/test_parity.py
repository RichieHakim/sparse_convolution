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

def test_parity(density, mode, batching):
    # Setup shapes
    x_shape = (50, 50)
    k_shape = (5, 5)
    
    # Generate kernel
    np.random.seed(42)
    kernel = np.random.rand(*k_shape).astype(np.float32)
    
    # Generate sparse input based on density
    if batching:
        # For batching, shape must be (batch_size, x_shape[0]*x_shape[1])
        batch_size = 10
        total_elements = batch_size * x_shape[0] * x_shape[1]
        num_non_zero = int(total_elements * density)
        x_dense = np.zeros((batch_size, x_shape[0] * x_shape[1]), dtype=np.float32)
        indices = np.random.choice(total_elements, num_non_zero, replace=False)
        x_dense.flat[indices] = np.random.rand(num_non_zero)
        x_sparse = scipy.sparse.csr_matrix(x_dense)
    else:
        total_elements = x_shape[0] * x_shape[1]
        num_non_zero = int(total_elements * density)
        x_dense = np.zeros(x_shape, dtype=np.float32)
        indices = np.random.choice(total_elements, num_non_zero, replace=False)
        x_dense.flat[indices] = np.random.rand(num_non_zero)
        x_sparse = scipy.sparse.csr_matrix(x_dense)

    # Instantiate convolution objects
    conv_lazy = Toeplitz_convolution2d(
        x_shape=x_shape,
        k=kernel,
        mode=mode,
        method='lazy',
        dtype=np.float32
    )

    conv_precomputed = Toeplitz_convolution2d(
        x_shape=x_shape,
        k=kernel,
        mode=mode,
        method='precomputed',
        dtype=np.float32
    )

    # Compute outputs
    out_lazy = conv_lazy(x=x_sparse, batching=batching)
    out_precomputed = conv_precomputed(x=x_sparse, batching=batching)

    # Compare
    if scipy.sparse.issparse(out_lazy):
        out_lazy = out_lazy.toarray()
    if scipy.sparse.issparse(out_precomputed):
        out_precomputed = out_precomputed.toarray()

    diff = np.abs(out_lazy - out_precomputed).max()
    print(f"✅ Pass -> mode: {mode}, density: {density}, batching: {batching} | Max diff: {diff:.2e}")
    assert diff < 1e-6, f"Max difference is {diff}, expected < 1e-6"

if __name__ == "__main__":
    import sys
    # Add -s to args to ensure stdout is printed
    pytest.main(["-s", __file__] + sys.argv[1:])
