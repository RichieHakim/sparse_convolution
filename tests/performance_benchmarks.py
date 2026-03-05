import time
import numpy as np
import scipy.sparse
from sparse_convolution import Toeplitz_convolution2d

def run_density_benchmark(matrix_size=(1000, 1000), kernel_size=(10, 10), densities=[0.001, 0.01, 0.05, 0.1]):
    """
    Runs a performance benchmark comparing 'lazy' and 'precomputed' methods
    across different matrix sparsity densities.
    """
    print(f"--- Performance Benchmark ---")
    print(f"Matrix Size: {matrix_size[0]}x{matrix_size[1]}")
    print(f"Kernel Size: {kernel_size[0]}x{kernel_size[1]}")
    print("-" * 65)
    print(f"{'Density':<10} | {'Lazy Time (s)':<15} | {'Precomputed Time (s)':<20}")
    print("-" * 65)
    
    kernel = np.random.rand(*kernel_size).astype(np.float32)
    results = []

    for density in densities:
        # Create sparse matrix
        total_elements = matrix_size[0] * matrix_size[1]
        num_non_zero = int(total_elements * density)
        x_dense = np.zeros(matrix_size, dtype=np.float32)
        indices = np.random.choice(total_elements, num_non_zero, replace=False)
        x_dense.flat[indices] = np.random.rand(num_non_zero)
        x_sparse = scipy.sparse.csr_matrix(x_dense)

        # Benchmark Lazy
        tic = time.time()
        conv_lazy = Toeplitz_convolution2d(matrix_size, kernel, method='lazy', dtype=np.float32)
        _ = conv_lazy(x_sparse, batching=False)
        time_lazy = time.time() - tic

        # Benchmark Precomputed
        tic = time.time()
        conv_precomputed = Toeplitz_convolution2d(matrix_size, kernel, method='precomputed', dtype=np.float32)
        _ = conv_precomputed(x_sparse, batching=False)
        time_precomputed = time.time() - tic

        print(f"{density:<10} | {time_lazy:<15.4f} | {time_precomputed:<20.4f}")
        results.append({
            'density': density,
            'time_lazy': time_lazy,
            'time_precomputed': time_precomputed
        })

    print("-" * 65)
    return results

if __name__ == "__main__":
    run_density_benchmark()
