import time
import numpy as np
import scipy.sparse
from sparse_convolution import Toeplitz_convolution2d

# Setup: 500x500 Fully Dense Matrix (100% intensity)
shape = (500, 500)
kernel = np.random.rand(5, 5)
x_dense = np.random.rand(*shape) 
x_sparse = scipy.sparse.csr_matrix(x_dense)

print(f"--- High Intensity Benchmark ({shape[0]}x{shape[1]}, Density: 100%) ---")

# --- Test: Broadcasting Engine (Lazy Approach) ---
tic = time.time()
conv_lazy = Toeplitz_convolution2d(shape, kernel, method='lazy')
init_lazy = time.time() - tic

tic = time.time()
out_lazy = conv_lazy(x_sparse, batching=False)
exec_lazy = time.time() - tic

print(f"\nBroadcasting Engine (Lazy):")
print(f"   Initialization: {init_lazy:.6f}s")
print(f"   Execution:      {exec_lazy:.6f}s")

# --- Test: Toeplitz Engine (Precomputed Approach) ---
tic = time.time()
conv_pre = Toeplitz_convolution2d(shape, kernel, method='precomputed')
init_pre = time.time() - tic

tic = time.time()
out_pre = conv_pre(x_sparse, batching=False)
exec_pre = time.time() - tic

print(f"\nToeplitz Engine (Precomputed):")
print(f"   Initialization: {init_pre:.6f}s")
print(f"   Execution:      {exec_pre:.6f}s")

# --- Repeated Execution Test (10 Iterations) ---
print(f"\nRepeated Execution Test (10 Iterations):")
tic = time.time()
for _ in range(10): conv_lazy(x_sparse, batching=False)
total_lazy = time.time() - tic
print(f"   Broadcasting Total: {total_lazy:.6f}s")

tic = time.time()
for _ in range(10): conv_pre(x_sparse, batching=False)
total_pre = time.time() - tic
print(f"   Toeplitz Total:     {total_pre:.6f}s")