# sparse_convolution
Sparse convolution in python. \
Uses Toeplitz convolutional matrix multiplication to perform sparse convolution. \
This allows for extremely fast convolution when: 
- The kernel is small (<= 100x100)
- The input array is sparse (<= 1% density)
- The input array is small (<= 1000x1000)
- Many arrays are convolved with the same kernel (large batch size >= 1000)

## Install: 
The package is available on PyPI. \
`pip install sparse_convolution`

<br>

Alternatively, you can install from source. \
`git clone https://github.com/RichieHakim/sparse_convolution` \
`cd sparse_convolution` \
`pip install -e .` 


## Basic usage: 
Convolve a single sparse 2D array with a 2D kernel.
```
import sparse_convolution as sc
import numpy as np
import scipy.sparse

# Create a single sparse matrix
A = scipy.sparse.rand(100, 100, density=0.1)

# Create a dense kernel
B = np.random.rand(3, 3)

# Prepare class
conv = sc.Toeplitz_convolution2d(
    x_shape=A.shape,
    k=B,
    mode='same',
    dtype=np.float32,
)

# Convolve
C = conv(
    x=A,
    batching=False,
).toarray()
```


## Batching usage: 
Convolve multiple sparse 2D arrays with a 2D kernel. \
The input arrays must be reshaped into flattened vectors and stacked into a single sparse array of shape: `(n_arrays, height * width)`. 
```
import sparse_convolution as sc
import numpy as np
import scipy.sparse

# Create multiple sparse matrices
# note that the shape of A will be (3, 100**2)
A = scipy.sparse.vstack([
    scipy.sparse.rand(100, 100, density=0.1).reshape(1, -1),
    scipy.sparse.rand(100, 100, density=0.1).reshape(1, -1),
    scipy.sparse.rand(100, 100, density=0.1).reshape(1, -1),
]).tocsr()

# Create a dense kernel
B = np.random.rand(3, 3)

# Prepare class
conv = sc.Toeplitz_convolution2d(
    x_shape=(100, 100),  # note that the input shape here is (100, 100)
    k=B,
    mode='same',
    dtype=np.float32,
)

# Convolve
C = conv(
    x=A,
    batching=True,
)

# Reshape the output back to (3, 100, 100)
C_reshaped = np.stack([c.reshape(100, 100).toarray() for c in C], axis=0)
```

## Performance & Scaling
The `Toeplitz_convolution2d` class now supports two backend methods: `lazy` and `precomputed` (which can be chosen by passing `method` to the constructor).
- **`method='lazy'`**: Optimised for very sparse inputs and **instant initialization**. Based on our benchmarks with a 10x10 kernel on a 1000x1000 matrix, the lazy method is significantly faster up to approximately **10% density (0.1)**. Since there is no huge Toeplitz matrix to build, it saves considerable memory.
- **`method='precomputed'`**: Slower to initialize because it builds a massive sparse Toeplitz matrix. However, it is suggested and highly efficient for extremely large batch sizes (e.g., > 75) where the initial cost is amortized over many quick dense-sparse matrix multiplications.

## References
- See: https://stackoverflow.com/a/51865516 and https://github.com/alisaaalehi/convolution_as_multiplication
    for a nice illustration.
- See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.convolution_matrix.html 
    for 1D version.
- See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html#scipy.linalg.matmul_toeplitz 
    for potential ways to make this implementation faster.

