# sparse_convolution
Sparse 2D convolution in Python via Toeplitz matrix methods.

Fast when the kernel is small, the input is sparse, and/or many arrays share the same kernel.

## Install
```
pip install sparse_convolution
```

Or from source:
```
git clone https://github.com/RichieHakim/sparse_convolution
cd sparse_convolution
pip install -e .
```

## Usage

### Single image
```python
import numpy as np
import scipy.sparse
import sparse_convolution as sc

x = scipy.sparse.random(100, 100, density=0.01)
k = np.random.rand(5, 5)

conv = sc.Toeplitz_convolution2d(x_shape=x.shape, k=k, mode='same')
out = conv(x=x, batching=False).toarray()
```

### Batched
Input: `(n_images, H * W)` sparse matrix. Output: `(n_images, H_out * W_out)`.
```python
x_batch = scipy.sparse.vstack([
    scipy.sparse.random(100, 100, density=0.01).reshape(1, -1)
    for _ in range(50)
]).tocsr()

conv = sc.Toeplitz_convolution2d(x_shape=(100, 100), k=k, mode='same')
out = conv(x=x_batch, batching=True)
```

## Methods and backends

Four methods, each with selectable backends:

| Method | numpy | numba | torch |
|---|:---:|:---:|:---:|
| `direct` | n/a | yes | n/a |
| `precomputed` | yes | yes | yes |
| `lazy` | yes | n/a | yes |
| `gather_scatter` | yes | yes | yes |

- **`direct`** (default): Two-pass batch-parallel scatter with thread-local dense buffers (numba only). Zero init overhead, O(nnz×K) per image. Fastest method across nearly all configurations. Requires `numba`.
- **`precomputed`**: Builds a sparse Toeplitz matrix at init; fast batched matmul. Best for large batches with the same kernel when numba is not available.
- **`lazy`**: COO broadcasting, no init cost. Best for very sparse inputs with small batches.
- **`gather_scatter`**: Per-kernel-position scatter into a dense accumulator. General-purpose method for sparse batched inputs.

Backend selection:
- **`numpy`**: scipy/numpy ops. Always available.
- **`numba`**: JIT-compiled parallel loops. Fastest on CPU for batched inputs. Requires `numba`.
- **`torch`**: PyTorch ops with optional GPU. Requires `torch`.

```python
conv = sc.Toeplitz_convolution2d(
    x_shape=(100, 100),
    k=k,
    mode='same',
    method='direct',       # default
    backend='numba',       # auto-selected for direct
)
```

If `backend=None` (default), auto-selects `numba` for `direct` and `gather_scatter` (if installed), `numpy` otherwise.

## References
- Toeplitz convolution: [stackoverflow.com/a/51865516](https://stackoverflow.com/a/51865516), [alisaaalehi/convolution_as_multiplication](https://github.com/alisaaalehi/convolution_as_multiplication)
- 1D convolution matrix: [scipy.linalg.convolution_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.convolution_matrix.html)

## Benchmarks

All benchmarks run on CPU with 2s minimum measurement time per configuration (median reported). Eight method+backend combinations compared across six scaling sweeps.

### Batch size scaling
100×100 images, 5×5 kernel, density=0.01
![Batch scaling](benchmarks/results/scaling_batch_scaling.png)

### Density scaling
100×100 images, 5×5 kernel, batch=100
![Density scaling](benchmarks/results/scaling_density_scaling.png)

### Image size scaling
5×5 kernel, density=0.01, batch=50
![Image size scaling](benchmarks/results/scaling_img_size_scaling.png)

### Kernel size scaling
100×100 images, density=0.01, batch=50
![Kernel size scaling](benchmarks/results/scaling_kernel_size_scaling.png)

### Batch scaling — high density
100×100 images, 5×5 kernel, density=0.1
![Batch scaling high density](benchmarks/results/scaling_batch_scaling_high_density.png)

### Batch scaling — very sparse
100×100 images, 5×5 kernel, density=0.001
![Batch scaling very sparse](benchmarks/results/scaling_batch_scaling_very_sparse.png)
