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

Three methods, each with selectable backends:

| Method | numpy | numba | torch |
|---|:---:|:---:|:---:|
| `precomputed` | yes | yes | yes |
| `lazy` | yes | n/a | yes |
| `gather_scatter` | yes | yes | yes |

- **`precomputed`**: Builds a sparse Toeplitz matrix at init; fast batched matmul. Best for large batches with the same kernel.
- **`lazy`** (default): COO broadcasting, no init cost. Best for very sparse inputs with small batches.
- **`gather_scatter`**: Per-kernel-position scatter into a dense accumulator. Best general-purpose method for sparse batched inputs.

Backend selection:
- **`numpy`**: scipy/numpy ops. Always available.
- **`numba`**: JIT-compiled parallel loops. Fastest on CPU for batched inputs. Requires `numba`.
- **`torch`**: PyTorch ops with optional GPU. Requires `torch`.

```python
conv = sc.Toeplitz_convolution2d(
    x_shape=(100, 100),
    k=k,
    mode='same',
    method='gather_scatter',
    backend='numba',
)
```

If `backend=None` (default), auto-selects `numba` for `gather_scatter` (if installed), `numpy` otherwise.

## References
- Toeplitz convolution: [stackoverflow.com/a/51865516](https://stackoverflow.com/a/51865516), [alisaaalehi/convolution_as_multiplication](https://github.com/alisaaalehi/convolution_as_multiplication)
- 1D convolution matrix: [scipy.linalg.convolution_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.convolution_matrix.html)
