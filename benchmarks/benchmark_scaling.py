"""
Benchmark testing various x_shape / k_shape combinations at scale.

Tests both 'lazy' and 'precomputed' methods across a variety of spatial
configurations: 1D-like arrays, small-to-large squares, and rectangular
kernels. Runs with batching enabled.

Originally adapted from tests/benchmarks.py (RH 2022).

Results are printed to stdout and saved to benchmarks/results_scaling.json.

Usage:
    python benchmarks/benchmark_scaling.py
"""

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse

from sparse_convolution import Toeplitz_convolution2d


## (x_height, x_width, k_height, k_width)
SHAPE_CONFIGS = [
    ## 1D-like inputs
    (1,   1000,   1, 5),
    (1,   10000,  1, 5),
    (1,   100000, 1, 5),
    ## Small to large squares with medium kernel
    (16,  16,  5, 5),
    (64,  64,  5, 5),
    (100, 100, 5, 5),
    (256, 256, 5, 5),
    ## Fixed x_shape, varying kernel size
    (100, 100, 2,  2),
    (100, 100, 5,  5),
    (100, 100, 10, 10),
    (100, 100, 20, 20),
    ## Rectangular kernels
    (100, 100, 5, 20),
    (100, 100, 20, 5),
]

BATCH_SIZES = [10, 100, 1000]
DENSITY = 0.001


def run_scaling_benchmark():
    """
    Run the scaling benchmark across all shape configs and batch sizes.
    """
    results = []

    print(f"Density: {DENSITY}")
    print(f"{'x_shape':<16} | {'k_shape':<12} | {'batch':<8} | {'method':<12} | {'init (s)':<12} | {'exec (s)':<12}")
    print("-" * 85)

    for (xh, xw, kh, kw) in SHAPE_CONFIGS:
        x_shape = (xh, xw)
        k_shape = (kh, kw)
        kernel = np.random.rand(kh, kw)

        for batch_size in BATCH_SIZES:
            ## Skip configs that would create very large Toeplitz matrices
            ## (precomputed init would be extremely slow)
            toeplitz_nnz = xh * xw * kh * kw
            skip_precomputed = toeplitz_nnz > 5e7

            ## Build sparse batched input
            x = scipy.sparse.random(
                m=batch_size,
                n=xh * xw,
                density=DENSITY,
                format='csr',
                random_state=0,
            )

            for method in ['lazy', 'precomputed']:
                if method == 'precomputed' and skip_precomputed:
                    print(f"{str(x_shape):<16} | {str(k_shape):<12} | {batch_size:<8} | {method:<12} | {'SKIPPED (too large)':>27}")
                    results.append({
                        'x_shape': list(x_shape),
                        'k_shape': list(k_shape),
                        'batch_size': batch_size,
                        'method': method,
                        'init_s': None,
                        'exec_s': None,
                        'skipped': True,
                    })
                    continue

                tic = time.perf_counter()
                conv = Toeplitz_convolution2d(
                    x_shape=x_shape, k=kernel, mode='same', method=method,
                )
                t_init = time.perf_counter() - tic

                tic = time.perf_counter()
                conv(x=x, batching=True)
                t_exec = time.perf_counter() - tic

                print(
                    f"{str(x_shape):<16} | {str(k_shape):<12} | {batch_size:<8} | "
                    f"{method:<12} | {t_init:<12.6f} | {t_exec:<12.6f}"
                )
                results.append({
                    'x_shape': list(x_shape),
                    'k_shape': list(k_shape),
                    'batch_size': batch_size,
                    'method': method,
                    'init_s': round(t_init, 6),
                    'exec_s': round(t_exec, 6),
                    'skipped': False,
                })

    return results


def main():
    np.random.seed(0)
    results = run_scaling_benchmark()

    filepath_out = str(Path(__file__).parent / "results_scaling.json")
    with open(filepath_out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filepath_out}")


if __name__ == "__main__":
    main()
