"""
Grid benchmark: image dimension x density, single-call comparison.

Measures the three fundamental cost components separately:
  - broadcast: COO scatter-add (lazy per-call cost)
  - matmul:    Toeplitz sparse matmul (precomputed per-call cost)
  - build:     Toeplitz matrix construction (one-time cost for both methods)

Outputs a table, a figure (benchmarks/figure_grid.png), and JSON results.

Usage:
    python benchmarks/benchmark_grid.py
"""

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sparse_convolution import Toeplitz_convolution2d


## ---- Grid parameters ----
DIMS = [50, 100, 200, 500, 1000, 2000]
DENSITIES = [0.0001, 0.001, 0.01, 0.1, 0.5]
K_SHAPE = (5, 5)
N_REPEATS = 3  ## average over repeats for stability
SEED = 42


def measure_cell(dim, density):
    """
    Measure broadcast, matmul, and build times for a single (dim, density) cell.
    Returns dict with times in seconds.
    """
    x_shape = (dim, dim)
    np.random.seed(SEED)
    kernel = np.random.rand(*K_SHAPE)
    x = scipy.sparse.random(x_shape[0], x_shape[1], density=density, format='csr', random_state=SEED)

    ## Measure build time
    times_build = []
    for _ in range(N_REPEATS):
        tic = time.perf_counter()
        dt, so = Toeplitz_convolution2d._build_toeplitz_matrix(x_shape, kernel, kernel.dtype)
        times_build.append(time.perf_counter() - tic)
    t_build = np.median(times_build)

    ## Measure matmul time (using pre-built dt)
    conv_pre = Toeplitz_convolution2d(x_shape=x_shape, k=kernel, method='precomputed')
    times_matmul = []
    for _ in range(N_REPEATS):
        tic = time.perf_counter()
        conv_pre._compute_toeplitz(x=x, mode='same', batching=False, issparse=True)
        times_matmul.append(time.perf_counter() - tic)
    t_matmul = np.median(times_matmul)

    ## Measure broadcast time
    conv_lazy = Toeplitz_convolution2d(x_shape=x_shape, k=kernel, method='lazy')
    times_broadcast = []
    for _ in range(N_REPEATS):
        tic = time.perf_counter()
        out = conv_lazy._compute_broadcasting(x=x, mode='same', batching=False)
        out.sum_duplicates()
        times_broadcast.append(time.perf_counter() - tic)
    t_broadcast = np.median(times_broadcast)

    return {
        'dim': dim,
        'density': density,
        'nnz': x.nnz,
        'broadcast_s': float(t_broadcast),
        'matmul_s': float(t_matmul),
        'build_s': float(t_build),
    }


def print_table(results):
    """Print a formatted table of results."""

    ## Group by density for column headers
    print("\n" + "=" * 90)
    print("BROADCAST time (ms) — per-call cost of method='lazy'")
    print("=" * 90)
    print(f"{'dim':<8}", end="")
    for d in DENSITIES:
        print(f" | {d:<12}", end="")
    print()
    print("-" * 90)
    for dim in DIMS:
        print(f"{dim:<8}", end="")
        for d in DENSITIES:
            r = next(r for r in results if r['dim'] == dim and r['density'] == d)
            print(f" | {r['broadcast_s']*1000:<12.3f}", end="")
        print()

    print("\n" + "=" * 90)
    print("MATMUL time (ms) — per-call cost of method='precomputed'")
    print("=" * 90)
    print(f"{'dim':<8}", end="")
    for d in DENSITIES:
        print(f" | {d:<12}", end="")
    print()
    print("-" * 90)
    for dim in DIMS:
        print(f"{dim:<8}", end="")
        for d in DENSITIES:
            r = next(r for r in results if r['dim'] == dim and r['density'] == d)
            print(f" | {r['matmul_s']*1000:<12.3f}", end="")
        print()

    print("\n" + "=" * 90)
    print("BUILD time (ms) — one-time Toeplitz construction cost (same for both methods)")
    print("=" * 90)
    print(f"{'dim':<8} | {'build (ms)':<12} | {'pixels':<12}")
    print("-" * 40)
    seen = set()
    for dim in DIMS:
        if dim not in seen:
            r = next(r for r in results if r['dim'] == dim)
            print(f"{dim:<8} | {r['build_s']*1000:<12.1f} | {dim*dim:<12}")
            seen.add(dim)

    print("\n" + "=" * 90)
    print("RATIO broadcast / matmul  (< 1 = lazy broadcast wins per-call)")
    print("=" * 90)
    print(f"{'dim':<8}", end="")
    for d in DENSITIES:
        print(f" | {d:<12}", end="")
    print()
    print("-" * 90)
    for dim in DIMS:
        print(f"{dim:<8}", end="")
        for d in DENSITIES:
            r = next(r for r in results if r['dim'] == dim and r['density'] == d)
            ratio = r['broadcast_s'] / r['matmul_s'] if r['matmul_s'] > 0 else float('inf')
            marker = "<" if ratio < 1 else ">"
            print(f" | {ratio:<5.2f} {marker:<6}", end="")
        print()


def make_figure(results, filepath_fig):
    """Create a 2-panel figure showing the performance tradeoff."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(DENSITIES)))

    ## ---- Panel 1: Absolute times vs dimension ----
    ax = axes[0]
    for i, d in enumerate(DENSITIES):
        cells = [r for r in results if r['density'] == d]
        cells.sort(key=lambda r: r['dim'])
        dims = [r['dim'] for r in cells]
        t_bcast = [r['broadcast_s'] * 1000 for r in cells]
        t_mat = [r['matmul_s'] * 1000 for r in cells]
        ax.plot(dims, t_bcast, 'o--', color=colors[i], label=f'broadcast d={d}', markersize=4)
        ax.plot(dims, t_mat, 's-', color=colors[i], label=f'matmul d={d}', markersize=4, alpha=0.6)

    ## Build time (independent of density)
    build_cells = []
    seen = set()
    for r in sorted(results, key=lambda r: r['dim']):
        if r['dim'] not in seen:
            build_cells.append(r)
            seen.add(r['dim'])
    ax.plot(
        [r['dim'] for r in build_cells],
        [r['build_s'] * 1000 for r in build_cells],
        'k^-', label='Toeplitz build', markersize=6, linewidth=2,
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Image dimension (NxN)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Per-call cost: broadcast (dashed) vs matmul (solid)\nand one-time Toeplitz build (black)')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)

    ## ---- Panel 2: Ratio broadcast/matmul ----
    ax = axes[1]
    for i, d in enumerate(DENSITIES):
        cells = [r for r in results if r['density'] == d]
        cells.sort(key=lambda r: r['dim'])
        dims = [r['dim'] for r in cells]
        ratios = [r['broadcast_s'] / r['matmul_s'] if r['matmul_s'] > 0 else float('inf') for r in cells]
        ax.plot(dims, ratios, 'o-', color=colors[i], label=f'density={d}', markersize=5)

    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(DIMS[0] * 0.8, 0.7, 'broadcast faster', fontsize=9, color='green', alpha=0.8)
    ax.text(DIMS[0] * 0.8, 1.5, 'matmul faster', fontsize=9, color='red', alpha=0.8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Image dimension (NxN)')
    ax.set_ylabel('Ratio: broadcast / matmul')
    ax.set_title(f'Per-call speedup (kernel={K_SHAPE[0]}x{K_SHAPE[1]})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath_fig, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {filepath_fig}")


def main():
    results = []
    total = len(DIMS) * len(DENSITIES)
    print(f"Running {total} cells ({len(DIMS)} dims x {len(DENSITIES)} densities), {N_REPEATS} repeats each...")

    for i, dim in enumerate(DIMS):
        for j, density in enumerate(DENSITIES):
            cell = measure_cell(dim, density)
            results.append(cell)
            n = i * len(DENSITIES) + j + 1
            print(f"  [{n:>3}/{total}] {dim:>5}x{dim:<5} d={density:<8} "
                  f"bcast={cell['broadcast_s']*1000:>8.3f}ms  "
                  f"matmul={cell['matmul_s']*1000:>8.3f}ms  "
                  f"build={cell['build_s']*1000:>8.1f}ms  "
                  f"nnz={cell['nnz']}")

    print_table(results)

    ## Save JSON
    filepath_json = str(Path(__file__).parent / "results_grid.json")
    with open(filepath_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved to {filepath_json}")

    ## Save figure
    filepath_fig = str(Path(__file__).parent / "figure_grid.png")
    make_figure(results, filepath_fig)


if __name__ == "__main__":
    main()
