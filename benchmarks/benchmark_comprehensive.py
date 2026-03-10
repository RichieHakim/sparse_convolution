"""
Comprehensive benchmark suite for sparse_convolution.

Runs all method+backend combinations across a wide grid of variables:
- x_shape (spatial dimensions)
- k_shape (kernel size)
- density (input sparsity)
- batch_size
- mode (same, full, valid)

Two modes of operation:
1. **Scaling sweeps**: Fix all but one variable, sweep that variable.
   Produces scaling curves.
2. **Grid search**: Sparse factorial grid across all variables.
   Produces an overall performance landscape.

Results are saved as JSON for downstream analysis and plotting.

Usage:
    python benchmark_comprehensive.py [--sweep-only] [--grid-only] [--quick]
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse

from sparse_convolution import Toeplitz_convolution2d


## ---------------------------------------------------------------------------
## Configuration
## ---------------------------------------------------------------------------

## All method+backend combinations to benchmark
def get_method_backends(include_gpu=False):
    """
    Return list of (method, backend, device) tuples to benchmark.

    Args:
        include_gpu (bool):
            If True and CUDA is available, includes torch+cuda variants.
    """
    combos = [
        ('precomputed', 'numpy', 'cpu'),
        ('precomputed', 'torch', 'cpu'),
        ('lazy', 'numpy', 'cpu'),
        ('lazy', 'torch', 'cpu'),
        ('gather_scatter', 'numpy', 'cpu'),
        ('gather_scatter', 'numba', 'cpu'),
        ('gather_scatter', 'torch', 'cpu'),
        ('direct', 'numba', 'cpu'),
    ]

    if include_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                combos.extend([
                    ('precomputed', 'torch', 'cuda'),
                    ('lazy', 'torch', 'cuda'),
                    ('gather_scatter', 'torch', 'cuda'),
                ])
        except ImportError:
            pass

    return combos


## ---------------------------------------------------------------------------
## Benchmark runner
## ---------------------------------------------------------------------------

def benchmark_one(method, backend, device, x_shape, k_shape, batch_size,
                  density, mode, min_time=2.0, n_warmup=2):
    """
    Benchmark a single method+backend+config combination using adaptive
    timing. Runs iterations until at least ``min_time`` seconds of wall
    time have elapsed, then reports median/mean/std.

    Args:
        method, backend, device, x_shape, k_shape, batch_size, density,
        mode: Benchmark configuration.
        min_time (float):
            Minimum total wall time (seconds) to spend on timed call
            iterations. More iterations = more stable statistics.
        n_warmup (int):
            Number of warmup iterations (unjudged).

    Returns:
        (dict or None): Benchmark results dict, or None if skipped/failed.
    """
    np.random.seed(42)
    kernel = np.random.rand(*k_shape).astype(np.float64)

    ## Generate sparse batched input
    n_pixels = x_shape[0] * x_shape[1]
    x_sparse = scipy.sparse.random(
        batch_size, n_pixels, density=density, format='csr', dtype=np.float64,
    )

    ## Skip configs that would be extremely slow (rough estimate)
    total_nnz = int(batch_size * x_shape[0] * x_shape[1] * density)
    k_nnz = k_shape[0] * k_shape[1]  ## upper bound (dense kernel)
    if total_nnz * k_nnz > 500_000_000:  ## >500M contributions
        return None

    try:
        ## Warmup: init + call (triggers numba JIT, torch compile, etc.)
        for _ in range(n_warmup):
            conv = Toeplitz_convolution2d(
                x_shape=x_shape, k=kernel, mode=mode,
                method=method, backend=backend, device=device,
            )
            _ = conv(x=x_sparse, batching=True)

        ## Timed init: adaptive loop until min_time elapsed
        init_times = []
        t_wall = time.perf_counter()
        while (time.perf_counter() - t_wall) < min_time:
            t0 = time.perf_counter()
            conv = Toeplitz_convolution2d(
                x_shape=x_shape, k=kernel, mode=mode,
                method=method, backend=backend, device=device,
            )
            init_times.append(time.perf_counter() - t0)

        ## Timed call: adaptive loop until min_time elapsed (reuse last conv)
        call_times = []
        t_wall = time.perf_counter()
        while (time.perf_counter() - t_wall) < min_time:
            t0 = time.perf_counter()
            _ = conv(x=x_sparse, batching=True)
            call_times.append(time.perf_counter() - t0)

        return {
            'method': method,
            'backend': backend,
            'device': device,
            'x_shape': list(x_shape),
            'k_shape': list(k_shape),
            'batch_size': batch_size,
            'density': density,
            'mode': mode,
            'nnz': int(x_sparse.nnz),
            'n_init_iters': len(init_times),
            'n_call_iters': len(call_times),
            'init_median': float(np.median(init_times)),
            'init_mean': float(np.mean(init_times)),
            'init_std': float(np.std(init_times)),
            'call_median': float(np.median(call_times)),
            'call_mean': float(np.mean(call_times)),
            'call_std': float(np.std(call_times)),
            'total_median': float(np.median(init_times) + np.median(call_times)),
        }
    except Exception as e:
        print(f"    FAILED: {method}+{backend}+{device}: {e}")
        return None


## ---------------------------------------------------------------------------
## Scaling sweeps
## ---------------------------------------------------------------------------

def run_scaling_sweeps(method_backends, min_time=2.0, quick=False):
    """
    Run 1D scaling sweeps: vary one variable at a time, fix the rest.
    Each method+config pair is timed for at least ``min_time`` seconds.
    """
    results = []

    ## Sweep definitions: (sweep_name, variable_name, values, fixed_params)
    sweeps = []

    ## 1. Batch size scaling
    if quick:
        batch_sizes = [1, 5, 10, 50, 100, 500]
    else:
        batch_sizes = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    sweeps.append((
        'batch_scaling',
        'batch_size',
        batch_sizes,
        {'x_shape': (100, 100), 'k_shape': (5, 5), 'density': 0.01, 'mode': 'same'},
    ))

    ## 2. Density scaling
    if quick:
        densities = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.3]
    else:
        densities = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    sweeps.append((
        'density_scaling',
        'density',
        densities,
        {'x_shape': (100, 100), 'k_shape': (5, 5), 'batch_size': 100, 'mode': 'same'},
    ))

    ## 3. Image size scaling (square images)
    if quick:
        img_sizes = [20, 50, 100, 200]
    else:
        img_sizes = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    sweeps.append((
        'img_size_scaling',
        'img_size',
        img_sizes,
        {'k_shape': (5, 5), 'batch_size': 50, 'density': 0.01, 'mode': 'same'},
    ))

    ## 4. Kernel size scaling (square kernels)
    if quick:
        k_sizes = [3, 5, 9, 15]
    else:
        k_sizes = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    sweeps.append((
        'kernel_size_scaling',
        'kernel_size',
        k_sizes,
        {'x_shape': (100, 100), 'batch_size': 50, 'density': 0.01, 'mode': 'same'},
    ))

    ## 5. Batch scaling at high density (precomputed's strength)
    if quick:
        batch_sizes_hd = [1, 10, 100, 500]
    else:
        batch_sizes_hd = [1, 5, 10, 50, 100, 500, 1000, 2000]
    sweeps.append((
        'batch_scaling_high_density',
        'batch_size',
        batch_sizes_hd,
        {'x_shape': (100, 100), 'k_shape': (5, 5), 'density': 0.1, 'mode': 'same'},
    ))

    ## 6. Batch scaling at very low density (lazy's strength)
    if quick:
        batch_sizes_ld = [1, 10, 100, 1000]
    else:
        batch_sizes_ld = [1, 5, 10, 50, 100, 500, 1000, 5000]
    sweeps.append((
        'batch_scaling_very_sparse',
        'batch_size',
        batch_sizes_ld,
        {'x_shape': (200, 200), 'k_shape': (5, 5), 'density': 0.001, 'mode': 'same'},
    ))

    for sweep_name, var_name, var_values, fixed in sweeps:
        print(f"\n{'='*60}")
        print(f"Sweep: {sweep_name} (varying {var_name})")
        print(f"Fixed: {fixed}")
        print(f"{'='*60}")

        for val in var_values:
            ## Build config from fixed params + current variable
            if var_name == 'batch_size':
                config = {**fixed, 'batch_size': val}
            elif var_name == 'density':
                config = {**fixed, 'density': val}
            elif var_name == 'img_size':
                config = {**fixed, 'x_shape': (val, val)}
            elif var_name == 'kernel_size':
                ## Ensure kernel fits in image for mode='valid'
                config = {**fixed, 'k_shape': (val, val)}
                if config.get('mode') == 'valid':
                    img_h, img_w = config.get('x_shape', (100, 100))
                    if val > img_h or val > img_w:
                        continue

            x_shape = config.get('x_shape', (100, 100))
            k_shape = config.get('k_shape', (5, 5))
            batch_size = config['batch_size']
            density = config['density']
            mode = config['mode']

            print(f"\n  {var_name}={val}  "
                  f"(x={x_shape}, k={k_shape}, batch={batch_size}, "
                  f"density={density})")

            for method, backend, device in method_backends:
                r = benchmark_one(
                    method=method, backend=backend, device=device,
                    x_shape=x_shape, k_shape=k_shape,
                    batch_size=batch_size, density=density, mode=mode,
                    min_time=min_time,
                )
                if r is not None:
                    r['sweep'] = sweep_name
                    r['sweep_var'] = var_name
                    r['sweep_val'] = val
                    results.append(r)
                    label = f"{method}+{backend}"
                    if device != 'cpu':
                        label += f"+{device}"
                    print(f"    {label:30s} init={r['init_median']:.6f}s  "
                          f"call={r['call_median']:.6f}s  "
                          f"total={r['total_median']:.6f}s  "
                          f"(n={r['n_call_iters']})")

    return results


## ---------------------------------------------------------------------------
## Grid search
## ---------------------------------------------------------------------------

def run_grid_search(method_backends, min_time=2.0, quick=False):
    """
    Sparse factorial grid search across all variables.
    Each method+config pair is timed for at least ``min_time`` seconds.
    """
    results = []

    ## Grid dimensions (sparse)
    if quick:
        x_shapes = [(50, 50), (100, 100)]
        k_shapes = [(5, 5), (11, 11)]
        batch_sizes = [1, 50, 500]
        densities = [0.001, 0.01, 0.1]
        modes = ['same']
    else:
        x_shapes = [(50, 50), (100, 100), (200, 200)]
        k_shapes = [(3, 3), (5, 5), (11, 11)]
        batch_sizes = [1, 10, 50, 200, 500]
        densities = [0.001, 0.01, 0.05, 0.1]
        modes = ['same']

    configs = list(itertools.product(x_shapes, k_shapes, batch_sizes, densities, modes))
    ## Filter invalid: kernel must be <= image for mode='valid'
    configs = [
        (xs, ks, bs, d, m)
        for xs, ks, bs, d, m in configs
        if m != 'valid' or (xs[0] >= ks[0] and xs[1] >= ks[1])
    ]

    print(f"\n{'='*60}")
    print(f"Grid search: {len(configs)} configurations x {len(method_backends)} method+backends")
    print(f"{'='*60}")

    for i, (x_shape, k_shape, batch_size, density, mode) in enumerate(configs):
        print(f"\n  [{i+1}/{len(configs)}] x={x_shape}, k={k_shape}, "
              f"batch={batch_size}, density={density}, mode={mode}")

        for method, backend, device in method_backends:
            r = benchmark_one(
                method=method, backend=backend, device=device,
                x_shape=x_shape, k_shape=k_shape,
                batch_size=batch_size, density=density, mode=mode,
                min_time=min_time,
            )
            if r is not None:
                r['sweep'] = 'grid'
                r['sweep_var'] = 'grid'
                r['sweep_val'] = i
                results.append(r)
                label = f"{method}+{backend}"
                if device != 'cpu':
                    label += f"+{device}"
                print(f"    {label:30s} total={r['total_median']:.6f}s  "
                      f"(n={r['n_call_iters']})")

    return results


## ---------------------------------------------------------------------------
## Plotting
## ---------------------------------------------------------------------------

def plot_scaling_sweeps(results, output_dir):
    """
    Generate scaling curve plots from sweep results.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    ## Group by sweep
    sweeps = {}
    for r in results:
        sweep = r.get('sweep', 'unknown')
        if sweep not in sweeps:
            sweeps[sweep] = []
        sweeps[sweep].append(r)

    for sweep_name, sweep_results in sweeps.items():
        if sweep_name == 'grid':
            continue  ## Grid results are plotted separately

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(sweep_name.replace('_', ' ').title(), fontsize=14)

        ## Group by method+backend+device
        combos = {}
        for r in sweep_results:
            key = f"{r['method']}+{r['backend']}"
            if r['device'] != 'cpu':
                key += f"+{r['device']}"
            if key not in combos:
                combos[key] = {'x': [], 'init': [], 'call': [], 'total': []}
            combos[key]['x'].append(r['sweep_val'])
            combos[key]['init'].append(r['init_median'])
            combos[key]['call'].append(r['call_median'])
            combos[key]['total'].append(r['total_median'])

        for key, data in combos.items():
            ## Sort by x value
            order = np.argsort(data['x'])
            x_vals = np.array(data['x'])[order]
            init_vals = np.array(data['init'])[order]
            call_vals = np.array(data['call'])[order]
            total_vals = np.array(data['total'])[order]

            axes[0].loglog(x_vals, call_vals, 'o-', label=key, markersize=4)
            axes[1].loglog(x_vals, total_vals, 'o-', label=key, markersize=4)

        sweep_var = sweep_results[0].get('sweep_var', 'x')
        axes[0].set_xlabel(sweep_var)
        axes[0].set_ylabel('Call time (s)')
        axes[0].set_title('Call time only')
        axes[0].legend(fontsize=7, loc='upper left')
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel(sweep_var)
        axes[1].set_ylabel('Total time (init + call) (s)')
        axes[1].set_title('Total time (init + call)')
        axes[1].legend(fontsize=7, loc='upper left')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = output_dir / f"scaling_{sweep_name}.png"
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"  Saved: {filepath}")


def plot_grid_heatmap(results, output_dir):
    """
    Generate a heatmap showing which method+backend is fastest for each
    grid configuration.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    grid_results = [r for r in results if r.get('sweep') == 'grid']
    if not grid_results:
        return

    ## Group by config → find winner
    configs = {}
    for r in grid_results:
        config_key = (
            tuple(r['x_shape']), tuple(r['k_shape']),
            r['batch_size'], r['density'], r['mode']
        )
        if config_key not in configs:
            configs[config_key] = {}
        label = f"{r['method']}+{r['backend']}"
        if r['device'] != 'cpu':
            label += f"+{r['device']}"
        configs[config_key][label] = r['total_median']

    ## Build summary table
    summary = []
    for config_key, timings in configs.items():
        winner = min(timings, key=timings.get)
        summary.append({
            'x_shape': list(config_key[0]),
            'k_shape': list(config_key[1]),
            'batch_size': config_key[2],
            'density': config_key[3],
            'mode': config_key[4],
            'winner': winner,
            'winner_time': timings[winner],
            'all_times': timings,
        })

    ## Save summary
    summary_path = output_dir / 'grid_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved grid summary: {summary_path}")

    ## Print summary table
    print(f"\n{'='*80}")
    print("GRID SEARCH WINNERS")
    print(f"{'='*80}")
    header = f"{'x_shape':>12s} {'k_shape':>8s} {'batch':>6s} {'density':>8s} {'mode':>6s} {'winner':>30s} {'time':>10s}"
    print(header)
    print('-' * len(header))
    for s in sorted(summary, key=lambda s: (s['batch_size'], s['density'])):
        print(f"{str(s['x_shape']):>12s} {str(s['k_shape']):>8s} "
              f"{s['batch_size']:>6d} {s['density']:>8.4f} {s['mode']:>6s} "
              f"{s['winner']:>30s} {s['winner_time']:>10.4f}s")


## ---------------------------------------------------------------------------
## Main
## ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Comprehensive sparse convolution benchmarks')
    parser.add_argument('--sweep-only', action='store_true', help='Run only scaling sweeps')
    parser.add_argument('--grid-only', action='store_true', help='Run only grid search')
    parser.add_argument('--quick', action='store_true', help='Reduced grid for quick testing')
    parser.add_argument('--min-time', type=float, default=2.0,
                        help='Minimum wall time (seconds) per method per config point')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--gpu', action='store_true', help='Include GPU benchmarks')
    args = parser.parse_args()

    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    method_backends = get_method_backends(include_gpu=args.gpu)
    print(f"Benchmarking {len(method_backends)} method+backend combinations "
          f"(min_time={args.min_time}s per measurement):")
    for m, b, d in method_backends:
        print(f"  {m}+{b} (device={d})")

    ## Numba warmup: trigger JIT compilation before timing
    print("\nWarming up numba JIT...")
    try:
        x_warmup = scipy.sparse.random(2, 100, density=0.1, format='csr')
        for warmup_method in ('gather_scatter', 'direct'):
            conv = Toeplitz_convolution2d(
                x_shape=(10, 10), k=np.ones((3, 3)),
                method=warmup_method, backend='numba',
            )
            _ = conv(x=x_warmup, batching=True)
        print("  Numba JIT warmup complete")
    except Exception:
        print("  Numba not available, skipping warmup")

    all_results = []
    sweep_results = []
    grid_results = []

    if not args.grid_only:
        print("\n" + "="*60)
        print("SCALING SWEEPS")
        print("="*60)
        sweep_results = run_scaling_sweeps(
            method_backends, min_time=args.min_time, quick=args.quick,
        )
        all_results.extend(sweep_results)

    if not args.sweep_only:
        print("\n" + "="*60)
        print("GRID SEARCH")
        print("="*60)
        grid_results = run_grid_search(
            method_backends, min_time=args.min_time, quick=args.quick,
        )
        all_results.extend(grid_results)

    ## Save results to separate files per run type (avoids overwrites)
    if not args.grid_only and sweep_results:
        sweep_path = output_dir / 'benchmark_sweeps.json'
        with open(sweep_path, 'w') as f:
            json.dump(sweep_results, f, indent=2, default=str)
        print(f"\nSweep results saved to: {sweep_path}")

    if not args.sweep_only and grid_results:
        grid_path = output_dir / 'benchmark_grid.json'
        with open(grid_path, 'w') as f:
            json.dump(grid_results, f, indent=2, default=str)
        print(f"\nGrid results saved to: {grid_path}")

    ## Also save combined file
    results_path = output_dir / 'benchmark_comprehensive.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"All results saved to: {results_path}")

    ## Plot
    if not args.no_plot:
        print("\nGenerating plots...")
        plot_scaling_sweeps(all_results, output_dir)
        plot_grid_heatmap(all_results, output_dir)

    ## Print top-level summary
    print(f"\nTotal benchmarks run: {len(all_results)}")


if __name__ == '__main__':
    main()
