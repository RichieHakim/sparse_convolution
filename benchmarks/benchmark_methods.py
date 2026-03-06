"""
Comprehensive benchmark comparing 'lazy' vs 'precomputed' convolution methods.

* ``'lazy'`` uses sparse COO broadcasting on every call. Cost scales with
  nnz(x) * nnz(k). Best for sparse inputs (density < ~0.1).
* ``'precomputed'`` builds a Toeplitz matrix at init, then uses sparse matmul.
  Cost is dominated by the one-time build; per-call matmul is density-independent.

Sweeps one factor at a time while holding others at defaults:
  1. Input density
  2. Input dimension (square images, varying side length)
  3. Input height (x_shape[0])
  4. Input width  (x_shape[1])
  5. Kernel height (k_shape[0])
  6. Kernel width  (k_shape[1])
  7. Batch size
  8. Amortization (repeated calls on same object)

Results are printed to stdout and saved to benchmarks/results_methods.json.

Usage:
    python benchmarks/benchmark_methods.py
"""

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse

from sparse_convolution import Toeplitz_convolution2d


## ---------------------------------------------------------------------------
## Timing helper
## ---------------------------------------------------------------------------

def time_method(
    method: str,
    x_shape: tuple,
    k_shape: tuple,
    density: float,
    batching: bool,
    batch_size: int,
    n_calls: int,
    seed: int = 0,
) -> dict:
    """
    Time initialization and execution for a single configuration.
    Returns:
        (dict):
            result (dict):
                Contains 'init_s', 'call1_s', 'call2plus_avg_s', 'total_s'.
    """
    np.random.seed(seed)
    kernel = np.random.rand(*k_shape)

    ## Build sparse input
    if batching:
        x = scipy.sparse.random(
            m=batch_size,
            n=x_shape[0] * x_shape[1],
            density=density,
            format='csr',
            random_state=seed,
        )
    else:
        x = scipy.sparse.random(
            m=x_shape[0],
            n=x_shape[1],
            density=density,
            format='csr',
            random_state=seed,
        )

    ## Time init
    tic = time.perf_counter()
    conv = Toeplitz_convolution2d(
        x_shape=x_shape, k=kernel, mode='same', method=method,
    )
    t_init = time.perf_counter() - tic

    ## Time first call (lazy: broadcasting + Toeplitz build; precomputed: matmul)
    tic = time.perf_counter()
    conv(x=x, batching=batching)
    t_call1 = time.perf_counter() - tic

    ## Time subsequent calls
    n_subsequent = max(n_calls - 1, 1)
    tic = time.perf_counter()
    for _ in range(n_subsequent):
        conv(x=x, batching=batching)
    t_subsequent = time.perf_counter() - tic

    t_total = t_init + t_call1 + t_subsequent

    return {
        'init_s': round(t_init, 6),
        'call1_s': round(t_call1, 6),
        'call2plus_avg_s': round(t_subsequent / n_subsequent, 6),
        'total_s': round(t_total, 6),
    }


## ---------------------------------------------------------------------------
## Sweep runner
## ---------------------------------------------------------------------------

HEADER_COLS = f"{'Value':<12} | {'Method':<12} | {'Init (s)':<12} | {'Call 1 (s)':<12} | {'Call 2+ (s)':<13} | {'Total (s)':<12}"
HEADER_SEP = "-" * len(HEADER_COLS)


def run_sweep(
    sweep_name: str,
    sweep_values: list,
    sweep_key: str,
    defaults: dict,
) -> list:
    """
    Run a single sweep, varying one parameter while holding others at defaults.
    """
    print(f"\n{'=' * len(HEADER_COLS)}")
    print(f"Sweep: {sweep_name}  (varying '{sweep_key}')")
    print(f"{'=' * len(HEADER_COLS)}")
    print(HEADER_COLS)
    print(HEADER_SEP)

    results = []
    for val in sweep_values:
        params = {**defaults, sweep_key: val}

        ## Adjust x_shape / k_shape if the sweep key is a sub-dimension
        if sweep_key == 'x_height':
            params['x_shape'] = (val, params['x_shape'][1])
        elif sweep_key == 'x_width':
            params['x_shape'] = (params['x_shape'][0], val)
        elif sweep_key == 'x_dim':
            params['x_shape'] = (val, val)
        elif sweep_key == 'k_height':
            params['k_shape'] = (val, params['k_shape'][1])
        elif sweep_key == 'k_width':
            params['k_shape'] = (params['k_shape'][0], val)
        elif sweep_key == 'batch_size':
            params['batching'] = val > 0
            params['batch_size'] = max(val, 1)

        ## Extract time_method params
        tm_params = {
            k: params[k] for k in
            ['x_shape', 'k_shape', 'density', 'batching', 'batch_size', 'n_calls', 'seed']
        }

        for method in ['lazy', 'precomputed']:
            r = time_method(method=method, **tm_params)
            print(
                f"{val!s:<12} | {method:<12} | {r['init_s']:<12.6f} | "
                f"{r['call1_s']:<12.6f} | {r['call2plus_avg_s']:<13.6f} | {r['total_s']:<12.6f}"
            )
            results.append({sweep_key: val, 'method': method, **r})

    return results


## ---------------------------------------------------------------------------
## Main
## ---------------------------------------------------------------------------

def main():
    defaults = {
        'x_shape': (100, 100),
        'k_shape': (5, 5),
        'density': 0.001,
        'batching': False,
        'batch_size': 1,
        'n_calls': 5,
        'seed': 42,
    }

    all_results = {}

    ## 1. Input dimension sweep (square images, small to large)
    all_results['x_dim'] = run_sweep(
        sweep_name="Image dimension (square, density=0.001)",
        sweep_values=[25, 50, 100, 200, 500, 1000, 2000],
        sweep_key='x_dim',
        defaults=defaults,
    )

    ## 2. Density sweep
    all_results['density'] = run_sweep(
        sweep_name="Input density (100x100)",
        sweep_values=[0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5],
        sweep_key='density',
        defaults=defaults,
    )

    ## 3. Density sweep at large image size
    all_results['density_large'] = run_sweep(
        sweep_name="Input density (1000x1000)",
        sweep_values=[0.0001, 0.001, 0.005, 0.01, 0.05],
        sweep_key='density',
        defaults={**defaults, 'x_shape': (1000, 1000)},
    )

    ## 4. Input height sweep (width fixed at 100)
    all_results['x_height'] = run_sweep(
        sweep_name="Input height (width=100)",
        sweep_values=[10, 50, 100, 200, 500, 1000],
        sweep_key='x_height',
        defaults=defaults,
    )

    ## 5. Input width sweep (height fixed at 100)
    all_results['x_width'] = run_sweep(
        sweep_name="Input width (height=100)",
        sweep_values=[10, 50, 100, 200, 500, 1000],
        sweep_key='x_width',
        defaults=defaults,
    )

    ## 6. Kernel height sweep
    all_results['k_height'] = run_sweep(
        sweep_name="Kernel height (k_shape[0])",
        sweep_values=[2, 3, 5, 10, 20],
        sweep_key='k_height',
        defaults=defaults,
    )

    ## 7. Kernel width sweep
    all_results['k_width'] = run_sweep(
        sweep_name="Kernel width (k_shape[1])",
        sweep_values=[2, 3, 5, 10, 20],
        sweep_key='k_width',
        defaults=defaults,
    )

    ## 8. Batch size sweep (100x100)
    all_results['batch_size'] = run_sweep(
        sweep_name="Batch size (100x100, density=0.001)",
        sweep_values=[1, 5, 10, 50, 100, 500, 2000],
        sweep_key='batch_size',
        defaults={**defaults, 'batching': True, 'batch_size': 1},
    )

    ## 9. Batch size sweep at larger image
    all_results['batch_size_large'] = run_sweep(
        sweep_name="Batch size (500x500, density=0.001)",
        sweep_values=[1, 5, 10, 50, 100],
        sweep_key='batch_size',
        defaults={**defaults, 'x_shape': (500, 500), 'batching': True, 'batch_size': 1},
    )

    ## 10. Amortization sweep (repeated calls)
    all_results['n_calls'] = run_sweep(
        sweep_name="Amortization (number of calls, 100x100)",
        sweep_values=[1, 3, 5, 10, 50],
        sweep_key='n_calls',
        defaults=defaults,
    )

    ## Save to JSON
    filepath_out = str(Path(__file__).parent / "results_methods.json")
    with open(filepath_out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {filepath_out}")


if __name__ == "__main__":
    main()
