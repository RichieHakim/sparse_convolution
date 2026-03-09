"""
Analyze and visualize benchmark results from benchmark_comprehensive.py.

Produces:
1. A multi-panel scaling figure (call time curves for all 6 sweeps)
2. A grid winner heatmap (batch×density panels for each x_shape×k_shape)
3. A summary table printed to stdout

Usage:
    python analyze_results.py [--results-dir benchmarks/results]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


## ---------------------------------------------------------------------------
## Data loading
## ---------------------------------------------------------------------------

def load_results(results_dir):
    """
    Load all benchmark results from JSON files in the results directory.
    Merges all ``benchmark_*.json`` files found.

    Args:
        results_dir (Path):
            Directory containing benchmark JSON files.

    Returns:
        (list):
            results (list):
                List of benchmark result dicts.
    """
    ## Prefer separate sweep/grid files; fall back to comprehensive
    sweep_path = results_dir / 'benchmark_sweeps.json'
    grid_path = results_dir / 'benchmark_grid.json'
    comp_path = results_dir / 'benchmark_comprehensive.json'

    all_results = []
    if sweep_path.exists() or grid_path.exists():
        for filepath in [sweep_path, grid_path]:
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                print(f"  Loaded {len(data)} entries from {filepath.name}")
                all_results.extend(data)
    elif comp_path.exists():
        with open(comp_path, 'r') as f:
            all_results = json.load(f)
        print(f"  Loaded {len(all_results)} entries from {comp_path.name}")

    return all_results


## ---------------------------------------------------------------------------
## Scaling sweep figure
## ---------------------------------------------------------------------------

## Color palette: distinct colors for each method+backend
METHOD_COLORS = {
    'precomputed+numpy':      '#1f77b4',  ## blue
    'precomputed+torch':      '#aec7e8',  ## light blue
    'lazy+numpy':             '#2ca02c',  ## green
    'lazy+torch':             '#98df8a',  ## light green
    'gather_scatter+numpy':   '#d62728',  ## red
    'gather_scatter+numba':   '#ff7f0e',  ## orange
    'gather_scatter+torch':   '#ffbb78',  ## light orange
    'precomputed+torch+cuda': '#9467bd',  ## purple
    'lazy+torch+cuda':        '#c5b0d5',  ## light purple
    'gather_scatter+torch+cuda': '#e377c2',  ## pink
}

METHOD_MARKERS = {
    'precomputed+numpy':      's',
    'precomputed+torch':      's',
    'lazy+numpy':             '^',
    'lazy+torch':             '^',
    'gather_scatter+numpy':   'o',
    'gather_scatter+numba':   'D',
    'gather_scatter+torch':   'o',
}

SWEEP_TITLES = {
    'batch_scaling':            'Batch scaling (d=0.01, 100²×5²)',
    'batch_scaling_high_density': 'Batch scaling (d=0.1, 100²×5²)',
    'batch_scaling_very_sparse':  'Batch scaling (d=0.001, 200²×5²)',
    'density_scaling':          'Density scaling (b=100, 100²×5²)',
    'img_size_scaling':         'Image size scaling (b=50, d=0.01, 5²)',
    'kernel_size_scaling':      'Kernel size scaling (b=50, d=0.01, 100²)',
}

SWEEP_XLABELS = {
    'batch_scaling':            'batch size',
    'batch_scaling_high_density': 'batch size',
    'batch_scaling_very_sparse':  'batch size',
    'density_scaling':          'density',
    'img_size_scaling':         'image side length',
    'kernel_size_scaling':      'kernel side length',
}


def plot_scaling_figure(results, output_dir):
    """
    Multi-panel scaling curves: one subplot per sweep, call time only.

    Args:
        results (list):
            Full benchmark results.
        output_dir (Path):
            Where to save the figure.
    """
    ## Filter to sweep results only
    sweep_results = [r for r in results if r.get('sweep', '') != 'grid']
    if not sweep_results:
        print("No sweep results found.")
        return

    ## Group by sweep name
    sweeps = {}
    for r in sweep_results:
        sweep = r['sweep']
        if sweep not in sweeps:
            sweeps[sweep] = []
        sweeps[sweep].append(r)

    sweep_order = [
        'batch_scaling', 'batch_scaling_high_density', 'batch_scaling_very_sparse',
        'density_scaling', 'img_size_scaling', 'kernel_size_scaling',
    ]
    sweep_names = [s for s in sweep_order if s in sweeps]
    n_sweeps = len(sweep_names)

    if n_sweeps == 0:
        return

    ## Layout: 2 rows × 3 cols (or fewer if less sweeps)
    n_cols = min(3, n_sweeps)
    n_rows = (n_sweeps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    if n_sweeps == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, sweep_name in enumerate(sweep_names):
        ax = axes[i]
        sweep_data = sweeps[sweep_name]

        ## Group by method+backend
        combos = {}
        for r in sweep_data:
            key = f"{r['method']}+{r['backend']}"
            if r['device'] != 'cpu':
                key += f"+{r['device']}"
            if key not in combos:
                combos[key] = {'x': [], 'call': []}
            combos[key]['x'].append(r['sweep_val'])
            combos[key]['call'].append(r['call_mean'])

        for key, data in sorted(combos.items()):
            order = np.argsort(data['x'])
            x = np.array(data['x'])[order]
            y = np.array(data['call'])[order]
            color = METHOD_COLORS.get(key, '#333333')
            marker = METHOD_MARKERS.get(key, 'o')
            ax.loglog(x, y, marker=marker, color=color, label=key,
                      markersize=4, linewidth=1.5, alpha=0.85)

        title = SWEEP_TITLES.get(sweep_name, sweep_name)
        xlabel = SWEEP_XLABELS.get(sweep_name, sweep_name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('call time (s)', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=8)

    ## Single shared legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    ## Hide unused axes
    for j in range(n_sweeps, len(axes)):
        axes[j].set_visible(False)

    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8,
              bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Sparse Convolution: Scaling Benchmarks (call time)', fontsize=13, y=1.01)
    plt.tight_layout()

    filepath = output_dir / 'figure_scaling_all.png'
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


## ---------------------------------------------------------------------------
## Grid winner heatmap
## ---------------------------------------------------------------------------

def plot_grid_heatmap(results, output_dir):
    """
    Grid winner heatmap: for each (x_shape, k_shape) panel, show a
    batch_size × density heatmap colored by the winner method+backend.

    Args:
        results (list):
            Full benchmark results.
        output_dir (Path):
            Where to save the figure.
    """
    grid_results = [r for r in results if r.get('sweep') == 'grid']
    if not grid_results:
        print("No grid results found.")
        return

    ## Collect all unique method+backend labels seen
    all_labels = set()
    ## Group by (x_shape, k_shape) → (batch, density) → {label: time}
    panels = {}
    for r in grid_results:
        xs = tuple(r['x_shape'])
        ks = tuple(r['k_shape'])
        panel_key = (xs, ks)
        if panel_key not in panels:
            panels[panel_key] = {}

        cell_key = (r['batch_size'], r['density'])
        if cell_key not in panels[panel_key]:
            panels[panel_key][cell_key] = {}

        label = f"{r['method']}+{r['backend']}"
        if r['device'] != 'cpu':
            label += f"+{r['device']}"
        panels[panel_key][cell_key][label] = r['total_mean']
        all_labels.add(label)

    ## Assign integer codes to labels
    ## Group by method for better color coherence
    label_order = [
        'precomputed+numpy', 'precomputed+torch',
        'lazy+numpy', 'lazy+torch',
        'gather_scatter+numpy', 'gather_scatter+numba', 'gather_scatter+torch',
        'precomputed+torch+cuda', 'lazy+torch+cuda', 'gather_scatter+torch+cuda',
    ]
    label_order = [l for l in label_order if l in all_labels]
    label_to_idx = {l: i for i, l in enumerate(label_order)}

    ## Colors for heatmap
    heatmap_colors = [METHOD_COLORS.get(l, '#cccccc') for l in label_order]
    cmap = ListedColormap(heatmap_colors)

    ## Collect unique batch_sizes and densities across all panels
    all_batches = sorted(set(ck[0] for panel in panels.values() for ck in panel))
    all_densities = sorted(set(ck[1] for panel in panels.values() for ck in panel))

    panel_keys = sorted(panels.keys())
    n_panels = len(panel_keys)
    n_cols = min(n_panels, 4)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes) if n_panels > 1 else axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for pi, panel_key in enumerate(panel_keys):
        ax = axes_flat[pi]
        panel = panels[panel_key]

        ## Build winner matrix and speedup annotation matrix
        winner_matrix = np.full((len(all_batches), len(all_densities)), np.nan)
        annot_matrix = [['' for _ in range(len(all_densities))] for _ in range(len(all_batches))]

        for bi, batch in enumerate(all_batches):
            for di, density in enumerate(all_densities):
                cell_key = (batch, density)
                if cell_key not in panel:
                    continue
                timings = panel[cell_key]
                winner = min(timings, key=timings.get)
                winner_time = timings[winner]
                winner_matrix[bi, di] = label_to_idx.get(winner, -1)

                ## Annotation: winner short name + time
                short = winner.replace('precomputed', 'pre').replace('gather_scatter', 'gs')
                annot_matrix[bi][di] = f"{short}\n{winner_time:.3f}s"

        im = ax.imshow(winner_matrix, cmap=cmap, vmin=-0.5,
                       vmax=len(label_order) - 0.5, aspect='auto',
                       interpolation='nearest')

        ## Add text annotations
        for bi in range(len(all_batches)):
            for di in range(len(all_densities)):
                if annot_matrix[bi][di]:
                    ax.text(di, bi, annot_matrix[bi][di], ha='center', va='center',
                            fontsize=6, color='white', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.4))

        ax.set_xticks(range(len(all_densities)))
        ax.set_xticklabels([f'{d}' for d in all_densities], fontsize=7)
        ax.set_yticks(range(len(all_batches)))
        ax.set_yticklabels([str(b) for b in all_batches], fontsize=7)
        ax.set_xlabel('density', fontsize=9)
        ax.set_ylabel('batch size', fontsize=9)
        xs, ks = panel_key
        ax.set_title(f'x={xs[0]}², k={ks[0]}²', fontsize=10)

    ## Hide unused axes
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    ## Legend: colored patches for each method
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=METHOD_COLORS.get(l, '#ccc'), label=l)
        for l in label_order
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=8,
              bbox_to_anchor=(0.5, -0.04))

    fig.suptitle('Grid Search: Fastest Method per Configuration (total time)', fontsize=13, y=1.01)
    plt.tight_layout()

    filepath = output_dir / 'figure_grid_winners.png'
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


## ---------------------------------------------------------------------------
## Summary statistics
## ---------------------------------------------------------------------------

def print_summary(results):
    """
    Print overall summary of which methods win and where.

    Args:
        results (list):
            Full benchmark results.
    """
    grid_results = [r for r in results if r.get('sweep') == 'grid']
    if not grid_results:
        print("No grid results to summarize.")
        return

    ## Group by config → winner
    configs = {}
    for r in grid_results:
        config_key = (tuple(r['x_shape']), tuple(r['k_shape']),
                      r['batch_size'], r['density'])
        if config_key not in configs:
            configs[config_key] = {}
        label = f"{r['method']}+{r['backend']}"
        if r['device'] != 'cpu':
            label += f"+{r['device']}"
        configs[config_key][label] = r['total_mean']

    ## Count wins and average speedup over second-best
    win_counts = {}
    speedups = {}
    for config_key, timings in configs.items():
        sorted_methods = sorted(timings.items(), key=lambda x: x[1])
        winner, winner_time = sorted_methods[0]
        second_time = sorted_methods[1][1] if len(sorted_methods) > 1 else winner_time

        win_counts[winner] = win_counts.get(winner, 0) + 1
        if winner not in speedups:
            speedups[winner] = []
        speedups[winner].append(second_time / winner_time)

    print(f"\n{'='*70}")
    print("METHOD WIN SUMMARY (total time = init + call)")
    print(f"{'='*70}")
    print(f"{'Method':<30s} {'Wins':>6s} {'Avg speedup vs 2nd':>20s}")
    print('-' * 60)
    for method in sorted(win_counts, key=win_counts.get, reverse=True):
        avg_speedup = np.mean(speedups[method])
        print(f"{method:<30s} {win_counts[method]:>6d} {avg_speedup:>19.2f}x")

    ## Also show win regions
    print(f"\n{'='*70}")
    print("WIN REGIONS")
    print(f"{'='*70}")
    for method in sorted(win_counts, key=win_counts.get, reverse=True):
        winning_configs = [
            ck for ck, timings in configs.items()
            if min(timings, key=timings.get) == method
        ]
        batches = [ck[2] for ck in winning_configs]
        densities = [ck[3] for ck in winning_configs]
        print(f"\n{method}:")
        print(f"  batch range: {min(batches)}-{max(batches)}")
        print(f"  density range: {min(densities)}-{max(densities)}")


## ---------------------------------------------------------------------------
## Main
## ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--results-dir', type=str,
                        default=str(Path(__file__).parent / 'results'),
                        help='Directory containing benchmark results')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)
    print(f"Loaded {len(results)} benchmark results from {results_dir}")

    plot_scaling_figure(results, results_dir)
    plot_grid_heatmap(results, results_dir)
    print_summary(results)


if __name__ == '__main__':
    main()
