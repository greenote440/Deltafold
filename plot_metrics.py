#!/usr/bin/env python
"""Plot ARI and TM-rho evolution across epochs."""

import gzip
import json
from pathlib import Path
import sys

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not available, using text-based visualization")

def parse_training_log(log_path):
    """Parse training log and extract epoch records."""
    records = []
    with gzip.open(log_path, 'rt') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass

    # Extract epoch records
    epochs_data = {}
    for rec in records:
        if rec.get('t') == 'epoch':
            ep = rec.get('ep')
            if ep:
                epochs_data[ep] = rec

    return epochs_data

def plot_metrics(epochs_data, output_path):
    """Plot ARI and TM-rho evolution."""

    if not HAS_MATPLOTLIB:
        print_text_plot(epochs_data)
        return

    # Extract data
    epochs = sorted(epochs_data.keys())
    ari_values = [epochs_data[ep].get('ari') for ep in epochs]
    tm_rho_values = []
    hdbscan_ari_values = []

    for ep in epochs:
        eval_metrics = epochs_data[ep].get('eval', {})
        tm_rho = eval_metrics.get('tm_rho')
        hdbscan_ari = eval_metrics.get('hdbscan_ari')
        tm_rho_values.append(tm_rho)
        hdbscan_ari_values.append(hdbscan_ari)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: ARI (k-means)
    ax = axes[0]
    ax.plot(epochs, ari_values, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('ARI (k-means)', fontsize=11, fontweight='bold')
    ax.set_title('Adjusted Rand Index\n(K-Means Clustering)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.15, 0.25])
    for i, (ep, val) in enumerate(zip(epochs, ari_values)):
        if val is not None:
            ax.text(ep, val + 0.005, f'{val:.4f}', ha='center', fontsize=9)

    # Plot 2: HDBSCAN ARI
    ax = axes[1]
    valid_epochs = [ep for ep, val in zip(epochs, hdbscan_ari_values) if val is not None]
    valid_hdbscan = [val for val in hdbscan_ari_values if val is not None]
    if valid_hdbscan:
        ax.plot(valid_epochs, valid_hdbscan, 's-', linewidth=2, markersize=8, color='#A23B72')
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('ARI (HDBSCAN)', fontsize=11, fontweight='bold')
        ax.set_title('HDBSCAN ARI\n(Density-Based Clustering)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        for ep, val in zip(valid_epochs, valid_hdbscan):
            if val is not None:
                ax.text(ep, val + 0.005, f'{val:.4f}', ha='center', fontsize=9)

    # Plot 3: TM-rho
    ax = axes[2]
    valid_epochs = [ep for ep, val in zip(epochs, tm_rho_values) if val is not None]
    valid_tm_rho = [val for val in tm_rho_values if val is not None]
    if valid_tm_rho:
        ax.plot(valid_epochs, valid_tm_rho, '^-', linewidth=2, markersize=8, color='#F18F01')
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('TM-rho', fontsize=11, fontweight='bold')
        ax.set_title('TM-Score Correlation\n(Structural Alignment)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.81, -0.77])
        for ep, val in zip(valid_epochs, valid_tm_rho):
            if val is not None:
                ax.text(ep, val - 0.005, f'{val:.4f}', ha='center', fontsize=9, va='top')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")
    plt.close()

def print_text_plot(epochs_data):
    """Print text-based visualization of metrics."""
    epochs = sorted(epochs_data.keys())

    print("\n" + "=" * 80)
    print("METRIC EVOLUTION")
    print("=" * 80)
    print(f"{'Epoch':<8} {'ARI (k-means)':<18} {'HDBSCAN ARI':<18} {'TM-rho':<18}")
    print("-" * 80)

    for ep in epochs:
        data = epochs_data[ep]
        ari = data.get('ari')
        eval_metrics = data.get('eval', {})
        hdbscan_ari = eval_metrics.get('hdbscan_ari')
        tm_rho = eval_metrics.get('tm_rho')

        ari_str = f"{ari:.4f}" if ari is not None else "N/A"
        hdbscan_str = f"{hdbscan_ari:.4f}" if hdbscan_ari is not None else "N/A"
        tm_rho_str = f"{tm_rho:.4f}" if tm_rho is not None else "N/A"

        print(f"{ep:<8} {ari_str:<18} {hdbscan_str:<18} {tm_rho_str:<18}")

    print("=" * 80)

if __name__ == '__main__':
    log_path = Path('/Users/macbook/Documents/Deltafold/checkpoints/training_log.jsonl.gz')

    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        sys.exit(1)

    print("Parsing training log...")
    epochs_data = parse_training_log(log_path)

    if not epochs_data:
        print("Error: No epoch records found in log")
        sys.exit(1)

    print(f"Found {len(epochs_data)} epochs")

    # Plot
    output_path = Path('/Users/macbook/Documents/Deltafold/checkpoints/metrics_evolution.png')
    plot_metrics(epochs_data, output_path)

    # Print text summary
    print_text_plot(epochs_data)

