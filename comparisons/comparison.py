#!/usr/bin/env python3
"""
compare_predictions.py
Analyzes model predictions vs ground truth compensation movements.

Usage:
    python compare_predictions.py ground_truth.csv model_predictions.csv
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath):
    """Load CSV with pic_number, delta_azimuth, delta_elevation"""
    df = pd.read_csv(filepath)
    
    # Ensure columns are correct
    required = ['pic_number', 'delta_azimuth', 'delta_elevation']
    if not all(col in df.columns for col in required):
        raise ValueError(f"CSV must have columns: {required}")
    
    return df


def compute_metrics(gt_df, pred_df):
    """Compute error metrics between ground truth and predictions"""
    
    # Merge on pic_number
    merged = pd.merge(gt_df, pred_df, on='pic_number', suffixes=('_gt', '_pred'))
    
    if len(merged) == 0:
        raise ValueError("No matching pic_numbers between ground truth and predictions!")
    
    # Compute errors
    err_az = merged['delta_azimuth_pred'] - merged['delta_azimuth_gt']
    err_el = merged['delta_elevation_pred'] - merged['delta_elevation_gt']
    
    metrics = {
        'n_samples': len(merged),
        'mae_az': np.abs(err_az).mean(),
        'mae_el': np.abs(err_el).mean(),
        'rmse_az': np.sqrt((err_az**2).mean()),
        'rmse_el': np.sqrt((err_el**2).mean()),
        'bias_az': err_az.mean(),
        'bias_el': err_el.mean(),
        'std_az': err_az.std(),
        'std_el': err_el.std(),
        'max_err_az': np.abs(err_az).max(),
        'max_err_el': np.abs(err_el).max(),
    }
    
    return metrics, merged, err_az, err_el


def print_report(metrics):
    """Print formatted metrics report"""
    print("\n" + "="*70)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*70)
    print(f"\nSamples analyzed: {metrics['n_samples']}")
    
    print("\n--- AZIMUTH ---")
    print(f"  MAE:       {metrics['mae_az']:.3f}°")
    print(f"  RMSE:      {metrics['rmse_az']:.3f}°")
    print(f"  Bias:      {metrics['bias_az']:+.3f}°")
    print(f"  Std Dev:   {metrics['std_az']:.3f}°")
    print(f"  Max Error: {metrics['max_err_az']:.3f}°")
    
    print("\n--- ELEVATION ---")
    print(f"  MAE:       {metrics['mae_el']:.3f}°")
    print(f"  RMSE:      {metrics['rmse_el']:.3f}°")
    print(f"  Bias:      {metrics['bias_el']:+.3f}°")
    print(f"  Std Dev:   {metrics['std_el']:.3f}°")
    print(f"  Max Error: {metrics['max_err_el']:.3f}°")
    
    print("\n--- COMBINED ---")
    mae_combined = (metrics['mae_az'] + metrics['mae_el']) / 2
    print(f"  Mean MAE:  {mae_combined:.3f}°")
    
    # Check spec
    spec_tolerance = 0.20  # Your ±0.2° requirement
    within_spec_az = metrics['mae_az'] < spec_tolerance and abs(metrics['bias_az']) < spec_tolerance
    within_spec_el = metrics['mae_el'] < spec_tolerance and abs(metrics['bias_el']) < spec_tolerance
    
    print(f"\n--- SPEC CHECK (±{spec_tolerance}°) ---")
    print(f"  Azimuth:   {'✓ PASS' if within_spec_az else '✗ FAIL'}")
    print(f"  Elevation: {'✓ PASS' if within_spec_el else '✗ FAIL'}")
    
    if within_spec_az and within_spec_el:
        print("\n  🎯 MODEL MEETS SPECIFICATION!")
    else:
        print("\n  ⚠ Model outside specification limits")
    
    print("="*70 + "\n")


def plot_results(merged, err_az, err_el, save_path='comparison_plot.png'):
    """Create visualization of results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model vs Ground Truth Comparison', fontsize=16)
    
    # Azimuth: Predicted vs Ground Truth
    ax = axes[0, 0]
    ax.scatter(merged['delta_azimuth_gt'], merged['delta_azimuth_pred'], alpha=0.6)
    ax.plot([merged['delta_azimuth_gt'].min(), merged['delta_azimuth_gt'].max()],
            [merged['delta_azimuth_gt'].min(), merged['delta_azimuth_gt'].max()],
            'r--', label='Perfect prediction')
    ax.set_xlabel('Ground Truth Azimuth (°)')
    ax.set_ylabel('Predicted Azimuth (°)')
    ax.set_title('Azimuth: Prediction vs Ground Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Elevation: Predicted vs Ground Truth
    ax = axes[0, 1]
    ax.scatter(merged['delta_elevation_gt'], merged['delta_elevation_pred'], alpha=0.6)
    ax.plot([merged['delta_elevation_gt'].min(), merged['delta_elevation_gt'].max()],
            [merged['delta_elevation_gt'].min(), merged['delta_elevation_gt'].max()],
            'r--', label='Perfect prediction')
    ax.set_xlabel('Ground Truth Elevation (°)')
    ax.set_ylabel('Predicted Elevation (°)')
    ax.set_title('Elevation: Prediction vs Ground Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Azimuth error over sequence
    ax = axes[1, 0]
    ax.plot(merged['pic_number'], err_az, 'o-', alpha=0.6)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.axhline(0.20, color='orange', linestyle=':', alpha=0.5, label='±0.2° spec')
    ax.axhline(-0.20, color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('Picture Number')
    ax.set_ylabel('Error (°)')
    ax.set_title('Azimuth Error Over Sequence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Elevation error over sequence
    ax = axes[1, 1]
    ax.plot(merged['pic_number'], err_el, 'o-', alpha=0.6, color='orange')
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.axhline(0.20, color='orange', linestyle=':', alpha=0.5, label='±0.2° spec')
    ax.axhline(-0.20, color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('Picture Number')
    ax.set_ylabel('Error (°)')
    ax.set_title('Elevation Error Over Sequence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved: {save_path}")
    
    # Show plot
    try:
        plt.show()
    except:
        print("(Display not available, plot saved to file)")


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_predictions.py ground_truth.csv model_predictions.csv")
        sys.exit(1)
    
    gt_file = sys.argv[1]
    pred_file = sys.argv[2]
    
    print(f"Loading ground truth: {gt_file}")
    gt_df = load_data(gt_file)
    print(f"  → {len(gt_df)} samples")
    
    print(f"Loading predictions: {pred_file}")
    pred_df = load_data(pred_file)
    print(f"  → {len(pred_df)} samples")
    
    # Compute metrics
    metrics, merged, err_az, err_el = compute_metrics(gt_df, pred_df)
    
    # Print report
    print_report(metrics)
    
    # Save detailed results
    output_csv = 'comparison_detailed.csv'
    merged['error_az'] = err_az
    merged['error_el'] = err_el
    merged.to_csv(output_csv, index=False)
    print(f"Detailed comparison saved: {output_csv}")
    
    # Plot
    plot_results(merged, err_az, err_el)


if __name__ == "__main__":
    main()