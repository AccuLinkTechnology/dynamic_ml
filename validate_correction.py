#!/usr/bin/env python3
"""
validate_correction.py
Tests the model correction against your ground truth data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_pid import create_calibrated_corrector


def validate_correction(gt_file, pred_file):
    """Validate correction performance on calibration data"""
    
    # Load data
    gt_df = pd.read_csv(gt_file)
    pred_df = pd.read_csv(pred_file)
    merged = pd.merge(gt_df, pred_df, on='pic_number', suffixes=('_gt', '_pred'))
    
    print("="*70)
    print("MODEL CORRECTION VALIDATION")
    print("="*70)
    print(f"\nSamples: {len(merged)}")
    
    # Create corrector
    corrector = create_calibrated_corrector()
    
    # Apply correction
    corrected_az = []
    corrected_el = []
    
    for _, row in merged.iterrows():
        az, el = corrector.correct(
            row['delta_azimuth_pred'],
            row['delta_elevation_pred'],
            enable_pid=False  # Linear correction only
        )
        corrected_az.append(az)
        corrected_el.append(el)
    
    merged['corrected_az'] = corrected_az
    merged['corrected_el'] = corrected_el
    
    # Compute errors
    merged['error_raw_az'] = merged['delta_azimuth_pred'] - merged['delta_azimuth_gt']
    merged['error_raw_el'] = merged['delta_elevation_pred'] - merged['delta_elevation_gt']
    merged['error_corr_az'] = merged['corrected_az'] - merged['delta_azimuth_gt']
    merged['error_corr_el'] = merged['corrected_el'] - merged['delta_elevation_gt']
    
    # Statistics
    print("\n" + "="*70)
    print("BEFORE CORRECTION")
    print("="*70)
    print(f"MAE Az:  {np.abs(merged['error_raw_az']).mean():.3f}°")
    print(f"MAE El:  {np.abs(merged['error_raw_el']).mean():.3f}°")
    print(f"Bias Az: {merged['error_raw_az'].mean():+.3f}°")
    print(f"Bias El: {merged['error_raw_el'].mean():+.3f}°")
    
    within_spec_raw = (
        (np.abs(merged['error_raw_az']) < 0.2) & 
        (np.abs(merged['error_raw_el']) < 0.2)
    )
    print(f"\nWithin ±0.2° spec: {within_spec_raw.sum()}/{len(merged)} ({100*within_spec_raw.mean():.1f}%)")
    
    print("\n" + "="*70)
    print("AFTER CORRECTION")
    print("="*70)
    print(f"MAE Az:  {np.abs(merged['error_corr_az']).mean():.3f}°")
    print(f"MAE El:  {np.abs(merged['error_corr_el']).mean():.3f}°")
    print(f"Bias Az: {merged['error_corr_az'].mean():+.3f}°")
    print(f"Bias El: {merged['error_corr_el'].mean():+.3f}°")
    print(f"Max Err Az: {np.abs(merged['error_corr_az']).max():.3f}°")
    print(f"Max Err El: {np.abs(merged['error_corr_el']).max():.3f}°")
    
    within_spec_corr = (
        (np.abs(merged['error_corr_az']) < 0.2) & 
        (np.abs(merged['error_corr_el']) < 0.2)
    )
    print(f"\nWithin ±0.2° spec: {within_spec_corr.sum()}/{len(merged)} ({100*within_spec_corr.mean():.1f}%)")
    
    # Improvement
    improvement_az = (1 - np.abs(merged['error_corr_az']).mean() / np.abs(merged['error_raw_az']).mean()) * 100
    improvement_el = (1 - np.abs(merged['error_corr_el']).mean() / np.abs(merged['error_raw_el']).mean()) * 100
    
    print("\n" + "="*70)
    print("IMPROVEMENT")
    print("="*70)
    print(f"Azimuth MAE:    {improvement_az:+.1f}%")
    print(f"Elevation MAE:  {improvement_el:+.1f}%")
    print(f"Spec compliance: {within_spec_raw.sum()} → {within_spec_corr.sum()} samples")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Correction Performance', fontsize=16)
    
    # Azimuth: Before vs After
    ax = axes[0, 0]
    ax.scatter(merged['delta_azimuth_gt'], merged['delta_azimuth_pred'], 
               alpha=0.5, label='Raw model', s=30)
    ax.scatter(merged['delta_azimuth_gt'], merged['corrected_az'],
               alpha=0.5, label='Corrected', s=30)
    lims = [merged['delta_azimuth_gt'].min(), merged['delta_azimuth_gt'].max()]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Ground Truth (°)')
    ax.set_ylabel('Predicted (°)')
    ax.set_title('Azimuth: Before vs After Correction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Elevation: Before vs After
    ax = axes[0, 1]
    ax.scatter(merged['delta_elevation_gt'], merged['delta_elevation_pred'],
               alpha=0.5, label='Raw model', s=30)
    ax.scatter(merged['delta_elevation_gt'], merged['corrected_el'],
               alpha=0.5, label='Corrected', s=30)
    lims = [merged['delta_elevation_gt'].min(), merged['delta_elevation_gt'].max()]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Ground Truth (°)')
    ax.set_ylabel('Predicted (°)')
    ax.set_title('Elevation: Before vs After Correction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Azimuth error distribution
    ax = axes[1, 0]
    ax.hist(merged['error_raw_az'], bins=30, alpha=0.5, label='Raw error')
    ax.hist(merged['error_corr_az'], bins=30, alpha=0.5, label='Corrected error')
    ax.axvline(-0.2, color='orange', linestyle=':', alpha=0.5)
    ax.axvline(0.2, color='orange', linestyle=':', alpha=0.5, label='±0.2° spec')
    ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Error (°)')
    ax.set_ylabel('Count')
    ax.set_title('Azimuth Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Elevation error distribution
    ax = axes[1, 1]
    ax.hist(merged['error_raw_el'], bins=30, alpha=0.5, label='Raw error')
    ax.hist(merged['error_corr_el'], bins=30, alpha=0.5, label='Corrected error')
    ax.axvline(-0.2, color='orange', linestyle=':', alpha=0.5)
    ax.axvline(0.2, color='orange', linestyle=':', alpha=0.5, label='±0.2° spec')
    ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Error (°)')
    ax.set_ylabel('Count')
    ax.set_title('Elevation Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correction_validation.png', dpi=150)
    print("\n" + "="*70)
    print("Plot saved: correction_validation.png")
    print("="*70)
    
    try:
        plt.show()
    except:
        pass
    
    return merged


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python validate_correction.py ground_truth.csv model_predictions.csv")
        sys.exit(1)
    
    validate_correction(sys.argv[1], sys.argv[2])