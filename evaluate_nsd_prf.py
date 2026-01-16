#!/usr/bin/env python3
"""
Evaluate deepRetinotopy predictions against NSD ground truth pRF data

This script:
1. Runs inference on NSD FreeSurfer data using curvature
2. Compares predictions with ground truth pRF measurements
3. Computes evaluation metrics (correlation, MAE, RMSE)
4. Generates visualization plots

Usage:
    python evaluate_nsd_prf.py --subject subj01 --hemisphere lh --prediction eccentricity
"""

import os
import argparse
import numpy as np
import nibabel as nib
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Set style for plots
sns.set_style("white")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


def load_mgz_data(mgz_path):
    """Load MGZ file and return data array"""
    img = nib.load(mgz_path)
    data = img.get_fdata().squeeze()
    return data


def load_gii_data(gii_path):
    """Load GIFTI file and return data array"""
    img = nib.load(gii_path)
    # Get the first data array (usually the only one)
    data = img.darrays[0].data
    return data


def compute_metrics(pred, gt, mask=None):
    """
    Compute evaluation metrics between prediction and ground truth
    
    Args:
        pred: Predicted values
        gt: Ground truth values
        mask: Optional mask to select valid vertices
    
    Returns:
        dict: Dictionary containing various metrics
    """
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    
    # Remove NaN and Inf values
    valid = np.isfinite(pred) & np.isfinite(gt)
    pred = pred[valid]
    gt = gt[valid]
    
    if len(pred) == 0:
        return {
            'n_vertices': 0,
            'correlation': np.nan,
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan
        }
    
    # Pearson correlation
    corr, p_value = stats.pearsonr(pred, gt)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(pred - gt))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    
    # R-squared
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        'n_vertices': len(pred),
        'correlation': corr,
        'p_value': p_value,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pred_mean': np.mean(pred),
        'pred_std': np.std(pred),
        'gt_mean': np.mean(gt),
        'gt_std': np.std(gt)
    }


def create_scatter_plot(pred, gt, mask, metrics, prediction_type, hemisphere, output_path):
    """Create scatter plot comparing prediction vs ground truth"""
    if mask is not None:
        pred_masked = pred[mask]
        gt_masked = gt[mask]
    else:
        pred_masked = pred
        gt_masked = gt
    
    # Remove invalid values
    valid = np.isfinite(pred_masked) & np.isfinite(gt_masked)
    pred_masked = pred_masked[valid]
    gt_masked = gt_masked[valid]
    
    if len(pred_masked) == 0:
        print(f"Warning: No valid data points for {prediction_type} {hemisphere}")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create hexbin plot for better visualization with many points
    if len(pred_masked) > 1000:
        hexbin = ax.hexbin(gt_masked, pred_masked, gridsize=50, cmap='viridis', 
                          mincnt=1, alpha=0.8)
        plt.colorbar(hexbin, ax=ax, label='Count')
    else:
        ax.scatter(gt_masked, pred_masked, alpha=0.5, s=10)
    
    # Add identity line
    min_val = min(np.min(gt_masked), np.min(pred_masked))
    max_val = max(np.max(gt_masked), np.max(pred_masked))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Identity')
    
    # Add regression line
    z = np.polyfit(gt_masked, pred_masked, 1)
    p = np.poly1d(z)
    ax.plot(gt_masked, p(gt_masked), 'b-', linewidth=2, alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Labels and title
    pred_name = prediction_type.replace('_', ' ').title()
    ax.set_xlabel(f'Ground Truth {pred_name}', fontsize=14)
    ax.set_ylabel(f'Predicted {pred_name}', fontsize=14)
    ax.set_title(f'{pred_name} - {hemisphere.upper()}\n' + 
                 f'r={metrics["correlation"]:.3f}, MAE={metrics["mae"]:.3f}, RMSE={metrics["rmse"]:.3f}',
                 fontsize=16)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    ax.legend(loc='upper left', frameon=False)
    
    # Add grid
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved scatter plot to {output_path}")


def create_distribution_plot(pred, gt, mask, prediction_type, hemisphere, output_path):
    """Create distribution comparison plot"""
    if mask is not None:
        pred_masked = pred[mask]
        gt_masked = gt[mask]
    else:
        pred_masked = pred
        gt_masked = gt
    
    # Remove invalid values
    valid_pred = np.isfinite(pred_masked)
    valid_gt = np.isfinite(gt_masked)
    pred_masked = pred_masked[valid_pred]
    gt_masked = gt_masked[valid_gt]
    
    if len(pred_masked) == 0 or len(gt_masked) == 0:
        print(f"Warning: No valid data for distribution plot {prediction_type} {hemisphere}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram comparison
    ax = axes[0]
    bins = 50
    ax.hist(gt_masked, bins=bins, alpha=0.5, label='Ground Truth', color='blue', density=True)
    ax.hist(pred_masked, bins=bins, alpha=0.5, label='Predicted', color='red', density=True)
    ax.set_xlabel(prediction_type.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title(f'Distribution Comparison - {hemisphere.upper()}', fontsize=16)
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    # Box plot comparison
    ax = axes[1]
    data_to_plot = [gt_masked, pred_masked]
    bp = ax.boxplot(data_to_plot, labels=['Ground Truth', 'Predicted'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    for box in bp['boxes']:
        box.set_alpha(0.5)
    ax.set_ylabel(prediction_type.replace('_', ' ').title(), fontsize=14)
    ax.set_title(f'Value Distribution - {hemisphere.upper()}', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distribution plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate deepRetinotopy on NSD data')
    parser.add_argument('--nsd_dir', type=str, 
                       default='/mnt/external_storage1/natural-scenes-dataset/nsddata/freesurfer',
                       help='Path to NSD FreeSurfer directory')
    parser.add_argument('--subject', type=str, required=True,
                       help='Subject ID (e.g., subj01)')
    parser.add_argument('--hemisphere', type=str, required=True, choices=['lh', 'rh'],
                       help='Hemisphere to process')
    parser.add_argument('--prediction', type=str, required=True,
                       choices=['eccentricity', 'polarAngle', 'pRFsize'],
                       help='Prediction target')
    parser.add_argument('--model_type', type=str, default='baseline',
                       help='Model type (baseline, transolver_optionA, etc.)')
    parser.add_argument('--myelination', type=str, default='False',
                       choices=['True', 'False'],
                       help='Use myelination data')
    parser.add_argument('--output_dir', type=str, default='./nsd_evaluation',
                       help='Output directory for results')
    parser.add_argument('--r2_threshold', type=float, default=0.0,
                       help='R2 threshold for masking ground truth vertices')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    results_dir = output_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("deepRetinotopy Evaluation on NSD Dataset")
    print("="*80)
    print(f"Subject: {args.subject}")
    print(f"Hemisphere: {args.hemisphere}")
    print(f"Prediction: {args.prediction}")
    print(f"Model: {args.model_type}")
    print(f"Myelination: {args.myelination}")
    print(f"R2 threshold: {args.r2_threshold}")
    print("="*80)
    
    # Map prediction names
    pred_map = {
        'eccentricity': 'prfeccentricity',
        'polarAngle': 'prfangle',
        'pRFsize': 'prfsize'
    }
    
    gt_name = pred_map[args.prediction]
    
    # Paths to ground truth files
    gt_path = Path(args.nsd_dir) / args.subject / 'label' / f'{args.hemisphere}.{gt_name}.mgz'
    r2_path = Path(args.nsd_dir) / args.subject / 'label' / f'{args.hemisphere}.prfR2.mgz'
    roi_path = Path(args.nsd_dir) / args.subject / 'label' / f'{args.hemisphere}.prf-visualrois.mgz'
    
    # Check if ground truth files exist
    if not gt_path.exists():
        print(f"ERROR: Ground truth file not found: {gt_path}")
        return
    
    print(f"\nLoading ground truth data from: {gt_path}")
    gt_data = load_mgz_data(str(gt_path))
    print(f"Ground truth shape: {gt_data.shape}")
    print(f"Ground truth range: [{np.nanmin(gt_data):.3f}, {np.nanmax(gt_data):.3f}]")
    
    # Load R2 mask
    mask = None
    if r2_path.exists():
        print(f"\nLoading R2 data from: {r2_path}")
        r2_data = load_mgz_data(str(r2_path))
        mask = (r2_data >= args.r2_threshold) & np.isfinite(gt_data) & (gt_data != 0)
        print(f"R2 mask: {np.sum(mask)} / {len(mask)} vertices ({100*np.sum(mask)/len(mask):.1f}%)")
    else:
        print(f"Warning: R2 file not found: {r2_path}")
        mask = np.isfinite(gt_data) & (gt_data != 0)
        print(f"Using non-zero mask: {np.sum(mask)} / {len(mask)} vertices")
    
    # Load ROI mask if available
    if roi_path.exists():
        print(f"\nLoading ROI data from: {roi_path}")
        roi_data = load_mgz_data(str(roi_path))
        roi_mask = (roi_data > 0) & np.isfinite(roi_data)
        print(f"ROI mask: {np.sum(roi_mask)} / {len(roi_mask)} vertices")
        if mask is not None:
            mask = mask & roi_mask
            print(f"Combined mask: {np.sum(mask)} / {len(mask)} vertices")
    
    # Path to predicted data (will be generated by inference)
    # We need to construct the expected output path based on the model
    model_name = "model" if args.model_type == "baseline" else args.model_type
    myelin_suffix = "_myelin" if args.myelination == "True" else ""
    model_suffix = f"_{args.model_type}" if args.model_type != "baseline" else ""
    
    # Expected prediction file path (in native space after conversion)
    pred_path = Path(args.nsd_dir) / args.subject / 'deepRetinotopy' / \
                f'{args.subject}.predicted_{args.prediction}_{model_name}.{args.hemisphere}.native.func.gii'
    
    # Alternative path (fsaverage space)
    pred_path_fslr = Path(args.nsd_dir) / args.subject / 'deepRetinotopy' / \
                     f'{args.subject}.predicted_{args.prediction}_{args.hemisphere}{myelin_suffix}{model_suffix}.func.gii'
    
    # Check if prediction exists
    if not pred_path.exists() and not pred_path_fslr.exists():
        print(f"\nERROR: Prediction file not found!")
        print(f"Expected paths:")
        print(f"  Native space: {pred_path}")
        print(f"  FSLR space: {pred_path_fslr}")
        print(f"\nPlease run inference first using the run_nsd_inference.sh script")
        return
    
    # Load prediction data
    if pred_path.exists():
        print(f"\nLoading prediction data from: {pred_path}")
        pred_data = load_gii_data(str(pred_path))
    else:
        print(f"\nLoading prediction data from: {pred_path_fslr}")
        pred_data = load_gii_data(str(pred_path_fslr))
        print("Warning: Using FSLR space prediction. Results may not be directly comparable.")
    
    print(f"Prediction shape: {pred_data.shape}")
    print(f"Prediction range: [{np.nanmin(pred_data):.3f}, {np.nanmax(pred_data):.3f}]")
    
    # Check dimension compatibility
    if len(pred_data) != len(gt_data):
        print(f"\nERROR: Dimension mismatch!")
        print(f"Prediction: {len(pred_data)} vertices")
        print(f"Ground truth: {len(gt_data)} vertices")
        return
    
    # Compute metrics
    print("\n" + "="*80)
    print("Computing Evaluation Metrics")
    print("="*80)
    
    metrics = compute_metrics(pred_data, gt_data, mask)
    
    print(f"\nResults for {args.prediction} ({args.hemisphere}):")
    print(f"  Number of vertices: {metrics['n_vertices']}")
    print(f"  Correlation (r): {metrics['correlation']:.4f} (p={metrics['p_value']:.2e})")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"\n  Prediction - Mean: {metrics['pred_mean']:.4f}, Std: {metrics['pred_std']:.4f}")
    print(f"  Ground Truth - Mean: {metrics['gt_mean']:.4f}, Std: {metrics['gt_std']:.4f}")
    
    # Save metrics to JSON
    results_file = results_dir / f'{args.subject}_{args.hemisphere}_{args.prediction}_{args.model_type}_metrics.json'
    results_dict = {
        'subject': args.subject,
        'hemisphere': args.hemisphere,
        'prediction': args.prediction,
        'model_type': args.model_type,
        'myelination': args.myelination,
        'r2_threshold': args.r2_threshold,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved metrics to: {results_file}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)
    
    # Scatter plot
    scatter_path = plots_dir / f'{args.subject}_{args.hemisphere}_{args.prediction}_{args.model_type}_scatter.png'
    create_scatter_plot(pred_data, gt_data, mask, metrics, args.prediction, args.hemisphere, scatter_path)
    
    # Distribution plot
    dist_path = plots_dir / f'{args.subject}_{args.hemisphere}_{args.prediction}_{args.model_type}_distribution.png'
    create_distribution_plot(pred_data, gt_data, mask, args.prediction, args.hemisphere, dist_path)
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
