#!/usr/bin/env python3
"""
Aggregate performance metrics across seeds for different experimental settings.

This script:
1. Finds all folders with 'seed' in their names in output_wandb directory
2. Loads *_best_test_results.pt files from each folder
3. Calculates performance metrics based on R2 thresholds and V1,V2,V3 masks
4. Groups results by (prediction_type, hemisphere, model_type)
5. Calculates mean and standard deviation across seeds
6. Creates a performance table

Usage:
    # Run with default settings (uses 'output_from_fs_curv' directory and 'aggregated_seed_results_from_fs_curv' as output name):
    python aggregate_seed_results.py
    
    # Specify custom output directory:
    python aggregate_seed_results.py --output_dir output_from_gifti
    
    # Specify custom output file name:
    python aggregate_seed_results.py --output_name aggregated_seed_results_custom
    
    # Specify both:
    python aggregate_seed_results.py --output_dir output_from_gifti --output_name aggregated_results_gifti
    
    # Run in Docker container with custom settings:
    docker exec deepretinotopy_train python /path/to/aggregate_seed_results.py --output_dir output_from_gifti --output_name aggregated_results_gifti
    
    # From project root in Docker:
    docker exec deepretinotopy_train bash -c "cd /path/to/deepRetinotopy && python aggregate_seed_results.py --output_dir output_from_gifti"

Arguments:
    --output_dir: Name of the directory containing the experiment results (default: 'output_from_fs_curv')
                  This directory should be inside the Models directory.
    --output_name: Base name for the output files (default: 'aggregated_seed_results_from_fs_curv')
                   Will create two files: {output_name}.csv and {output_name}.txt
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Try to import torch
try:
    import torch
except ImportError:
    print("Error: PyTorch (torch) is not installed.")
    print("Please install it using: pip install torch")
    print("Or run this script in an environment where torch is available (e.g., Docker container).")
    sys.exit(1)

# Import ROI and V1-V3 mask functions
try:
    from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
    from Retinotopy.functions.plusFovea import add_fovea, add_fovea_R
except ImportError as e:
    print(f"Error: Could not import ROI functions: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


def circular_correlation(x, y):
    """
    Calculate circular correlation coefficient for angular data.
    Supports both numpy.ndarray and torch.Tensor input.

    Parameters:
    -----------
    x, y : array-like or torch.Tensor
        Angular data in degrees (0-360)

    Returns:
    --------
    r : float
        Circular correlation coefficient
    """
    # Check if x and y are torch.Tensor or numpy arrays
    is_torch = isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor)

    # If inputs are numpy arrays or lists
    if not is_torch:
        x_rad = np.deg2rad(x)
        y_rad = np.deg2rad(y)
        x_complex = np.exp(1j * x_rad)
        y_complex = np.exp(1j * y_rad)
        x_complex_mean = np.mean(x_complex)
        y_complex_mean = np.mean(y_complex)
        x_complex_centered = x_complex - x_complex_mean
        y_complex_centered = y_complex - y_complex_mean
        numerator = np.real(np.mean(x_complex_centered * np.conj(y_complex_centered)))
        denominator = np.sqrt(np.mean(np.abs(x_complex_centered) ** 2) * np.mean(np.abs(y_complex_centered) ** 2))
        r_circular = numerator / denominator if denominator > 0 else 0.0
        return r_circular

    # If inputs are torch.Tensor
    # Ensure inputs are float, on CPU, and 1D (flatten if necessary)
    x = x.detach().cpu().float().flatten()
    y = y.detach().cpu().float().flatten()
    # Convert degrees to radians
    x_rad = x * (torch.pi / 180.0)
    y_rad = y * (torch.pi / 180.0)
    # Convert to complex numbers: e^{i*x_rad}
    x_complex = torch.cos(x_rad) + 1j * torch.sin(x_rad)
    y_complex = torch.cos(y_rad) + 1j * torch.sin(y_rad)
    # Calculate circular mean
    x_complex_mean = torch.mean(x_complex)
    y_complex_mean = torch.mean(y_complex)
    # Center
    x_complex_centered = x_complex - x_complex_mean
    y_complex_centered = y_complex - y_complex_mean
    # Numerator (real part of mean of product with conjugate)
    numerator = torch.real(torch.mean(x_complex_centered * torch.conj(y_complex_centered)))
    # Denominator
    denom_x = torch.mean(torch.abs(x_complex_centered) ** 2)
    denom_y = torch.mean(torch.abs(y_complex_centered) ** 2)
    denominator = torch.sqrt(denom_x * denom_y)
    r_circular = (numerator / denominator).item() if denominator > 0 else 0.0
    return r_circular


def load_V1V2V3_mask(hemisphere: str):
    """
    Load V1, V2, V3 mask for the specified hemisphere.
    
    Args:
        hemisphere: 'Left' or 'Right'
    
    Returns:
        v1v2v3_mask_roi: Boolean mask for V1-V3 areas within ROI (length matches ROI vertices)
    """
    primary_visual_areas = ['V1d', 'V1v', 'fovea_V1', 'V2d', 'V2v', 'fovea_V2', 'V3d', 'V3v', 'fovea_V3']
    
    if hemisphere.lower() == 'left':
        V1, V2, V3 = add_fovea(primary_visual_areas)
    elif hemisphere.lower() == 'right':
        V1, V2, V3 = add_fovea_R(primary_visual_areas)
    else:
        raise ValueError("Hemisphere must be 'Left' or 'Right'.")
    
    V1V2V3_mask_full = ((V1 > 0) | (V2 > 0) | (V3 > 0)).astype(bool)
    
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(label_primary_visual_areas)
    
    if hemisphere.lower() == 'left':
        hemisphere_mask = final_mask_L
    else:
        hemisphere_mask = final_mask_R
    
    # Only keep voxels/vertices that are both in the area mask and ROI mask
    V1V2V3_mask_roi = V1V2V3_mask_full[hemisphere_mask == 1]
    
    return V1V2V3_mask_roi


def parse_folder_name(folder_name: str) -> Optional[Dict[str, str]]:
    """
    Parse folder name to extract experimental settings.
    
    Expected format: {prediction_type}_{hemisphere}_{model_type}_noMyelin_seed{seed}
    Examples:
    - eccentricity_Left_baseline_noMyelin_seed0
    - polarAngle_Right_transolver_optionA_noMyelin_seed1
    - pRFsize_Left_transolver_optionC_noMyelin_seed2
    
    Returns:
        Dictionary with keys: prediction_type, hemisphere, model_type, seed
        Returns None if parsing fails
    """
    # Pattern: {prediction_type}_{hemisphere}_{model_type}_noMyelin_seed{seed}
    pattern = r'^(eccentricity|polarAngle|pRFsize)_(Left|Right)_(baseline|transolver_optionA|transolver_optionC)_noMyelin_seed(\d+)$'
    match = re.match(pattern, folder_name)
    
    if match:
        return {
            'prediction_type': match.group(1),
            'hemisphere': match.group(2),
            'model_type': match.group(3),
            'seed': match.group(4)
        }
    return None


def load_results_file(file_path: Path, prediction_type: str, hemisphere: str) -> Optional[Dict[str, float]]:
    """
    Load results from a *_best_test_results.pt file and calculate performance metrics.
    
    Args:
        file_path: Path to the results file
        prediction_type: Type of prediction ('polarAngle', 'eccentricity', 'pRFsize')
        hemisphere: Hemisphere ('Left' or 'Right')
    
    Returns:
        Dictionary with calculated metrics, or None if loading fails
    """
    try:
        data = torch.load(file_path, map_location='cpu')
        
        # Extract data
        Predicted_values = data.get('Predicted_values', None)
        Measured_values = data.get('Measured_values', None)
        R2 = data.get('R2', None)
        
        if Predicted_values is None or Measured_values is None or R2 is None:
            print(f"Warning: Missing required data in {file_path}")
            return None
        
        # Load V1-V3 mask
        V1V2V3_mask_roi = load_V1V2V3_mask(hemisphere)
        
        # Store metrics per subject
        metrics_per_subject = {
            'r2_gt2.2': [],
            'r2_gt10': [],
            'r2_gt17': [],
            'v1v2v3_corr': [],
            'v1v2v3_r2_gt10_corr': []
        }
        
        # Calculate metrics for each subject
        for pred, meas, r2 in zip(Predicted_values, Measured_values, R2):
            # Convert to numpy if torch tensors
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            if isinstance(meas, torch.Tensor):
                meas = meas.detach().cpu().numpy()
            if isinstance(r2, torch.Tensor):
                r2 = r2.detach().cpu().numpy()
            
            # Ensure 1D arrays
            pred = pred.flatten()
            meas = meas.flatten()
            r2 = r2.flatten()
            
            # R2 > 2.2
            mask_2_2 = r2 > 2.2
            if mask_2_2.sum() > 1:
                if prediction_type == 'polarAngle':
                    r2_gt2_2_corr = circular_correlation(pred[mask_2_2], meas[mask_2_2])
                else:
                    r2_gt2_2_corr = np.corrcoef(pred[mask_2_2], meas[mask_2_2])[0, 1]
                metrics_per_subject['r2_gt2.2'].append(r2_gt2_2_corr)
            else:
                metrics_per_subject['r2_gt2.2'].append(np.nan)
            
            # R2 > 10
            mask_10 = r2 > 10
            if mask_10.sum() > 1:
                if prediction_type == 'polarAngle':
                    r2_gt10_corr = circular_correlation(pred[mask_10], meas[mask_10])
                else:
                    r2_gt10_corr = np.corrcoef(pred[mask_10], meas[mask_10])[0, 1]
                metrics_per_subject['r2_gt10'].append(r2_gt10_corr)
            else:
                metrics_per_subject['r2_gt10'].append(np.nan)
            
            # R2 > 17
            mask_17 = r2 > 17
            if mask_17.sum() > 1:
                if prediction_type == 'polarAngle':
                    r2_gt17_corr = circular_correlation(pred[mask_17], meas[mask_17])
                else:
                    r2_gt17_corr = np.corrcoef(pred[mask_17], meas[mask_17])[0, 1]
                metrics_per_subject['r2_gt17'].append(r2_gt17_corr)
            else:
                metrics_per_subject['r2_gt17'].append(np.nan)
            
            # V1,V2,V3 only within ROI for this hemisphere
            if V1V2V3_mask_roi.sum() > 1:
                if prediction_type == 'polarAngle':
                    v1v2v3_corr = circular_correlation(pred[V1V2V3_mask_roi], meas[V1V2V3_mask_roi])
                else:
                    v1v2v3_corr = np.corrcoef(pred[V1V2V3_mask_roi], meas[V1V2V3_mask_roi])[0, 1]
                metrics_per_subject['v1v2v3_corr'].append(v1v2v3_corr)
            else:
                metrics_per_subject['v1v2v3_corr'].append(np.nan)
            
            # R2 > 10 AND in V1V2V3 + ROI for this hemisphere (final correlation)
            v1v2v3_and_r2_mask = (r2 > 10) & V1V2V3_mask_roi
            if v1v2v3_and_r2_mask.sum() > 1:
                if prediction_type == 'polarAngle':
                    correlation = circular_correlation(pred[v1v2v3_and_r2_mask], meas[v1v2v3_and_r2_mask])
                else:
                    correlation = np.corrcoef(pred[v1v2v3_and_r2_mask], meas[v1v2v3_and_r2_mask])[0, 1]
                metrics_per_subject['v1v2v3_r2_gt10_corr'].append(correlation)
            else:
                metrics_per_subject['v1v2v3_r2_gt10_corr'].append(np.nan)
        
        # Calculate mean across subjects for each metric
        result = {}
        for metric_name, values in metrics_per_subject.items():
            values_array = np.array(values)
            valid_values = values_array[~np.isnan(values_array)]
            if valid_values.size > 0:
                result[metric_name] = float(np.mean(valid_values))
            else:
                result[metric_name] = np.nan
        
        return result
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_result_files(base_dir: Path) -> Dict[Tuple[str, str, str], List[Dict[str, float]]]:
    """
    Find and load all result files, grouped by experimental setting.
    
    Returns:
        Dictionary mapping (prediction_type, hemisphere, model_type) to list of results
    """
    results = defaultdict(list)
    
    # Find all directories with 'seed' in name
    seed_dirs = [d for d in base_dir.iterdir() 
                 if d.is_dir() and 'seed' in d.name]
    
    print(f"Found {len(seed_dirs)} directories with 'seed' in name")
    
    for seed_dir in seed_dirs:
        # Parse folder name
        settings = parse_folder_name(seed_dir.name)
        if settings is None:
            print(f"Warning: Could not parse folder name: {seed_dir.name}")
            continue
        
        # Find result file
        result_files = list(seed_dir.glob('*_best_test_results.pt'))
        
        if len(result_files) == 0:
            print(f"Warning: No result file found in {seed_dir.name}")
            continue
        
        if len(result_files) > 1:
            print(f"Warning: Multiple result files found in {seed_dir.name}, using first")
        
        result_file = result_files[0]
        
        # Load results
        metrics = load_results_file(result_file, settings['prediction_type'], settings['hemisphere'])
        if metrics is None:
            continue
        
        # Group by (prediction_type, hemisphere, model_type)
        key = (settings['prediction_type'], settings['hemisphere'], settings['model_type'])
        results[key].append({
            'seed': settings['seed'],
            **metrics
        })
    
    return results


def aggregate_results(results: Dict[Tuple[str, str, str], List[Dict[str, float]]]) -> pd.DataFrame:
    """
    Aggregate results across seeds and create a summary table.
    
    Returns:
        DataFrame with columns: Prediction, Hemisphere, Model, and various correlation metrics
    """
    rows = []
    
    for (prediction_type, hemisphere, model_type), seed_results in results.items():
        if len(seed_results) == 0:
            continue
        
        # Extract metrics for all seeds
        metric_names = ['r2_gt2.2', 'r2_gt10', 'r2_gt17', 'v1v2v3_corr', 'v1v2v3_r2_gt10_corr']
        
        row = {
            'Prediction': prediction_type,
            'Hemisphere': hemisphere,
            'Model': model_type,
            'N_seeds': len(seed_results),
            'Seeds': ', '.join(sorted([r['seed'] for r in seed_results]))
        }
        
        # Calculate mean and std for each metric across seeds
        for metric_name in metric_names:
            values = [r.get(metric_name, np.nan) for r in seed_results]
            values_array = np.array(values)
            valid_values = values_array[~np.isnan(values_array)]
            
            if valid_values.size > 0:
                row[f'{metric_name}_mean'] = float(np.mean(valid_values))
                row[f'{metric_name}_std'] = float(np.std(valid_values, ddof=1)) if valid_values.size > 1 else 0.0
            else:
                row[f'{metric_name}_mean'] = np.nan
                row[f'{metric_name}_std'] = np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by Prediction, Hemisphere, Model
    df = df.sort_values(['Prediction', 'Hemisphere', 'Model']).reset_index(drop=True)
    
    return df


def format_table(df: pd.DataFrame) -> str:
    """
    Format the DataFrame as a nicely formatted table string.
    """
    # Create formatted strings
    formatted_rows = []
    
    for _, row in df.iterrows():
        formatted_row = {
            'Prediction': row['Prediction'],
            'Hemisphere': row['Hemisphere'],
            'Model': row['Model'],
            'N_seeds': row['N_seeds']
        }
        
        # Format each metric
        metric_names = ['r2_gt2.2', 'r2_gt10', 'r2_gt17', 'v1v2v3_corr', 'v1v2v3_r2_gt10_corr']
        for metric_name in metric_names:
            mean_col = f'{metric_name}_mean'
            std_col = f'{metric_name}_std'
            if pd.notna(row[mean_col]):
                formatted_row[metric_name] = f"{row[mean_col]:.4f} ± {row[std_col]:.4f}"
            else:
                formatted_row[metric_name] = "N/A"
        
        formatted_rows.append(formatted_row)
    
    formatted_df = pd.DataFrame(formatted_rows)
    
    # Convert to string with nice formatting
    return formatted_df.to_string(index=False)


def main():
    """Main function to aggregate and display results."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Aggregate performance metrics across seeds for different experimental settings.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output_from_fs_curv',
        help='Name of the directory containing the experiment results (default: output_from_fs_curv). '
             'This directory should be inside the Models directory.'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default='aggregated_seed_results_from_fs_curv',
        help='Base name for the output files (default: aggregated_seed_results_from_fs_curv). '
             'Will create two files: {output_name}.csv and {output_name}.txt'
    )
    
    args = parser.parse_args()
    
    # Set base directory using the argument
    base_dir = Path(__file__).parent / 'Models' / args.output_dir
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return
    
    print(f"Searching for results in: {base_dir}")
    print("-" * 80)
    
    # Find and load all result files
    results = find_result_files(base_dir)
    
    if len(results) == 0:
        print("No results found!")
        return
    
    print(f"\nFound results for {len(results)} experimental settings")
    print("-" * 80)
    
    # Aggregate results
    df = aggregate_results(results)
    
    # Display results
    print("\nAggregated Performance Results:")
    print("=" * 120)
    print("Metrics:")
    print("  - r2_gt2.2: Correlation for vertices with R2 > 2.2")
    print("  - r2_gt10: Correlation for vertices with R2 > 10")
    print("  - r2_gt17: Correlation for vertices with R2 > 17")
    print("  - v1v2v3_corr: Correlation for vertices in V1-V3 ROI")
    print("  - v1v2v3_r2_gt10_corr: Final correlation (R2 > 10 AND in V1-V3 ROI)")
    print("=" * 120)
    print(format_table(df))
    print("=" * 120)
    
    # Save to CSV using the output_name argument
    output_csv = Path(__file__).parent / f'{args.output_name}.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Also save formatted version using the output_name argument
    output_txt = Path(__file__).parent / f'{args.output_name}.txt'
    with open(output_txt, 'w') as f:
        f.write("Aggregated Performance Results Across Seeds\n")
        f.write("=" * 120 + "\n\n")
        f.write("Metrics:\n")
        f.write("  - r2_gt2.2: Correlation for vertices with R2 > 2.2\n")
        f.write("  - r2_gt10: Correlation for vertices with R2 > 10\n")
        f.write("  - r2_gt17: Correlation for vertices with R2 > 17\n")
        f.write("  - v1v2v3_corr: Correlation for vertices in V1-V3 ROI\n")
        f.write("  - v1v2v3_r2_gt10_corr: Final correlation (R2 > 10 AND in V1-V3 ROI)\n")
        f.write("=" * 120 + "\n\n")
        f.write(format_table(df))
        f.write("\n\n")
        f.write("=" * 120 + "\n")
        f.write(f"Generated from: {base_dir}\n")
        f.write("Note: polarAngle uses circular correlation, others use Pearson correlation\n")
    print(f"Formatted results saved to: {output_txt}")


if __name__ == '__main__':
    main()
