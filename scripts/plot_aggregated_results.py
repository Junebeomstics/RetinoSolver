#!/usr/bin/env python3
"""
Visualize aggregated seed results comparing models across predictions.

This script creates:
- Three plots for MAE_thr (one per prediction type)
- Three plots for Correlation (one per prediction type)
- Each plot compares models (baseline, transolver_optionA, transolver_optionC) across hemispheres
- Error bars show 95% confidence intervals
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Using approximate t-value for 95% CI.")

# Set style according to visualization rules
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Color palette (colorblind-friendly)
COLORS = {
    'baseline': '#1f77b4',  # Blue
    'transolver_optionA': '#ff7f0e',  # Orange
    'transolver_optionC': '#2ca02c',  # Green
}

# Model display names
MODEL_NAMES = {
    'baseline': 'deepRetinotopy',
    'transolver_optionA': 'Retinosolver',
    'transolver_optionC': 'Transolver',
}

# Model order for plotting: deepRetinotopy, Transolver, Retinosolver
MODEL_ORDER = ['baseline', 'transolver_optionC', 'transolver_optionA']

def calculate_95ci(mean, std, n):
    """
    Calculate 95% confidence interval using t-distribution.
    
    Args:
        mean: Mean value
        std: Standard deviation
        n: Sample size
    
    Returns:
        (lower_bound, upper_bound) tuple
    """
    if n < 2:
        return (mean, mean)
    # t-value for 95% CI with n-1 degrees of freedom
    # For n=3, df=2, t(0.975, df=2) ≈ 2.776
    if SCIPY_AVAILABLE:
        t_value = stats.t.ppf(0.975, df=n-1)
    else:
        # Approximate t-values for common sample sizes
        t_values = {2: 4.303, 3: 2.776, 4: 2.571, 5: 2.447, 10: 2.228, 30: 2.045}
        t_value = t_values.get(n-1, 2.0)  # Default to 2.0 if not in table
    margin = t_value * std / np.sqrt(n)
    return (mean - margin, mean + margin)

def create_comparison_plot(data_dict, metric, metric_mean_col, metric_std_col, title_suffix, output_dir):
    """
    Create comparison plots for a given metric.
    
    Args:
        data_dict: Dictionary with data organized by prediction -> model -> hemisphere
        metric: Metric name ('MAE_thr' or 'Correlation')
        metric_mean_col: Column name for mean values
        metric_std_col: Column name for std values
        title_suffix: Suffix for plot title
        output_dir: Directory to save plots
    """
    predictions = list(data_dict.keys())
    
    for prediction in predictions:
        pred_data = data_dict[prediction]
        
        # Prepare data for plotting
        # Order: deepRetinotopy (baseline), Transolver (optionC), Retinosolver (optionA)
        models = MODEL_ORDER
        hemispheres = ['Left', 'Right']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(hemispheres))
        width = 0.25  # Width of bars
        
        for i, model in enumerate(models):
            means = []
            ci_lowers = []
            ci_uppers = []
            
            for hemisphere in hemispheres:
                key = (model, hemisphere)
                if key in pred_data:
                    mean_val = float(pred_data[key][metric_mean_col])
                    std_val = float(pred_data[key][metric_std_col])
                    n_seeds = int(pred_data[key]['N_seeds'])
                    
                    ci_lower, ci_upper = calculate_95ci(mean_val, std_val, n_seeds)
                    means.append(mean_val)
                    ci_lowers.append(ci_lower)
                    ci_uppers.append(ci_upper)
                else:
                    means.append(0)
                    ci_lowers.append(0)
                    ci_uppers.append(0)
            
            # Calculate error bar positions (distance from mean)
            yerr_lower = [mean - lower for mean, lower in zip(means, ci_lowers)]
            yerr_upper = [upper - mean for mean, upper in zip(means, ci_uppers)]
            yerr = [yerr_lower, yerr_upper]
            
            # Plot bars with model display name
            offset = (i - 1) * width
            bars = ax.bar(x_pos + offset, means, width, 
                         label=MODEL_NAMES[model],
                         color=COLORS[model],
                         yerr=yerr,
                         capsize=5,
                         alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Hemisphere', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=14, fontweight='bold')
        ax.set_title(f'{prediction} - {title_suffix}', fontsize=16, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(hemispheres)
        ax.legend(loc='best', frameon=True)
        
        # Apply visualization rules
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(False)
        
        # Ensure axis lines don't extend beyond tick marks
        yticks = ax.get_yticks()
        if len(yticks) > 0:
            ax.spines['left'].set_bounds(yticks[0], yticks[-1])
            # Limit y-axis to show only data within tick range
            ax.set_ylim(yticks[0], yticks[-1])
        xticks = ax.get_xticks()
        # if len(xticks) > 0:
        #     ax.spines['bottom'].set_bounds(xticks[0], xticks[-1])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_file = output_dir / f'{prediction}_{metric}_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def main():
    """Main function to create visualization plots."""
    # Load data
    script_dir = Path(__file__).parent
    csv_file = script_dir / 'aggregated_seed_results.csv'
    
    if not csv_file.exists():
        print(f"Error: File not found: {csv_file}")
        return
    
    # Parse CSV file
    data_dict = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prediction = row['Prediction']
            model = row['Model']
            hemisphere = row['Hemisphere']
            
            if prediction not in data_dict:
                data_dict[prediction] = {}
            
            key = (model, hemisphere)
            data_dict[prediction][key] = row
    
    print(f"Loaded data from {csv_file}")
    print(f"Predictions: {list(data_dict.keys())}")
    
    # Create output directory
    output_dir = script_dir / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    # Create MAE_thr plots
    print("\nCreating MAE_thr comparison plots...")
    create_comparison_plot(data_dict, 'MAE_thr', 'MAE_thr_mean', 'MAE_thr_std', 
                          'MAE_thr Comparison', output_dir)
    
    # Create Correlation plots
    print("\nCreating Correlation comparison plots...")
    create_comparison_plot(data_dict, 'Correlation', 'Correlation_mean', 'Correlation_std',
                          'Correlation Comparison', output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == '__main__':
    main()
