#!/usr/bin/env python3
"""
Summarize deepRetinotopy evaluation results on NSD dataset

This script reads all evaluation JSON files and creates:
1. Summary table with all metrics
2. Comparison plots across predictions and hemispheres
3. Text report
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("white")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16


def load_results(results_dir):
    """Load all JSON result files"""
    results_dir = Path(results_dir)
    results = []
    
    for json_file in results_dir.glob('*_metrics.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results


def create_summary_table(results):
    """Create summary DataFrame from results"""
    rows = []
    
    for result in results:
        row = {
            'Subject': result['subject'],
            'Hemisphere': result['hemisphere'].upper(),
            'Prediction': result['prediction'],
            'Model': result['model_type'],
            'N_vertices': result['metrics']['n_vertices'],
            'Correlation': result['metrics']['correlation'],
            'R²': result['metrics']['r2'],
            'MAE': result['metrics']['mae'],
            'RMSE': result['metrics']['rmse'],
            'Pred_Mean': result['metrics']['pred_mean'],
            'Pred_Std': result['metrics']['pred_std'],
            'GT_Mean': result['metrics']['gt_mean'],
            'GT_Std': result['metrics']['gt_std']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def create_comparison_plot(df, output_path):
    """Create comparison plot across predictions and hemispheres"""
    # Prepare data
    predictions = df['Prediction'].unique()
    hemispheres = df['Hemisphere'].unique()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('deepRetinotopy Performance on NSD Dataset', fontsize=18, y=0.995)
    
    # Metrics to plot
    metrics = ['Correlation', 'R²', 'MAE', 'RMSE']
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        # Prepare data for grouped bar plot
        x = np.arange(len(predictions))
        width = 0.35
        
        lh_values = []
        rh_values = []
        
        for pred in predictions:
            lh_val = df[(df['Prediction'] == pred) & (df['Hemisphere'] == 'LH')][metric].values
            rh_val = df[(df['Prediction'] == pred) & (df['Hemisphere'] == 'RH')][metric].values
            
            lh_values.append(lh_val[0] if len(lh_val) > 0 else 0)
            rh_values.append(rh_val[0] if len(rh_val) > 0 else 0)
        
        # Create bars
        bars1 = ax.bar(x - width/2, lh_values, width, label='LH', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, rh_values, width, label='RH', alpha=0.8, color='#e74c3c')
        
        # Customize
        ax.set_xlabel('Prediction Target', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('pRFsize', 'pRF Size').replace('polarAngle', 'Polar Angle').title() 
                            for p in predictions], rotation=15, ha='right')
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to: {output_path}")


def create_heatmap(df, output_path):
    """Create heatmap of correlations"""
    # Pivot table for correlation
    pivot = df.pivot_table(values='Correlation', 
                          index='Prediction', 
                          columns='Hemisphere')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, center=0.5,
                cbar_kws={'label': 'Correlation'},
                ax=ax, linewidths=2, linecolor='white')
    
    ax.set_title('Correlation Heatmap: Prediction vs Ground Truth', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Hemisphere', fontsize=14)
    ax.set_ylabel('Prediction Target', fontsize=14)
    
    # Format y-axis labels
    yticklabels = [label.get_text().replace('pRFsize', 'pRF Size').replace('polarAngle', 'Polar Angle').title() 
                   for label in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap to: {output_path}")


def generate_text_report(df, output_path):
    """Generate text report"""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("deepRetinotopy Evaluation Summary on NSD Dataset\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of evaluations: {len(df)}\n")
        f.write(f"Subjects: {', '.join(df['Subject'].unique())}\n")
        f.write(f"Model: {', '.join(df['Model'].unique())}\n")
        f.write(f"Predictions: {', '.join(df['Prediction'].unique())}\n")
        f.write(f"Hemispheres: {', '.join(df['Hemisphere'].unique())}\n")
        f.write("\n")
        
        # Average metrics
        f.write("Average Metrics Across All Evaluations:\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Correlation: {df['Correlation'].mean():.4f} ± {df['Correlation'].std():.4f}\n")
        f.write(f"Mean R²: {df['R²'].mean():.4f} ± {df['R²'].std():.4f}\n")
        f.write(f"Mean MAE: {df['MAE'].mean():.4f} ± {df['MAE'].std():.4f}\n")
        f.write(f"Mean RMSE: {df['RMSE'].mean():.4f} ± {df['RMSE'].std():.4f}\n")
        f.write("\n")
        
        # By prediction type
        f.write("Metrics by Prediction Type:\n")
        f.write("-"*80 + "\n")
        for pred in df['Prediction'].unique():
            pred_df = df[df['Prediction'] == pred]
            f.write(f"\n{pred}:\n")
            f.write(f"  Correlation: {pred_df['Correlation'].mean():.4f} ± {pred_df['Correlation'].std():.4f}\n")
            f.write(f"  R²: {pred_df['R²'].mean():.4f} ± {pred_df['R²'].std():.4f}\n")
            f.write(f"  MAE: {pred_df['MAE'].mean():.4f} ± {pred_df['MAE'].std():.4f}\n")
            f.write(f"  RMSE: {pred_df['RMSE'].mean():.4f} ± {pred_df['RMSE'].std():.4f}\n")
        f.write("\n")
        
        # By hemisphere
        f.write("Metrics by Hemisphere:\n")
        f.write("-"*80 + "\n")
        for hemi in df['Hemisphere'].unique():
            hemi_df = df[df['Hemisphere'] == hemi]
            f.write(f"\n{hemi}:\n")
            f.write(f"  Correlation: {hemi_df['Correlation'].mean():.4f} ± {hemi_df['Correlation'].std():.4f}\n")
            f.write(f"  R²: {hemi_df['R²'].mean():.4f} ± {hemi_df['R²'].std():.4f}\n")
            f.write(f"  MAE: {hemi_df['MAE'].mean():.4f} ± {hemi_df['MAE'].std():.4f}\n")
            f.write(f"  RMSE: {hemi_df['RMSE'].mean():.4f} ± {hemi_df['RMSE'].std():.4f}\n")
        f.write("\n")
        
        # Detailed results table
        f.write("Detailed Results:\n")
        f.write("="*80 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")
    
    print(f"Saved text report to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Summarize NSD evaluation results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing JSON result files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for summary')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Summarizing NSD Evaluation Results")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load results
    print("Loading results...")
    results = load_results(results_dir)
    
    if len(results) == 0:
        print("ERROR: No result files found!")
        return
    
    print(f"Found {len(results)} result files")
    
    # Create summary table
    print("\nCreating summary table...")
    df = create_summary_table(results)
    
    # Save CSV
    csv_path = output_dir / 'summary_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved summary table to: {csv_path}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Comparison plot
    comparison_path = output_dir / 'comparison_plot.png'
    create_comparison_plot(df, comparison_path)
    
    # Heatmap
    heatmap_path = output_dir / 'correlation_heatmap.png'
    create_heatmap(df, heatmap_path)
    
    # Generate text report
    print("\nGenerating text report...")
    report_path = output_dir / 'summary_report.txt'
    generate_text_report(df, report_path)
    
    print("\n" + "="*80)
    print("Summary Complete!")
    print("="*80)
    print(f"All outputs saved to: {output_dir}")
    print()
    
    # Print quick summary to console
    print("Quick Summary:")
    print("-"*80)
    print(f"Average Correlation: {df['Correlation'].mean():.4f}")
    print(f"Average R²: {df['R²'].mean():.4f}")
    print(f"Average MAE: {df['MAE'].mean():.4f}")
    print(f"Average RMSE: {df['RMSE'].mean():.4f}")
    print("-"*80)


if __name__ == '__main__':
    main()
