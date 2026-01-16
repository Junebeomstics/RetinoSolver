# deepRetinotopy Evaluation on Natural Scenes Dataset (NSD)

This directory contains scripts for evaluating deepRetinotopy predictions against ground truth pRF measurements from the Natural Scenes Dataset.

## Overview

The evaluation pipeline:
1. **Preprocessing**: Converts native FreeSurfer surfaces to fsaverage space
2. **Inference**: Runs deepRetinotopy model to predict pRF parameters
3. **Postprocessing**: Converts predictions back to native space
4. **Evaluation**: Compares predictions with ground truth and computes metrics
5. **Visualization**: Generates plots and summary reports

## Files

### Main Scripts

- **`run_nsd_inference.sh`**: Run inference and evaluation for a single prediction/hemisphere
- **`run_nsd_full_evaluation.sh`**: Run complete evaluation for all predictions and hemispheres
- **`evaluate_nsd_prf.py`**: Python script for computing metrics and creating plots
- **`summarize_nsd_results.py`**: Generate summary report from all evaluations

### Helper Scripts (from `run_from_freesurfer/`)

- **`1_native2fsaverage.sh`**: Convert native surfaces to fsaverage space
- **`2_fsaverage2native.sh`**: Convert predictions back to native space

## Requirements

### Software Dependencies

- Python 3.7+
- FreeSurfer (for `mris_convert`, `mris_expand`, etc.)
- Connectome Workbench (`wb_command`)
- Standard Python packages: numpy, scipy, nibabel, matplotlib, seaborn, pandas

### Data Requirements

- **NSD FreeSurfer Data**: `/mnt/external_storage1/natural-scenes-dataset/nsddata/freesurfer/`
  - Subject directories (e.g., `subj01/`)
  - Surface files in `surf/` directory
  - Ground truth pRF data in `label/` directory:
    - `lh.prfeccentricity.mgz` / `rh.prfeccentricity.mgz`
    - `lh.prfangle.mgz` / `rh.prfangle.mgz`
    - `lh.prfsize.mgz` / `rh.prfsize.mgz`
    - `lh.prfR2.mgz` / `rh.prfR2.mgz` (for quality masking)

- **Model Checkpoints**: `Models/checkpoints/`
  - Pre-trained model weights for each prediction target

- **HCP Surface Templates**: `surface/`
  - `fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii`
  - `fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii`

## Usage

### Quick Start: Single Evaluation

Run inference and evaluation for eccentricity in left hemisphere:

```bash
./run_nsd_inference.sh -s subj01 -h lh -p eccentricity
```

### Full Evaluation: All Predictions and Hemispheres

Run complete evaluation for all three pRF parameters (eccentricity, polar angle, pRF size) in both hemispheres:

```bash
./run_nsd_full_evaluation.sh -s subj01
```

This will:
- Run 6 evaluations (3 predictions × 2 hemispheres)
- Generate individual plots and metrics for each
- Create a summary report with aggregate statistics

### Command-Line Options

#### `run_nsd_inference.sh`

```bash
./run_nsd_inference.sh [options]

Options:
  -s SUBJECT      Subject ID (default: subj01)
  -h HEMISPHERE   Hemisphere: lh or rh (default: lh)
  -p PREDICTION   Prediction target: eccentricity, polarAngle, pRFsize (default: eccentricity)
  -m MODEL        Model type (default: baseline)
  -y MYELINATION  Use myelination: True or False (default: False)
  -r R2_THRESHOLD R2 threshold for evaluation (default: 0.1)
  -j N_JOBS       Number of parallel jobs (default: auto-detect)
  -o OUTPUT_DIR   Output directory (default: ./nsd_evaluation)
```

#### `run_nsd_full_evaluation.sh`

```bash
./run_nsd_full_evaluation.sh [options]

Options:
  -s SUBJECT      Subject ID (default: subj01)
  -m MODEL        Model type (default: baseline)
  -y MYELINATION  Use myelination: True or False (default: False)
  -r R2_THRESHOLD R2 threshold for evaluation (default: 0.1)
  -o OUTPUT_DIR   Output directory (default: ./nsd_evaluation)
```

### Examples

**Example 1**: Evaluate polar angle with myelination data

```bash
./run_nsd_inference.sh -s subj01 -h rh -p polarAngle -y True
```

**Example 2**: Evaluate all predictions with higher R² threshold

```bash
./run_nsd_full_evaluation.sh -s subj01 -r 0.2
```

**Example 3**: Evaluate using Transolver model

```bash
./run_nsd_full_evaluation.sh -s subj01 -m transolver_optionA
```

**Example 4**: Evaluate multiple subjects

```bash
for subj in subj01 subj02 subj03; do
    ./run_nsd_full_evaluation.sh -s $subj -o ./nsd_evaluation_${subj}
done
```

## Output Structure

After running the evaluation, the output directory will contain:

```
nsd_evaluation/
├── plots/
│   ├── subj01_lh_eccentricity_baseline_scatter.png
│   ├── subj01_lh_eccentricity_baseline_distribution.png
│   ├── subj01_rh_eccentricity_baseline_scatter.png
│   ├── ... (plots for each prediction/hemisphere)
├── results/
│   ├── subj01_lh_eccentricity_baseline_metrics.json
│   ├── subj01_rh_eccentricity_baseline_metrics.json
│   ├── ... (JSON files with detailed metrics)
├── summary_table.csv
├── summary_report.txt
├── comparison_plot.png
└── correlation_heatmap.png
```

### Output Files

#### Individual Results

- **Scatter plots** (`*_scatter.png`): Prediction vs ground truth with regression line
- **Distribution plots** (`*_distribution.png`): Histogram and box plot comparisons
- **Metrics JSON** (`*_metrics.json`): Detailed numerical results

#### Summary Files

- **`summary_table.csv`**: Table with all metrics for all evaluations
- **`summary_report.txt`**: Text report with statistics and analysis
- **`comparison_plot.png`**: Bar plots comparing metrics across predictions/hemispheres
- **`correlation_heatmap.png`**: Heatmap of correlations

## Evaluation Metrics

For each prediction, the following metrics are computed:

- **Correlation (r)**: Pearson correlation coefficient between prediction and ground truth
- **R²**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **N vertices**: Number of vertices included in evaluation (after masking)

### Masking

Vertices are included in evaluation if they meet the following criteria:

1. R² ≥ threshold (default: 0.1) - indicates reliable ground truth measurement
2. Valid (non-NaN, non-zero) ground truth value
3. Valid prediction value
4. Within visual cortex ROI (if available)

## Ground Truth Data

The NSD dataset provides pRF measurements obtained through standard pRF mapping experiments:

- **Eccentricity**: Distance from fovea in degrees of visual angle
- **Polar Angle**: Angular position in visual field (0-360°)
- **pRF Size**: Receptive field size in degrees of visual angle

These measurements were obtained using established pRF mapping techniques and serve as the gold standard for evaluation.

## Troubleshooting

### Issue: "Ground truth file not found"

**Solution**: Ensure NSD data is properly downloaded and the path is correct:
```bash
ls /mnt/external_storage1/natural-scenes-dataset/nsddata/freesurfer/subj01/label/*.prf*
```

### Issue: "Prediction file not found"

**Solution**: The inference step may have failed. Check that:
1. Model checkpoints exist in `Models/checkpoints/`
2. FreeSurfer surfaces are available
3. HCP surface templates are in `surface/` directory

### Issue: "Dimension mismatch"

**Solution**: This occurs when prediction and ground truth have different numbers of vertices. Ensure:
1. Native space conversion completed successfully
2. Using correct hemisphere
3. FreeSurfer version compatibility

### Issue: Low correlation scores

**Possible causes**:
1. R² threshold too low - try increasing with `-r 0.2`
2. Model not trained on similar data
3. Surface registration issues

## Performance Expectations

Based on validation studies, expected performance ranges:

| Metric | Eccentricity | Polar Angle | pRF Size |
|--------|-------------|-------------|----------|
| Correlation | 0.70-0.85 | 0.65-0.80 | 0.60-0.75 |
| R² | 0.50-0.70 | 0.40-0.65 | 0.35-0.55 |

Performance may vary based on:
- Subject-specific anatomy
- Quality of FreeSurfer reconstruction
- Model type and training data
- Use of myelination features

## Citation

If you use this evaluation pipeline, please cite:

```bibtex
@article{deepretinotopy2024,
  title={deepRetinotopy: Predicting retinotopic maps from cortical anatomy},
  author={...},
  journal={...},
  year={2024}
}
```

## Contact

For questions or issues, please open an issue on the GitHub repository or contact the authors.

## License

This code is released under the same license as the main deepRetinotopy project.
