# Inference Scripts Documentation

This directory contains scripts for running inference with pre-trained deepRetinotopy models. There are two main scripts:

1. **`run_inference_from_fslr_curv_using_checkpoint.py`** - Single inference script for one subject and one checkpoint
2. **`run_batch_inference_from_checkpoints.py`** - Batch inference script for multiple subjects and multiple checkpoints

## Table of Contents

- [Single Inference Script](#single-inference-script)
- [Batch Inference Script](#batch-inference-script)
- [Checkpoint Path Format](#checkpoint-path-format)
- [Output Format](#output-format)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Single Inference Script

**Script:** `run_inference_from_fslr_curv_using_checkpoint.py`  
**Location:** Project root directory  
**Purpose:** Run inference for a single subject using a single checkpoint

### Features

- Automatically extracts model information from checkpoint path (prediction type, hemisphere, model type, myelination, seed)
- Supports both subject index and subject ID
- Handles all model types: baseline, transolver_optionA, transolver_optionB, transolver_optionC
- Supports custom hyperparameters for transolver_optionC

### Required Arguments

- `--checkpoint_path`: Path to the pre-trained model checkpoint (.pt file)
- Either `--subject_index` OR `--subject_id` (mutually exclusive)

### Optional Arguments

- `--data_dir`: Path to `Retinotopy/data/raw/converted` directory (default: `Retinotopy/data/raw/converted`)
- `--output_dir`: Output directory for predictions (default: `./fslr_inference_output/seed{seed}/`)
- `--n_layers`: Number of Transolver blocks for optionC (default: 8)
- `--n_hidden`: Hidden dimension for optionC (default: 128)
- `--n_heads`: Number of attention heads for optionC (default: 8)
- `--slice_num`: Number of slice tokens in Physics Attention for optionC (default: 64)
- `--mlp_ratio`: MLP ratio in Transolver blocks for optionC (default: 1)
- `--dropout`: Dropout rate for optionC (default: 0.0)
- `--ref`: Reference grid size for unified_pos for optionC (default: 8)
- `--unified_pos`: Use unified position encoding for optionC (0=False, 1=True, default: 0)

### Usage Examples

#### Using Subject ID (Recommended)

```bash
python run_inference_from_fslr_curv_using_checkpoint.py \
    --subject_id 157336 \
    --checkpoint_path Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed0/ecc_Left_baseline_noMyelin_seed0_best_model_epoch66.pt
```

#### Using Subject Index

```bash
python run_inference_from_fslr_curv_using_checkpoint.py \
    --subject_index 0 \
    --checkpoint_path Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed0/ecc_Left_baseline_noMyelin_seed0_best_model_epoch66.pt
```

#### Custom Output Directory

```bash
python run_inference_from_fslr_curv_using_checkpoint.py \
    --subject_id 157336 \
    --checkpoint_path Models/output_wandb/polarAngle_Right_transolver_optionC_noMyelin_seed1/PA_Right_transolver_optionC_noMyelin_seed1_best_model_epoch62.pt \
    --output_dir /path/to/custom/output
```

#### With Custom Hyperparameters (for transolver_optionC)

```bash
python run_inference_from_fslr_curv_using_checkpoint.py \
    --subject_id 157336 \
    --checkpoint_path Models/output_wandb/eccentricity_Left_transolver_optionC_noMyelin_seed0/ecc_Left_transolver_optionC_noMyelin_seed0_best_model_epoch50.pt \
    --n_layers 8 \
    --n_hidden 128 \
    --n_heads 8 \
    --slice_num 64
```

#### Docker Usage

```bash
docker exec -it <CONTAINER_ID> python run_inference_from_fslr_curv_using_checkpoint.py \
    --subject_id 157336 \
    --checkpoint_path Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed1/ecc_Left_baseline_noMyelin_seed1_best_model_epoch66.pt
```

**Note:** 
- Use subject_id without 'sub-' prefix (e.g., `157336` instead of `sub-157336`)
- Checkpoint path is relative to workspace directory (`/workspace`)

---

## Batch Inference Script

**Script:** `run_batch_inference_from_checkpoints.py`  
**Location:** `scripts/` directory  
**Purpose:** Run inference for multiple subjects with multiple checkpoints

### Features

- Processes all combinations of subjects × checkpoints
- Automatically finds `best_model*.pt` files in checkpoint directories
- Supports parallel processing for faster execution
- Provides summary of successful/failed tasks
- Uses `run_inference_from_fslr_curv_using_checkpoint.py` internally

### Required Arguments

- `--subject_list`: Path to text file containing subject IDs (one per line)
- `--checkpoint_dirs`: One or more checkpoint directories (will use `best_model*.pt` from each)

### Optional Arguments

- `--output_base_dir`: Base directory for all inference outputs (default: `inference_output_batch`)
- `--data_dir`: Path to data directory (default: `Retinotopy/data/raw/converted`)
- `--inference_script`: Path to inference script (default: `run_inference_from_fslr_curv_using_checkpoint.py`)
- `--parallel`: Enable parallel processing (flag)
- `--num_workers`: Number of parallel workers (default: 4, only used with `--parallel`)

### Usage Examples

#### Sequential Execution (Single Checkpoint)

```bash
python scripts/run_batch_inference_from_checkpoints.py \
    --subject_list Retinotopy/data/subject_splits/seed0/test_subjects.txt \
    --checkpoint_dirs Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed0
```

#### Sequential Execution (Multiple Checkpoints)

```bash
python scripts/run_batch_inference_from_checkpoints.py \
    --subject_list Retinotopy/data/subject_splits/seed0/test_subjects.txt \
    --checkpoint_dirs Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed0 \
                     Models/output_wandb/eccentricity_Left_transolver_optionA_noMyelin_seed0 \
                     Models/output_wandb/eccentricity_Left_transolver_optionB_noMyelin_seed0
```

#### Parallel Execution

```bash
python scripts/run_batch_inference_from_checkpoints.py \
    --subject_list Retinotopy/data/subject_splits/seed0/test_subjects.txt \
    --checkpoint_dirs Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed0 \
                     Models/output_wandb/eccentricity_Left_transolver_optionA_noMyelin_seed0 \
    --parallel \
    --num_workers 8
```

#### Custom Output Directory

```bash
python scripts/run_batch_inference_from_checkpoints.py \
    --subject_list Retinotopy/data/subject_splits/seed0/test_subjects.txt \
    --checkpoint_dirs Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed0 \
    --output_base_dir /path/to/custom/output
```

### Subject List File Format

The subject list file should contain one subject ID per line (without 'sub-' prefix):

```
157336
191033
192540
195041
...
```

---

## Checkpoint Path Format

The scripts automatically extract model information from checkpoint filenames. Expected format:

```
{prediction_short}_{hemisphere}_{model_type}{myelination_suffix}{seed_suffix}_best_model_epoch{epoch}.pt
```

### Examples

- `ecc_Left_baseline_noMyelin_seed0_best_model_epoch66.pt`
- `PA_Right_transolver_optionA_noMyelin_best_model_epoch50.pt`
- `size_Left_transolver_optionC_noMyelin_seed1_best_model_epoch62.pt`

### Prediction Type Mapping

- `ecc` → `eccentricity`
- `PA` → `polarAngle`
- `size` → `pRFsize`

### Hemisphere Mapping

- `Left`, `LH`, `lh` → `Left`
- `Right`, `RH`, `rh` → `Right`

### Model Type Mapping

- `baseline` → `baseline`
- `transolver_optionA`, `optionA` → `transolver_optionA`
- `transolver_optionB`, `optionB` → `transolver_optionB`
- `transolver_optionC`, `optionC` → `transolver_optionC`

### Myelination Detection

- If filename contains `noMyelin` or `no_myelin` → `myelination=False`
- If filename contains `myelin` (case insensitive) → `myelination=True`
- Default: `myelination=False` if not specified

### Seed Extraction

- Pattern: `_seed{number}_` or `_seed{number}` before `_best_model`
- Default: `seed=0` if not found

---

## Output Format

### Single Inference Output

**Location:** `{output_dir}/seed{seed}/{subject_id}_{prediction_short}_{hemisphere}_{model_type}{myelination_suffix}_prediction.pt`

**Contents:**
- `subject_id`: Subject identifier
- `subject_index`: Subject index (0-based)
- `prediction_type`: Type of prediction (eccentricity, polarAngle, pRFsize)
- `hemisphere`: Hemisphere (Left or Right)
- `model_type`: Model architecture type
- `myelination`: Whether myelination was used
- `seed`: Random seed used
- `checkpoint_path`: Path to the checkpoint used
- `predicted_values`: NumPy array of predicted values
- `visual_mask`: Visual cortex mask
- `num_vertices`: Number of vertices predicted

**Example:**
```
fslr_inference_output/
└── seed0/
    └── 157336_ecc_Left_baseline_noMyelin_prediction.pt
```

### Batch Inference Output

**Location:** `{output_base_dir}/{checkpoint_dir_name}/seed{seed}/{subject_id}_{prediction_short}_{hemisphere}_{model_type}{myelination_suffix}_prediction.pt`

**Structure:**
```
inference_output_batch/
├── eccentricity_Left_baseline_noMyelin_seed0/
│   └── seed0/
│       ├── 157336_ecc_Left_baseline_noMyelin_prediction.pt
│       ├── 191033_ecc_Left_baseline_noMyelin_prediction.pt
│       └── ...
└── eccentricity_Left_transolver_optionA_noMyelin_seed0/
    └── seed0/
        ├── 157336_ecc_Left_transolver_optionA_noMyelin_prediction.pt
        ├── 191033_ecc_Left_transolver_optionA_noMyelin_prediction.pt
        └── ...
```

---

## Examples

### Example 1: Single Subject Evaluation

Evaluate one subject with multiple models:

```bash
# Subject: 157336
# Models: baseline, optionA, optionB, optionC

python run_inference_from_fslr_curv_using_checkpoint.py \
    --subject_id 157336 \
    --checkpoint_path Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed0/ecc_Left_baseline_noMyelin_seed0_best_model_epoch66.pt

python run_inference_from_fslr_curv_using_checkpoint.py \
    --subject_id 157336 \
    --checkpoint_path Models/output_wandb/eccentricity_Left_transolver_optionA_noMyelin_seed0/ecc_Left_transolver_optionA_noMyelin_seed0_best_model_epoch50.pt

python run_inference_from_fslr_curv_using_checkpoint.py \
    --subject_id 157336 \
    --checkpoint_path Models/output_wandb/eccentricity_Left_transolver_optionB_noMyelin_seed0/ecc_Left_transolver_optionB_noMyelin_seed0_best_model_epoch55.pt

python run_inference_from_fslr_curv_using_checkpoint.py \
    --subject_id 157336 \
    --checkpoint_path Models/output_wandb/eccentricity_Left_transolver_optionC_noMyelin_seed0/ecc_Left_transolver_optionC_noMyelin_seed0_best_model_epoch62.pt
```

### Example 2: Test Set Evaluation

Evaluate entire test set with one model:

```bash
python scripts/run_batch_inference_from_checkpoints.py \
    --subject_list Retinotopy/data/subject_splits/seed0/test_subjects.txt \
    --checkpoint_dirs Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed0 \
    --parallel \
    --num_workers 4
```

### Example 3: Model Comparison

Compare multiple models on test set:

```bash
python scripts/run_batch_inference_from_checkpoints.py \
    --subject_list Retinotopy/data/subject_splits/seed0/test_subjects.txt \
    --checkpoint_dirs Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed0 \
                     Models/output_wandb/eccentricity_Left_transolver_optionA_noMyelin_seed0 \
                     Models/output_wandb/eccentricity_Left_transolver_optionB_noMyelin_seed0 \
                     Models/output_wandb/eccentricity_Left_transolver_optionC_noMyelin_seed0 \
    --parallel \
    --num_workers 8 \
    --output_base_dir model_comparison_results
```

### Example 4: Multiple Seeds Evaluation

Evaluate models trained with different seeds:

```bash
python scripts/run_batch_inference_from_checkpoints.py \
    --subject_list Retinotopy/data/subject_splits/seed0/test_subjects.txt \
    --checkpoint_dirs Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed0 \
                     Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed1 \
                     Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed2 \
    --parallel \
    --num_workers 6
```

---

## Troubleshooting

### Common Issues

#### 1. Checkpoint Path Not Found

**Error:** `FileNotFoundError: Checkpoint file not found`

**Solution:** 
- Check that the checkpoint path is correct
- Use absolute path or path relative to current working directory
- Ensure checkpoint file exists and has `.pt` extension

#### 2. Subject ID Not Found

**Error:** `ValueError: Subject ID 'XXX' not found in list_subj`

**Solution:**
- Verify subject ID exists in `Retinotopy/data/list_subj` file
- Use subject ID without 'sub-' prefix (e.g., `157336` not `sub-157336`)
- Check spelling and case sensitivity

#### 3. Checkpoint Path Parsing Error

**Error:** `ValueError: Could not extract prediction type from filename`

**Solution:**
- Ensure checkpoint filename follows expected format
- Check that filename contains prediction type (`ecc`, `PA`, or `size`)
- Verify hemisphere (`Left`/`Right` or `LH`/`RH`) is present
- Ensure model type is correctly specified

#### 4. Data Directory Not Found

**Error:** `FileNotFoundError: Data directory not found`

**Solution:**
- Verify `Retinotopy/data/raw/converted` directory exists
- Specify `--data_dir` with correct path
- Ensure data files are properly formatted

#### 5. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
- Reduce batch size (if applicable)
- Use CPU instead: Set `CUDA_VISIBLE_DEVICES=""` before running
- Process fewer subjects at once in parallel mode
- Reduce `--num_workers` in batch inference

#### 6. Model Architecture Mismatch

**Error:** `RuntimeError: Error(s) in loading state_dict`

**Solution:**
- Ensure checkpoint matches model type specified in filename
- For `transolver_optionC`, provide correct hyperparameters (`--n_layers`, `--n_hidden`, etc.)
- Check that model was trained with same number of features (myelination setting)

### Getting Help

For additional help:

1. Check script help: `python run_inference_from_fslr_curv_using_checkpoint.py --help`
2. Check batch script help: `python scripts/run_batch_inference_from_checkpoints.py --help`
3. Review checkpoint path format requirements above
4. Verify data directory structure matches expected format

---

## Notes

- Both scripts automatically detect GPU availability and use CUDA if available
- Output files are saved in PyTorch format (`.pt`) containing both predictions and metadata
- Batch inference script provides progress updates and summary statistics
- Parallel execution in batch script uses multiprocessing and may require sufficient system resources
- Subject IDs should match those in `Retinotopy/data/list_subj` file
- Checkpoint paths can be absolute or relative to current working directory
