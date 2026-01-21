# Seed-based Subject Splitting Guide

This guide explains how to use the seed-based subject splitting feature for running experiments with different subject combinations.

## Overview

The system now supports different subject splits using seeds (0, 1, 2, etc.). Each seed produces a different train/dev/test split, allowing you to run multiple experiments with different subject combinations.

## Step 1: Generate Subject Splits

First, generate the subject splits for different seeds:

```bash
python generate_seed_splits.py
```

This will create subject list files in `Retinotopy/data/subject_splits/`:
- `seed0/train_subjects.txt`, `seed0/dev_subjects.txt`, `seed0/test_subjects.txt`
- `seed1/train_subjects.txt`, `seed1/dev_subjects.txt`, `seed1/test_subjects.txt`
- `seed2/train_subjects.txt`, `seed2/dev_subjects.txt`, `seed2/test_subjects.txt`

## Step 2: Process Data for Each Seed

The processed data files will be automatically generated when you first run training with a specific seed. The dataset class will:
1. Shuffle subjects using the specified seed
2. Split into train (0-160), dev (161-170), test (171-end)
3. Save processed files with seed suffix (e.g., `training_ecc_LH_myelincurv_ROI_seed0.pt`)

## Step 3: Run Training with Seed

Use the `--data_seed` parameter to specify which seed to use:

```bash
# Train with seed 0
python Models/train_unified.py \
    --model_type transolver_optionC \
    --prediction pRFsize \
    --hemisphere Right \
    --data_seed 0 \
    --myelination True \
    --n_epochs 200

# Train with seed 1
python Models/train_unified.py \
    --model_type transolver_optionC \
    --prediction pRFsize \
    --hemisphere Right \
    --data_seed 1 \
    --myelination True \
    --n_epochs 200

# Train with seed 2
python Models/train_unified.py \
    --model_type transolver_optionC \
    --prediction pRFsize \
    --hemisphere Right \
    --data_seed 2 \
    --myelination True \
    --n_epochs 200
```

## File Naming Convention

### Processed Data Files
- Without seed: `training_ecc_LH_myelincurv_ROI.pt`
- With seed 0: `training_ecc_LH_myelincurv_ROI_seed0.pt`
- With seed 1: `training_ecc_LH_myelincurv_ROI_seed1.pt`
- With seed 2: `training_ecc_LH_myelincurv_ROI_seed2.pt`

### Model Checkpoints
- Without seed: `pRFsize_Right_transolver_optionC_best_model_epoch50.pt`
- With seed 0: `pRFsize_Right_transolver_optionC_seed0_best_model_epoch50.pt`
- With seed 1: `pRFsize_Right_transolver_optionC_seed1_best_model_epoch50.pt`
- With seed 2: `pRFsize_Right_transolver_optionC_seed2_best_model_epoch50.pt`

### Wandb Project Names
- Without seed: `deepRetinotopy`
- With seed 0: `deepRetinotopy_seed0`
- With seed 1: `deepRetinotopy_seed1`
- With seed 2: `deepRetinotopy_seed2`

## Subject List Files

Subject lists are saved in `Retinotopy/data/subject_splits/seedX/`:
- `train_subjects.txt`: List of subject IDs in training set
- `dev_subjects.txt`: List of subject IDs in development set
- `test_subjects.txt`: List of subject IDs in test set
- `all_subjects.txt`: Combined list with comments indicating splits

## Notes

- If `--data_seed` is not specified (or set to `None`), the system uses the default seed=1 (original behavior)
- The seed affects only the subject shuffling/splitting, not the model training randomness (use `--seed` for that)
- Each seed produces a different but reproducible subject split
- Processed data files are cached, so subsequent runs with the same seed will use the cached files
