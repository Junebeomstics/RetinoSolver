# Unified Training Script for deepRetinotopy Models

This script integrates baseline models and Transolver-based models to run various experiments with a single script.

## File Structure

```
Models/
├── train_unified.py              # Unified training script
├── run_all_experiments.sh        # Script to run all experiment combinations
├── run_single_experiment.sh      # Example script to run a single experiment
├── models/                       # Model classes
│   ├── __init__.py
│   ├── baseline.py                # Baseline model
│   ├── transolver_optionA.py     # Transolver Option A model
│   ├── transolver_optionB.py     # Transolver Option B model
│   ├── physics_attention.py      # Physics Attention modules
│   └── utils.py                  # Utility functions
└── README_unified_training.md    # This file
```

## Output Directory Structure

Experiment results are saved in the following structure:

```
output/
├── baseline_ecc_Left/
│   ├── baseline_ecc_Left_output_epoch25.pt
│   ├── baseline_ecc_Left_output_epoch50.pt
│   ├── ...
│   └── baseline_ecc_Left_model.pt
├── baseline_ecc_Right/
├── baseline_PA_Left/
├── transolver_optionA_ecc_Left/
├── transolver_optionA_ecc_Right/
├── transolver_optionA_PA_Left/
├── transolver_optionB_ecc_Left/
└── ...
```

Each experiment is saved in a separate folder named `{model_type}_{prediction}_{hemisphere}`.

## Usage

### 1. Running a Single Experiment (Using Docker)

**Running with Docker (Recommended):**

```bash
# Basic usage
./run_single_experiment.sh baseline eccentricity Left

# Specify Docker image
DOCKER_IMAGE=your_image:tag ./run_single_experiment.sh baseline eccentricity Left

# Disable GPU
USE_GPU=false ./run_single_experiment.sh baseline eccentricity Left

# Disable Neptune
USE_NEPTUNE=false ./run_single_experiment.sh baseline eccentricity Left
```

The script automatically:
- Checks if Docker image exists
- Installs required packages (neptune, einops)
- Runs training in Docker container

**Running locally:**

```bash
python train_unified.py \
    --model_type baseline \
    --prediction eccentricity \
    --hemisphere Left \
    --n_epochs 200 \
    --lr_init 0.01 \
    --lr_decay_epoch 100 \
    --lr_decay 0.005
```

### 2. Running All Experiments

```bash
./run_all_experiments.sh
```

This script runs all combinations of:
- Model types: `baseline`, `transolver_optionA`, `transolver_optionB`
- Predictions: `eccentricity`, `polarAngle`
- Hemispheres: `Left`, `Right`

A total of 12 experiments will be run (3 × 2 × 2).

### 3. Enabling Neptune Logging

To use Neptune:

```bash
python train_unified.py \
    --model_type baseline \
    --prediction eccentricity \
    --hemisphere Left \
    --use_neptune \
    --project your_project_name \
    --api_token your_api_token
```

Or use environment variable:

```bash
export NEPTUNE_API_TOKEN=your_api_token
python train_unified.py \
    --model_type baseline \
    --prediction eccentricity \
    --hemisphere Left \
    --use_neptune \
    --project your_project_name
```

## Main Arguments

### Required Arguments

- `--model_type`: Model type selection
  - `baseline`: Basic SplineConv model
  - `transolver_optionA`: Transolver Physics Attention (without edge information)
  - `transolver_optionB`: Transolver Physics Attention (with encoded edge information)

- `--prediction`: Prediction target
  - `eccentricity`: Eccentricity prediction
  - `polarAngle`: Polar Angle prediction

- `--hemisphere`: Hemisphere selection
  - `Left`: Left hemisphere
  - `Right`: Right hemisphere

### Optional Arguments

- `--n_epochs`: Number of training epochs (default: 200)
- `--lr_init`: Initial learning rate (default: 0.01)
- `--lr_decay_epoch`: Epoch for learning rate decay (default: 100)
- `--lr_decay`: Learning rate after decay (default: 0.005)
- `--interm_save_every`: Intermediate result save interval (default: 25)
- `--batch_size`: Batch size (default: 1)
- `--n_examples`: Number of examples (default: 181)
- `--output_dir`: Output directory (default: ./output)
- `--myelination`: Whether to use Myelination features (default: True)

### Neptune Arguments

- `--use_neptune`: Enable Neptune logging
- `--project`: Neptune project name
- `--api_token`: Neptune API token (can use `NEPTUNE_API_TOKEN` environment variable)

## Model Descriptions

### Baseline Model
- Pure SplineConv-based model
- Consists of 12 SplineConv layers
- Directly uses edge information

### Transolver Option A
- SplineConv + Physics Attention hybrid model
- Does not use edge information
- Physics Attention learns physical states using only node features

### Transolver Option B
- SplineConv + Physics Attention hybrid model
- Encodes edge information as features for use
- Converts K-NN distance, node degree, local density, etc. into features

## Examples

### Eccentricity Prediction (Left Hemisphere, Baseline)

```bash
python train_unified.py \
    --model_type baseline \
    --prediction eccentricity \
    --hemisphere Left
```

### Polar Angle Prediction (Right Hemisphere, Transolver Option A)

```bash
python train_unified.py \
    --model_type transolver_optionA \
    --prediction polarAngle \
    --hemisphere Right
```

### Transolver Option B with Neptune

```bash
python train_unified.py \
    --model_type transolver_optionB \
    --prediction eccentricity \
    --hemisphere Left \
    --use_neptune \
    --project your_project \
    --api_token your_token
```

## Notes

### When Using Docker

1. **Pull Docker Image:**
   ```bash
   # Pull the pre-built Docker image
   docker pull vnmd/deepretinotopy_1.0.18:latest
   ```

2. **Automatic Package Installation:**
   - The script automatically installs `neptune` and `einops`
   - Since it installs on each run, the first run may take some time

3. **Change Docker Image Name:**
   - Can be specified via environment variable: `DOCKER_IMAGE=your_image:tag`

4. **GPU Usage:**
   - Uses GPU by default (`USE_GPU=true`)
   - To disable GPU: `USE_GPU=false`

### General Notes

1. If not using Neptune, set `USE_NEPTUNE=false` or modify it in the script.

2. Each experiment runs independently, and results are saved in separate folders.

3. If not using Docker, the following packages are required locally:
   ```bash
   pip install einops neptune
   ```
