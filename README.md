# DeepRetinotopy

This repository extends and improves upon the original [deepRetinotopy repository](https://github.com/Puckett-Lab/deepRetinotopy) developed by Ribeiro et al. (2021) in their work "Predicting the retinotopic organization of human visual cortex from anatomy using geometric deep learning" published in [NeuroImage](https://www.sciencedirect.com/science/article/pii/S1053811921008971).

**Purpose:** This repository aims to evaluate and compare multiple model architectures beyond the original deepRetinotopy baseline, including various Transolver-based architectures. Additionally, we extend the evaluation to test multiple retinotopic variables (eccentricity, polar angle, and pRF size) beyond the original PRF predictions.

## Table of Contents
* [Quick Start with Docker](#quick-start-with-docker)
* [Running Experiments](#running-experiments)
* [Running Inference on FreeSurfer Data](#running-inference-on-freesurfer-data)
* [Models](#models)
* [Retinotopy](#retinotopy)
* [Citation](#citation)
* [Contact](#contact)

## Quick Start with Docker

The easiest way to run experiments is using the pre-built Docker image.

### Prerequisites

- Docker installed (version 20.10 or higher recommended)
- NVIDIA Docker (nvidia-container-toolkit) for GPU support (optional but recommended)

**Installing NVIDIA Container Toolkit (for GPU support):**

If you want to use GPU with Docker, you need to install nvidia-container-toolkit. You can use the provided installation script:

```bash
# Make the script executable
chmod +x install_nvidia_container_toolkit.sh

# Run the installation script (requires sudo)
sudo ./install_nvidia_container_toolkit.sh
```

Alternatively, you can install it manually following the [official NVIDIA documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Pulling the Docker Image

```bash
# Pull the pre-built Docker image
docker pull vnmd/deepretinotopy_1.0.18:latest
```

The image is ready to use. The experiment scripts will automatically use this image when running experiments.

### Preparing Data

**Downloading Data:**

Due to the large size of the data files, the `Retinotopy/data` folder contents are hosted on Google Drive instead of being included in this repository. Please download the data from the following link:

📦 **Data Download**: [Google Drive Link](https://drive.google.com/drive/folders/1o-MFVX_vOQ82qFhDibA8fBIP28_DaKpl?usp=sharing)

After downloading, extract the contents to `Retinotopy/data/` directory. The folder structure should be:
```
Retinotopy/data/
├── raw/
│   ├── converted/        # Pre-converted data files (.mat files)
│   └── surfaces/         # Surface files (.surf.gii files)
└── processed/            # Pre-processed dataset files (.pt files)
```

**Important:** The Google Drive package includes both `converted/` and `processed/` folders. If you have the complete package, you can skip the data processing step and proceed directly to running experiments.

**Data Processing:**

The Google Drive data package includes both `converted/` and `processed/` folders. If you have downloaded the complete data package:

- **If `Retinotopy/data/processed/` folder exists**: The training scripts will automatically use the pre-processed data files. No additional processing is needed.
- **If only `Retinotopy/data/raw/converted/` folder exists**: The training scripts will automatically process the data on first run. The processed files will be generated in `Retinotopy/data/processed/` automatically.

**Manual Processing (Optional):**

If you need to manually process the raw data (e.g., to regenerate processed files or process additional prediction types), you can use the `process_raw.py` script:

```bash
# Make sure the Docker image is pulled
docker pull vnmd/deepretinotopy_1.0.18:latest

# Process raw data files
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -v $(pwd)/Retinotopy/data:/workspace/Retinotopy/data \
  -w /workspace \
  vnmd/deepretinotopy_1.0.18:latest \
  python process_raw.py
```

**Note:** The training scripts (`train_unified.py`) automatically check for existing processed files and will skip processing if they already exist. If you have downloaded the complete data package from Google Drive, you can proceed directly to running experiments without running `process_raw.py`.

## Running Experiments

### Using the Unified Training Script

This repository provides a unified training script (`Models/train_unified.py`) that supports multiple model architectures:
- `baseline`: Original SplineConv-based model
- `transolver_optionA`: Hybrid Transolver with SplineConv & Physics Attention (without edge information)
- `transolver_optionB`: Hybrid Transolver with SplineConv & Physics Attention (with encoded edge information)
- `transolver_optionC`: Original Transolver with full Physics Attention architecture 

### Running All Experiments (Recommended)

The easiest way to run all experiment combinations is using the provided shell script:

```bash
cd Models
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

This script will:
- Automatically check for Docker and pull the required Docker image if not present
- Create or reuse a Docker container
- Run all combinations of:
  - Model types: `baseline`, `transolver_optionA`, `transolver_optionB`, `transolver_optionC`
  - Predictions: `eccentricity`, `polarAngle`, `pRFsize`
  - Hemispheres: `Left`, `Right`
- Automatically apply optimized hyperparameters for `transolver_optionC` when selected (see Hyperparameters section below)
- Support Wandb logging for experiment tracking

**Configuration:**

You can customize the script by editing `Models/run_all_experiments.sh`:
- `DOCKER_IMAGE`: Docker image name (default: `vnmd/deepretinotopy_1.0.18:latest`)
- `USE_GPU`: Enable/disable GPU (default: `true`)
- `USE_WANDB`: Enable/disable Wandb logging (default: `true`)
- `WANDB_PROJECT`: Wandb project name (default: `retinotopic_mapping`)
- `MODEL_TYPES`, `PREDICTIONS`, `HEMISPHERES`: Arrays defining which experiments to run
- Training hyperparameters: `N_EPOCHS`, `LR_INIT`, `LR_DECAY_EPOCH`, etc.

**Hyperparameters:**

The script uses different hyperparameter settings for `transolver_optionC` compared to other models, considering the original Transolver architecture requirements:
- **transolver_optionC**: 500 epochs, AdamW optimizer, cosine scheduler, initial learning rate 0.001 (decays to 0.0001 at epoch 250), 8 layers, 128 hidden dimensions, 8 attention heads, weight decay 1e-5, max gradient norm 0.1, dropout 0.0
- **Other models** (baseline, transolver_optionA, transolver_optionB): Use standard hyperparameters defined by `N_EPOCHS`, `LR_INIT`, `LR_DECAY_EPOCH`, etc.

You can customize the transolver_optionC parameters by editing `Models/run_all_experiments.sh`:
- `N_EPOCHS_OPTIONC`, `LR_INIT_OPTIONC`, `LR_DECAY_OPTIONC`: Training parameters
- `N_LAYERS_OPTIONC`, `N_HIDDEN_OPTIONC`, `N_HEADS_OPTIONC`: Architecture parameters
- `SLICE_NUM_OPTIONC`, `MLP_RATIO_OPTIONC`, `DROPOUT_OPTIONC`: Model-specific parameters

**Example: Running specific experiments**

Edit the arrays in `run_all_experiments.sh`:
```bash
MODEL_TYPES=("baseline" "transolver_optionA")
PREDICTIONS=("eccentricity")
HEMISPHERES=("Left")
```

### Running a Single Experiment

The experiment scripts automatically use Docker. To run a single experiment manually, you can execute the training script inside a Docker container:

```bash
# Make sure the Docker image is pulled
docker pull vnmd/deepretinotopy_1.0.18:latest

# Run a single experiment in Docker
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -v $(pwd)/Retinotopy/data:/workspace/Retinotopy/data \
  -w /workspace \
  vnmd/deepretinotopy_1.0.18:latest \
  bash -c "cd Models && python train_unified.py \
    --model_type baseline \
    --prediction eccentricity \
    --hemisphere Left \
    --n_epochs 200 \
    --lr_init 0.01 \
    --lr_decay_epoch 100 \
    --lr_decay 0.005"
```

### Available Arguments

**Required:**
- `--model_type`: Model architecture (`baseline`, `transolver_optionA`, `transolver_optionB`, `transolver_optionC`)
- `--prediction`: Prediction target (`eccentricity`, `polarAngle`, `pRFsize`)
- `--hemisphere`: Hemisphere (`Left`, `Right`)

**Optional:**
- `--n_epochs`: Number of training epochs (default: 200)
- `--lr_init`: Initial learning rate (default: 0.01)
- `--lr_decay_epoch`: Epoch for learning rate decay (default: 100)
- `--lr_decay`: Learning rate after decay (default: 0.005)
- `--scheduler`: Learning rate scheduler (`step`, `cosine`, `onecycle`, default: `cosine`)
- `--optimizer`: Optimizer type (`Adam`, `AdamW`, default: `AdamW`)
- `--weight_decay`: Weight decay for optimizer (default: 1e-5)
- `--max_grad_norm`: Maximum gradient norm for clipping (default: 0.1)
- `--batch_size`: Batch size (default: 1)
- `--n_examples`: Number of examples (default: 181)
- `--output_dir`: Output directory (default: `./output`)

**Transolver Option C specific arguments:**
- `--n_layers`: Number of transformer layers (default: 8)
- `--n_hidden`: Hidden dimension size (default: 128)
- `--n_heads`: Number of attention heads (default: 8)
- `--slice_num`: Number of slices for physics attention (default: 64)
- `--mlp_ratio`: MLP ratio in transformer blocks (default: 1)
- `--dropout`: Dropout rate (default: 0.0)
- `--ref`: Reference parameter (default: 8)
- `--unified_pos`: Unified position encoding flag (default: 0)

For more detailed information, see [Models/README_unified_training.md](Models/README_unified_training.md).

### Output Structure

Results are saved in the following structure (default output directory: `Models/output_wandb/`):

**Without Wandb (or when Wandb is disabled):**
```
Models/output_wandb/
├── {prediction}_{hemisphere}_{model_type}/
│   ├── {prediction_short}_{hemisphere}_{model_type}_best_model_epoch{epoch}.pt
│   ├── {prediction_short}_{hemisphere}_{model_type}_best_test_results.pt
│   ├── {prediction_short}_{hemisphere}_{model_type}_final_model.pt
│   └── {prediction_short}_{hemisphere}_{model_type}_final_test_results.pt
├── {prediction}_{hemisphere}_{model_type}_noMyelin/
│   └── ...
└── ...
```

**With Wandb enabled:**
```
Models/output_wandb/
├── {wandb_run_name}/
│   ├── {prediction_short}_{hemisphere}_{model_type}_best_model_epoch{epoch}.pt
│   ├── {prediction_short}_{hemisphere}_{model_type}_best_test_results.pt
│   ├── {prediction_short}_{hemisphere}_{model_type}_final_model.pt
│   └── {prediction_short}_{hemisphere}_{model_type}_final_test_results.pt
└── ...
```

**File naming convention:**
- `{prediction_short}`: `ecc` (eccentricity), `PA` (polarAngle), `size` (pRFsize)
- `{hemisphere}`: `Left` or `Right`
- `{model_type}`: `baseline`, `transolver_optionA`, `transolver_optionB`, `transolver_optionC`
- `{myelination_suffix}`: `_noMyelin` if myelination is disabled, empty otherwise
- `{wandb_run_name}`: Automatically generated run name when Wandb is enabled (includes model type, prediction, hemisphere, and myelination info)

**Output files:**
- **Best model** (`*_best_model_epoch{epoch}.pt`): Model checkpoint from the epoch with the lowest validation MAE_thr
- **Final model** (`*_final_model.pt`): Model checkpoint from the final training epoch
- **Best test results** (`*_best_test_results.pt`): Test set evaluation results using the best model (contains predictions, ground truth, R2 values, and MAE metrics)
- **Final test results** (`*_final_test_results.pt`): Test set evaluation results using the final model

## Running Inference on FreeSurfer Data

This repository provides a pipeline to run inference on new subjects with FreeSurfer-processed data. The `run_from_freesurfer` directory contains scripts for processing FreeSurfer outputs and generating retinotopic predictions.

### Overview

The FreeSurfer inference pipeline consists of three main steps:

1. **Native to fsaverage conversion**: Converts FreeSurfer native surface data to fsaverage space (standard template space used for training)
2. **Inference**: Runs the trained model to generate retinotopic predictions in fsaverage space
3. **Fsaverage to native conversion**: Converts predictions back to the subject's native space

### Prerequisites

- FreeSurfer-processed subject data (with `surf/` directory containing surface files)
- Trained model checkpoint (from the training experiments)
- Docker (same image used for training)

### Running the Pipeline

The main script `run_from_freesurfer/run_deepRetinotopy_freesurfer_with_docker.sh` handles the complete pipeline:

```bash
cd run_from_freesurfer
chmod +x run_deepRetinotopy_freesurfer_with_docker.sh

./run_deepRetinotopy_freesurfer_with_docker.sh \
  --freesurfer_dir /path/to/freesurfer/subject/directory \
  --subject_id SUBJECT_ID \
  --hemisphere lh \
  --model_type baseline \
  --prediction eccentricity \
  --checkpoint /path/to/checkpoint.pt
```

### Required Arguments

- `--freesurfer_dir`: Path to FreeSurfer subject directory (should contain `surf/` directory)
- `--subject_id`: Subject identifier
- `--hemisphere`: Hemisphere (`lh` or `rh`)
- `--model_type`: Model architecture (`baseline`, `transolver_optionA`, `transolver_optionB`, `transolver_optionC`)
- `--prediction`: Prediction target (`eccentricity`, `polarAngle`, `pRFsize`)

### Optional Arguments

- `--checkpoint`: Path to model checkpoint file (if not provided, script will search for best checkpoint automatically)
- `--myelination`: Use myelination features (`True` or `False`, default: `False`)
- `--output_dir`: Output directory for results (default: same as `freesurfer_dir`)
- `--skip_preprocessing`: Skip native to fsaverage conversion (if already done)
- `--skip_native_conversion`: Skip fsaverage to native conversion
- `--skip_myelin`: Skip myelin map generation

### Automatic Checkpoint Detection

If `--checkpoint` is not specified, the script automatically searches for the best checkpoint based on:
- Model type (`--model_type`)
- Prediction target (`--prediction`)
- Hemisphere (`--hemisphere`)
- Myelination setting (`--myelination`)

The search pattern is: `Models/output_wandb/{prediction}_{hemisphere}_{model_type}[_noMyelin]/{prediction_short}_{hemisphere}_{model_type}[_noMyelin]_best_model_epoch*.pt`

### Output

The pipeline generates retinotopic predictions in both fsaverage and native spaces:
- **Fsaverage space**: Predictions aligned with the training data template
- **Native space**: Predictions mapped back to the subject's individual anatomy

Results are saved in the specified output directory (or `freesurfer_dir` if not specified) as GIFTI surface files (`.func.gii`).

### Additional Scripts

The `run_from_freesurfer` directory also contains helper scripts:
- `0_generate_myelin.sh`: Generate myelin maps from FreeSurfer data
- `1_native2fsaverage.sh`: Convert native surfaces to fsaverage space
- `2_fsaverage2native.sh`: Convert fsaverage predictions back to native space
- `midthickness_surf.py`: Generate midthickness surfaces

These scripts are automatically called by the main pipeline script and can also be used independently if needed.

## Models

This folder contains all source code necessary to train new models and generate predictions. The current implementation uses a unified training system with the following scripts:

### Main Scripts

- **`train_unified.py`**: Unified training script that supports multiple model architectures (baseline, transolver_optionA, transolver_optionB, transolver_optionC). This is the main script used for training and evaluation. It handles:
  - Model training with configurable hyperparameters
  - Validation and test set evaluation
  - Checkpoint saving and loading
  - Wandb logging integration
  - Early stopping
  - Test set evaluation with results saved in both `.pt` and `.npz` formats

- **`run_all_experiments.sh`**: Automated script to run all experiment combinations. It:
  - Automatically pulls Docker image if not present
  - Creates or reuses a Docker container
  - Runs all combinations of model types (including `transolver_optionC`), predictions, and hemispheres
  - Automatically applies optimized hyperparameters for `transolver_optionC` when selected
  - Configurable via environment variables or script editing
  - Supports Wandb logging for experiment tracking

- **`run_transolver_optionC_hyperparameter_search.sh`**: Script for hyperparameter search experiments with `transolver_optionC` (optional, for advanced users)

- **`run_test_from_checkpoint.sh`**: Script to load a checkpoint and run test set evaluation only (no training). Useful for:
  - Evaluating pre-trained models
  - Testing models from different checkpoints
  - Running inference on test sets without retraining

### Legacy Scripts (from Original Paper)

**Note:** The scripts mentioned in `Models/README.md` (e.g., `deepRetinotopy_ecc_LH.py`, `deepRetinotopy_PA_LH.py`, `ModelGeneralizability_*.py`) are from the original paper implementation and are **not actively used** in the current workflow. The current implementation uses the unified training system described above. These legacy scripts are kept for reference and reproducibility of the original paper results.

## Retinotopy

This folder contains all source code necessary to replicate datasets generation, in addition to functions and labels 
used for figures and models' evaluation.

**Note:** The data files in `Retinotopy/data/` are not included in this repository due to their large size. Please download them from [Google Drive](https://drive.google.com/drive/folders/1o-MFVX_vOQ82qFhDibA8fBIP28_DaKpl?usp=sharing) and place them in the `Retinotopy/data/` directory before running experiments. 

## Citation

Please cite our paper if you used our model or if it was somewhat helpful for you :wink:

	@article{Ribeiro2021,
		author = {Ribeiro, Fernanda L and Bollmann, Steffen and Puckett, Alexander M},
		doi = {https://doi.org/10.1016/j.neuroimage.2021.118624},
		issn = {1053-8119},
		journal = {NeuroImage},
		keywords = {cortical surface, high-resolution fMRI, machine learning, manifold, visual hierarchy,Vision},
		pages = {118624},
		title = {{Predicting the retinotopic organization of human visual cortex from anatomy using geometric deep learning}},
		url = {https://www.sciencedirect.com/science/article/pii/S1053811921008971},
		year = {2021}
	}


## Contact

For questions and inquiries about this repository, please contact:

Junbeom Kwon <[kjb961013@gmail.com](mailto:kjb961013@gmail.com)>

**Contributors:**
- Sanga Lee (PhD Student, Boston University)
- Satvik Agarwal (Undergraduate Intern, University of Texas at Austin)
- Gabriele Amorosino (Postdoc, University of Texas at Austin)

**Original Authors:**
- Fernanda Ribeiro <[fernanda.ribeiro@uq.edu.au](mailto:fernanda.ribeiro@uq.edu.au)>
- Alex Puckett <[a.puckett@uq.edu.au](mailto:a.puckett@uq.edu.au)>
