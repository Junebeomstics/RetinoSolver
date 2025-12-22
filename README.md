# DeepRetinotopy

This repository contains all source code necessary to replicate our recent work entitled "Predicting the retinotopic organization of human visual cortex from anatomy using geometric deep learning" available in [NeuroImage](https://www.sciencedirect.com/science/article/pii/S1053811921008971).

**Note:** This repository is developed based on the original [deepRetinotopy repository](https://github.com/Puckett-Lab/deepRetinotopy). We extend the original implementation with additional features and improvements.

## Table of Contents
* [Quick Start with Docker](#quick-start-with-docker)
* [Running Experiments](#running-experiments)
* [Manuscript](#manuscript)
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
│   ├── converted/
│   └── surfaces/
└── processed/
```

**Processing Raw Data:**

Before running experiments, you need to process the raw data files. The raw data should be placed in `Retinotopy/data/raw/`, and then processed using the `process_raw.py` script to generate the processed data files in `Retinotopy/data/processed/`.

**Processing raw data:**

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

This script will:
- Process raw data files from `Retinotopy/data/raw/`
- Generate processed data files in `Retinotopy/data/processed/`
- Create datasets for both eccentricity and polar angle predictions
- Process data for both left and right hemispheres

The processed files will be automatically used by the training scripts.

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
  - Model types: `baseline`, `transolver_optionA`, `transolver_optionB`
  - Predictions: `eccentricity`, `polarAngle`
  - Hemispheres: `Left`, `Right`

**Configuration:**

You can customize the script by editing `Models/run_all_experiments.sh`:
- `DOCKER_IMAGE`: Docker image name (default: `vnmd/deepretinotopy_1.0.18:latest`)
- `USE_GPU`: Enable/disable GPU (default: `true`)
- `USE_NEPTUNE`: Enable/disable Neptune logging (default: `true`)
- `MODEL_TYPES`, `PREDICTIONS`, `HEMISPHERES`: Arrays defining which experiments to run
- Training hyperparameters: `N_EPOCHS`, `LR_INIT`, `LR_DECAY_EPOCH`, etc.

**Example: Running specific experiments**

Edit the arrays in `run_all_experiments.sh`:
```bash
MODEL_TYPES=("baseline" "transolver_optionA")
PREDICTIONS=("eccentricity")
HEMISPHERES=("Left")
```

### Running Transolver Option C Experiments

For `transolver_optionC` model, which uses the original Transolver hyperparameters optimized for unstructured mesh experiments, use the dedicated script:

```bash
cd Models
chmod +x run_transolver_optionC_experiments_with_original_hyperparameters.sh
./run_transolver_optionC_experiments_with_original_hyperparameters.sh
```

This script runs `transolver_optionC` with the following optimized hyperparameters:
- **Training**: 500 epochs, AdamW optimizer, cosine scheduler
- **Learning rate**: Initial 0.001, decays to 0.0001 at epoch 250
- **Architecture**: 8 layers, 128 hidden dimensions, 8 attention heads
- **Other settings**: Weight decay 1e-5, max gradient norm 0.1, dropout 0.0

The script automatically runs all combinations of:
- Predictions: `eccentricity`, `polarAngle`
- Hemispheres: `Left`, `Right`

**Configuration:**

You can customize the script by editing `Models/run_transolver_optionC_experiments_with_original_hyperparameters.sh`:
- `DOCKER_IMAGE`: Docker image name (default: `vnmd/deepretinotopy_1.0.18:latest`)
- `USE_GPU`: Enable/disable GPU (default: `true`)
- `USE_NEPTUNE`: Enable/disable Neptune logging (default: `true`)
- `PREDICTIONS`, `HEMISPHERES`: Arrays defining which experiments to run
- Architecture hyperparameters: `N_LAYERS`, `N_HIDDEN`, `N_HEADS`, `SLICE_NUM`, etc.

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
- `--prediction`: Prediction target (`eccentricity`, `polarAngle`)
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

**Neptune Logging:**
- `--use_neptune`: Enable Neptune logging
- `--project`: Neptune project name
- `--api_token`: Neptune API token (or set `NEPTUNE_API_TOKEN` environment variable)

For more detailed information, see [Models/README_unified_training.md](Models/README_unified_training.md).

### Output Structure

Results are saved in the following structure:
```
Models/output/
├── RET-XXX
│   ├── baseline_ecc_Left_best_model_epoch25.pt
│   ├── baseline_ecc_Left_final_model.pt
│   └── ...
├── RET-XXX
├── RET-XXX
└── ...
```

Each experiment creates a folder named the Neptune project folder, which includes files named `{model_type}_{prediction}_{hemisphere}`, containing both the best and final models.

## Manuscript

This folder contains all source code necessary to reproduce all figures and summary statistics in our manuscript.

## Models

This folder contains all source code necessary to train new models and generate predictions. The current implementation uses a unified training system with the following scripts:

### Main Scripts

- **`train_unified.py`**: Unified training script that supports multiple model architectures (baseline, transolver_optionA, transolver_optionB, transolver_optionC). This is the main script used for training and evaluation. It handles:
  - Model training with configurable hyperparameters
  - Validation and test set evaluation
  - Checkpoint saving and loading
  - Neptune logging integration
  - Early stopping
  - Test set evaluation with results saved in both `.pt` and `.npz` formats

- **`run_all_experiments.sh`**: Automated script to run all experiment combinations. It:
  - Automatically pulls Docker image if not present
  - Creates or reuses a Docker container
  - Runs all combinations of model types, predictions, and hemispheres
  - Configurable via environment variables or script editing
  - Supports Neptune logging

- **`run_transolver_optionC_experiments_with_original_hyperparameters.sh`**: Dedicated script for running `transolver_optionC` experiments with optimized hyperparameters matching the original Transolver experiments (500 epochs, AdamW optimizer, cosine scheduler, etc.)

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

**Original Authors:**
- Fernanda Ribeiro <[fernanda.ribeiro@uq.edu.au](mailto:fernanda.ribeiro@uq.edu.au)>
- Alex Puckett <[a.puckett@uq.edu.au](mailto:a.puckett@uq.edu.au)>
