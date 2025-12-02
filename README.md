# DeepRetinotopy

This repository contains all source code necessary to replicate our recent work entitled "Predicting the retinotopic organization of human visual cortex from anatomy using geometric deep learning" available in [NeuroImage](https://www.sciencedirect.com/science/article/pii/S1053811921008971).

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

This folder contains all source code necessary to train a new model and to generate predictions on the test dataset 
using our pre-trained models.

## Retinotopy

This folder contains all source code necessary to replicate datasets generation, in addition to functions and labels 
used for figures and models' evaluation. 

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
Fernanda Ribeiro <[fernanda.ribeiro@uq.edu.au](fernanda.ribeiro@uq.edu.au)>

Alex Puckett <[a.puckett@uq.edu.au](a.puckett@uq.edu.au)>
