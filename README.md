# DeepRetinotopy

This repository contains all source code necessary to replicate our recent work entitled "Predicting the retinotopic organization of human visual cortex from anatomy using geometric deep learning" available in [NeuroImage](https://www.sciencedirect.com/science/article/pii/S1053811921008971).

## Table of Contents
* [Installation and requirements](#installation-and-requirements)
* [Quick Start with Docker](#quick-start-with-docker)
* [Running Experiments](#running-experiments)
* [Manuscript](#manuscript)
* [Models](#models)
* [Retinotopy](#retinotopy)
* [Citation](#citation)
* [Contact](#contact)


## Installation and requirements 

Models were generated using [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/). Since this package is under constant updates, we highly recommend that 
you follow the following steps to run our models locally:

- Create a conda environment (or docker container)
- Install torch first:
		
```bash
$ pip install torch==0.4.1    
$ pip install torchvision==0.2.1
```
	
	
- Install torch-scatter, torch-sparse, torch-cluster, torch-spline-conv and torch-geometric:

```bash
$ pip install torch-scatter==1.0.4
$ pip install torch-sparse==0.2.2
$ pip install torch-cluster==1.1.5
$ pip install torch-spline-conv==1.0.4
$ pip install torch-geometric==0.3.1
```

- Install the remaining required packages that are available at requirements.txt: 

```bash
$ pip install -r requirements.txt
```   
    
- Clone DeepRetinotopy:
```bash
$ git clone git@github.com:Puckett-Lab/deepRetinotopy.git
```   

Finally, install the following git repository for plots:
```bash
$ pip install git+https://github.com/felenitaribeiro/nilearn.git
```

## Quick Start with Docker

The easiest way to run experiments is using Docker. This project includes Docker configuration files for easy setup.

### Prerequisites

- Docker installed (version 20.10 or higher recommended)
- NVIDIA Docker (nvidia-container-toolkit) for GPU support (optional but recommended)
- Docker Compose (optional, for easier container management)

### Building the Docker Image

```bash
# Build the Docker image
docker build -t vnmd/deepretinotopy_1.0.18:latest .
```

For detailed Docker setup instructions, see [DOCKER_README.md](DOCKER_README.md).

## Running Experiments

### Using the Unified Training Script

This repository provides a unified training script (`Models/train_unified.py`) that supports multiple model architectures:
- `baseline`: Original SplineConv-based model
- `transolver_optionA`: Transolver with Physics Attention (without edge information)
- `transolver_optionB`: Transolver with Physics Attention (with encoded edge information)

### Running All Experiments (Recommended)

The easiest way to run all experiment combinations is using the provided shell script:

```bash
cd Models
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

This script will:
- Automatically check for Docker and the required Docker image
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

### Running a Single Experiment

**Using Docker (Recommended):**

```bash
cd Models
python train_unified.py \
    --model_type baseline \
    --prediction eccentricity \
    --hemisphere Left \
    --n_epochs 200 \
    --lr_init 0.01 \
    --lr_decay_epoch 100 \
    --lr_decay 0.005
```

**Direct execution (without Docker):**

Make sure you have all dependencies installed (see [Installation and requirements](#installation-and-requirements)), then:

```bash
cd Models
python train_unified.py \
    --model_type baseline \
    --prediction eccentricity \
    --hemisphere Left
```

### Available Arguments

**Required:**
- `--model_type`: Model architecture (`baseline`, `transolver_optionA`, `transolver_optionB`)
- `--prediction`: Prediction target (`eccentricity`, `polarAngle`)
- `--hemisphere`: Hemisphere (`Left`, `Right`)

**Optional:**
- `--n_epochs`: Number of training epochs (default: 200)
- `--lr_init`: Initial learning rate (default: 0.01)
- `--lr_decay_epoch`: Epoch for learning rate decay (default: 100)
- `--lr_decay`: Learning rate after decay (default: 0.005)
- `--batch_size`: Batch size (default: 1)
- `--n_examples`: Number of examples (default: 181)
- `--output_dir`: Output directory (default: `./output`)

**Neptune Logging:**
- `--use_neptune`: Enable Neptune logging
- `--project`: Neptune project name
- `--api_token`: Neptune API token (or set `NEPTUNE_API_TOKEN` environment variable)

For more detailed information, see [Models/README_unified_training.md](Models/README_unified_training.md).

### Output Structure

Results are saved in the following structure:
```
Models/output/
├── baseline_ecc_Left/
│   ├── baseline_ecc_Left_output_epoch25.pt
│   ├── baseline_ecc_Left_output_epoch50.pt
│   └── ...
├── baseline_ecc_Right/
├── transolver_optionA_ecc_Left/
└── ...
```

Each experiment creates a folder named `{model_type}_{prediction}_{hemisphere}`.

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
