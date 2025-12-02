#!/bin/bash

# Script to run all experiment combinations using Docker container
# This script runs all combinations of model types, predictions, and hemispheres in Docker

# Set base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Docker settings
DOCKER_IMAGE=${DOCKER_IMAGE:-"vnmd/deepretinotopy_1.0.18:latest"}  # Change to your Docker image name
CONTAINER_NAME="deepretinotopy_train"
USE_GPU=${USE_GPU:-"true"}  # Set to "false" to disable GPU

# Default parameters (optimized based on Transolver unstructured mesh experiments)
N_EPOCHS=500
LR_INIT=0.001
LR_DECAY_EPOCH=250
LR_DECAY=0.0001
INTERM_SAVE_EVERY=25
BATCH_SIZE=1
N_EXAMPLES=181
OUTPUT_DIR="./output"

# Optimizer and scheduler settings (matching Transolver experiments)
OPTIMIZER="AdamW"  # Options: Adam, AdamW
SCHEDULER="cosine"  # Options: step, cosine, onecycle
WEIGHT_DECAY=1e-5
MAX_GRAD_NORM=0.1

# Model architecture hyperparameters (matching Transolver_Irregular_Mesh)
N_LAYERS=8
N_HIDDEN=128
N_HEADS=8
SLICE_NUM=64
MLP_RATIO=1
DROPOUT=0.0
REF=8
UNIFIED_POS=0

# Neptune settings (optional)
USE_NEPTUNE=${USE_NEPTUNE:-"true"}  # Set to "true" to enable, "false" to disable
NEPTUNE_PROJECT="kjb961013/Retinosolver"  # Change to your Neptune project
NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZWNlMzk4Ny1hOTVlLTRjMzgtOWI4ZS1hY2FkYTY4MzNhYzMifQ=="  # Set your API token or use environment variable

# Model types to run
MODEL_TYPES=("transolver_optionC") # "transolver_optionB"

# Predictions to run
PREDICTIONS=("polarAngle" "eccentricity") # "eccentricity"

# Hemispheres to run
HEMISPHERES=("Left" "Right") # "Right" 

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker image exists, if not try to pull it
if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
    echo "Docker image '$DOCKER_IMAGE' not found. Attempting to pull..."
    if ! docker pull "$DOCKER_IMAGE"; then
        echo "Error: Failed to pull Docker image '$DOCKER_IMAGE'."
        echo "Please check the image name or set DOCKER_IMAGE environment variable."
        exit 1
    fi
fi

# Check if container is already running, or create/start it
CONTAINER_RUNNING=false
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    CONTAINER_RUNNING=true
    echo "Container '$CONTAINER_NAME' is already running. Using existing container."
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' exists but is stopped. Starting it..."
    docker start "$CONTAINER_NAME" > /dev/null 2>&1
    CONTAINER_RUNNING=true
else
    echo "Creating new container '$CONTAINER_NAME'..."
    DOCKER_CMD="docker run -d"
    if [ "$USE_GPU" = "true" ]; then
        DOCKER_CMD="$DOCKER_CMD --gpus all"
    fi
    DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"
    DOCKER_CMD="$DOCKER_CMD -v $PROJECT_ROOT:/workspace"
    DOCKER_CMD="$DOCKER_CMD -v $PROJECT_ROOT/Retinotopy/data:/workspace/Retinotopy/data"
    DOCKER_CMD="$DOCKER_CMD -w /workspace"
    DOCKER_CMD="$DOCKER_CMD $DOCKER_IMAGE"
    DOCKER_CMD="$DOCKER_CMD tail -f /dev/null"  # Keep container running
    
    eval "$DOCKER_CMD" > /dev/null 2>&1
    CONTAINER_RUNNING=true
fi

# Install required packages once (neptune and einops)
echo "Installing required packages (neptune, einops) in container..."
INSTALL_CMD="pip install --quiet --no-cache-dir neptune einops"
docker exec "$CONTAINER_NAME" bash -c "$INSTALL_CMD" > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "Warning: Failed to install packages. Continuing anyway..."
fi

# Function to run a single experiment
run_experiment() {
    local model_type=$1
    local prediction=$2
    local hemisphere=$3
    
    echo "=========================================="
    echo "Running experiment in Docker:"
    echo "  Model: $model_type"
    echo "  Prediction: $prediction"
    echo "  Hemisphere: $hemisphere"
    echo "=========================================="
    
    # Build Python command
    PYTHON_CMD="cd Models && python train_unified.py \
        --model_type $model_type \
        --prediction $prediction \
        --hemisphere $hemisphere \
        --n_epochs $N_EPOCHS \
        --lr_init $LR_INIT \
        --lr_decay_epoch $LR_DECAY_EPOCH \
        --lr_decay $LR_DECAY \
        --scheduler $SCHEDULER \
        --optimizer $OPTIMIZER \
        --weight_decay $WEIGHT_DECAY \
        --max_grad_norm $MAX_GRAD_NORM \
        --n_layers $N_LAYERS \
        --n_hidden $N_HIDDEN \
        --n_heads $N_HEADS \
        --slice_num $SLICE_NUM \
        --mlp_ratio $MLP_RATIO \
        --dropout $DROPOUT \
        --ref $REF \
        --unified_pos $UNIFIED_POS \
        --interm_save_every $INTERM_SAVE_EVERY \
        --batch_size $BATCH_SIZE \
        --n_examples $N_EXAMPLES \
        --output_dir $OUTPUT_DIR"
    
    # Add Neptune options if enabled
    if [ "$USE_NEPTUNE" = "true" ]; then
        PYTHON_CMD="$PYTHON_CMD --use_neptune --project $NEPTUNE_PROJECT"
        if [ ! -z "$NEPTUNE_API_TOKEN" ]; then
            PYTHON_CMD="$PYTHON_CMD --api_token $NEPTUNE_API_TOKEN"
        fi
    fi
    
    # Run the command in Docker container using exec
    docker exec "$CONTAINER_NAME" bash -c "$PYTHON_CMD" & > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ Experiment completed successfully"
    else
        echo "✗ Experiment failed"
        return 1
    fi
    
    echo ""
}

# Main execution
echo "=========================================="
echo "Starting all experiments in Docker..."
echo "  Docker Image: $DOCKER_IMAGE"
echo "  GPU: $USE_GPU"
echo "  Neptune: $USE_NEPTUNE"
echo "Total experiments: $((${#MODEL_TYPES[@]} * ${#PREDICTIONS[@]} * ${#HEMISPHERES[@]}))"
echo "=========================================="
echo ""

# Run all combinations
for model_type in "${MODEL_TYPES[@]}"; do
    for prediction in "${PREDICTIONS[@]}"; do
        for hemisphere in "${HEMISPHERES[@]}"; do
            run_experiment $model_type $prediction $hemisphere
        done
    done
done

echo "=========================================="
echo "All experiments completed!"
echo "Note: Container '$CONTAINER_NAME' is still running."
echo "To stop it, run: docker stop $CONTAINER_NAME"
echo "To remove it, run: docker rm -f $CONTAINER_NAME"
echo "=========================================="

