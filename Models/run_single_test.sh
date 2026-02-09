#!/bin/bash

# Script to run a single experiment for testing purposes
# This script runs one experiment configuration in Docker

# Set base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Docker settings
DOCKER_IMAGE=${DOCKER_IMAGE:-"vnmd/deepretinotopy_1.0.18:latest"}  # Change to your Docker image name
CONTAINER_NAME="deepretinotopy_train"
USE_GPU=${USE_GPU:-"true"}  # Set to "false" to disable GPU

# Test configuration - Modify these values for your test
MODEL_TYPE="transolver_optionC"
PREDICTION="eccentricity"
HEMISPHERE="Left"
SEED=0  # Set to empty string "" to use default seed=1

# Default parameters
N_EPOCHS=200
LR_INIT=0.01
LR_DECAY_EPOCH=100
LR_DECAY=0.005
INTERM_SAVE_EVERY=25
BATCH_SIZE=1
N_EXAMPLES=181
OUTPUT_DIR="./output_wandb"

# Parameters for transolver_optionC
N_EPOCHS_OPTIONC=500
LR_INIT_OPTIONC=0.001
LR_DECAY_EPOCH_OPTIONC=250
LR_DECAY_OPTIONC=0.0001
OPTIMIZER_OPTIONC="AdamW"
SCHEDULER_OPTIONC="cosine"
WEIGHT_DECAY_OPTIONC=1e-5
MAX_GRAD_NORM_OPTIONC=0.1
N_LAYERS_OPTIONC=8
N_HIDDEN_OPTIONC=128
N_HEADS_OPTIONC=8
SLICE_NUM_OPTIONC=64
MLP_RATIO_OPTIONC=1
DROPOUT_OPTIONC=0.0
REF_OPTIONC=8
UNIFIED_POS_OPTIONC=0

# Myelination setting
USE_MYELINATION="True"

# Wandb settings (optional)
USE_WANDB=${USE_WANDB:-"true"}  # Set to "true" to enable, "false" to disable
WANDB_PROJECT="retinotopic_mapping"  # Change to your Wandb project name
WANDB_ENTITY=${WANDB_ENTITY:-""}  # Optional: Set your Wandb entity/team name
WANDB_MODE="offline"  # Set to "offline" to run wandb in offline mode

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
    # Pass Wandb mode as environment variable for offline mode
    if [ "$USE_WANDB" = "true" ] && [ ! -z "$WANDB_MODE" ]; then
        DOCKER_CMD="$DOCKER_CMD -e WANDB_MODE=$WANDB_MODE"
    fi
    DOCKER_CMD="$DOCKER_CMD $DOCKER_IMAGE"
    DOCKER_CMD="$DOCKER_CMD tail -f /dev/null"  # Keep container running
    
    eval "$DOCKER_CMD" > /dev/null 2>&1
    CONTAINER_RUNNING=true
fi

# Install required packages once (wandb and einops)
echo "Installing required packages (wandb, einops) in container..."
INSTALL_CMD="pip install --quiet --no-cache-dir wandb einops"
docker exec "$CONTAINER_NAME" bash -c "$INSTALL_CMD" > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "Warning: Failed to install packages. Continuing anyway..."
fi

# Setup Wandb for offline mode
if [ "$USE_WANDB" = "true" ]; then
    if [ "$WANDB_MODE" = "offline" ]; then
        echo "Wandb will run in offline mode (WANDB_MODE=offline)"
    fi
fi

# Build Python command based on model type
echo "=========================================="
echo "Running test experiment in Docker:"
echo "  Model: $MODEL_TYPE"
echo "  Prediction: $PREDICTION"
echo "  Hemisphere: $HEMISPHERE"
if [ ! -z "$SEED" ]; then
    echo "  Seed: $SEED"
else
    echo "  Seed: (default seed=1)"
fi
echo "=========================================="

# Check if this is transolver_optionC and use appropriate parameters
if [ "$MODEL_TYPE" = "transolver_optionC" ]; then
    # Use transolver_optionC specific parameters
    USE_N_EPOCHS=$N_EPOCHS_OPTIONC
    USE_LR_INIT=$LR_INIT_OPTIONC
    USE_LR_DECAY_EPOCH=$LR_DECAY_EPOCH_OPTIONC
    USE_LR_DECAY=$LR_DECAY_OPTIONC
    
    # Build Python command with transolver_optionC parameters
    PYTHON_CMD="cd Models && python train_unified.py \
        --model_type $MODEL_TYPE \
        --prediction $PREDICTION \
        --hemisphere $HEMISPHERE \
        --n_epochs $USE_N_EPOCHS \
        --lr_init $USE_LR_INIT \
        --lr_decay_epoch $USE_LR_DECAY_EPOCH \
        --lr_decay $USE_LR_DECAY \
        --scheduler $SCHEDULER_OPTIONC \
        --optimizer $OPTIMIZER_OPTIONC \
        --weight_decay $WEIGHT_DECAY_OPTIONC \
        --max_grad_norm $MAX_GRAD_NORM_OPTIONC \
        --n_layers $N_LAYERS_OPTIONC \
        --n_hidden $N_HIDDEN_OPTIONC \
        --n_heads $N_HEADS_OPTIONC \
        --slice_num $SLICE_NUM_OPTIONC \
        --mlp_ratio $MLP_RATIO_OPTIONC \
        --dropout $DROPOUT_OPTIONC \
        --ref $REF_OPTIONC \
        --unified_pos $UNIFIED_POS_OPTIONC \
        --interm_save_every $INTERM_SAVE_EVERY \
        --batch_size $BATCH_SIZE \
        --n_examples $N_EXAMPLES \
        --output_dir $OUTPUT_DIR \
        --myelination $USE_MYELINATION"
    
    # Add seed parameter if provided
    if [ ! -z "$SEED" ]; then
        PYTHON_CMD="$PYTHON_CMD --seed $SEED"
    fi
else
    # Use default parameters for other model types
    USE_N_EPOCHS=$N_EPOCHS
    USE_LR_INIT=$LR_INIT
    USE_LR_DECAY_EPOCH=$LR_DECAY_EPOCH
    USE_LR_DECAY=$LR_DECAY
    
    # Build Python command with default parameters
    PYTHON_CMD="cd Models && python train_unified.py \
        --model_type $MODEL_TYPE \
        --prediction $PREDICTION \
        --hemisphere $HEMISPHERE \
        --n_epochs $USE_N_EPOCHS \
        --lr_init $USE_LR_INIT \
        --lr_decay_epoch $USE_LR_DECAY_EPOCH \
        --lr_decay $USE_LR_DECAY \
        --interm_save_every $INTERM_SAVE_EVERY \
        --batch_size $BATCH_SIZE \
        --n_examples $N_EXAMPLES \
        --output_dir $OUTPUT_DIR \
        --myelination $USE_MYELINATION"
    
    # Add seed parameter if provided
    if [ ! -z "$SEED" ]; then
        PYTHON_CMD="$PYTHON_CMD --seed $SEED"
    fi
fi

# Add Wandb options if enabled
if [ "$USE_WANDB" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --use_wandb --wandb_project $WANDB_PROJECT"
    if [ ! -z "$WANDB_ENTITY" ]; then
        PYTHON_CMD="$PYTHON_CMD --wandb_entity $WANDB_ENTITY"
    fi
fi

# Run the command in Docker container
echo ""
echo "Executing command..."
echo ""

# Set WANDB_MODE environment variable for offline mode
if [ "$USE_WANDB" = "true" ] && [ ! -z "$WANDB_MODE" ]; then
    docker exec -e WANDB_MODE="$WANDB_MODE" "$CONTAINER_NAME" bash -c "$PYTHON_CMD"
else
    docker exec "$CONTAINER_NAME" bash -c "$PYTHON_CMD"
fi

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Test experiment completed successfully!"
else
    echo "Test experiment failed with exit code: $EXIT_CODE"
fi
echo "Note: Container '$CONTAINER_NAME' is still running."
echo "To stop it, run: docker stop $CONTAINER_NAME"
echo "To remove it, run: docker rm -f $CONTAINER_NAME"
echo "=========================================="

exit $EXIT_CODE
