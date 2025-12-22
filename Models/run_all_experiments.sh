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

# Default parameters
N_EPOCHS=200
LR_INIT=0.01
LR_DECAY_EPOCH=100
LR_DECAY=0.005
INTERM_SAVE_EVERY=25
BATCH_SIZE=1
N_EXAMPLES=181
OUTPUT_DIR="./output_wandb"

# Parameters for transolver_optionC (from run_transolver_optionC.sh)
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

# Wandb settings (optional)
USE_WANDB=${USE_WANDB:-"true"}  # Set to "true" to enable, "false" to disable
WANDB_PROJECT="retinotopic_mapping"  # Change to your Wandb project name
WANDB_ENTITY=${WANDB_ENTITY:-""}  # Optional: Set your Wandb entity/team name
WANDB_API_KEY=${WANDB_API_KEY:-"c3775da3e8a4df79fc7cd2a26025023d557c14ca"}  # Wandb API key (get from https://wandb.ai/authorize)
                                      # Or set environment variable: export WANDB_API_KEY=your_api_key

# Neptune settings (optional) - COMMENTED OUT
# USE_NEPTUNE=${USE_NEPTUNE:-"true"}  # Set to "true" to enable, "false" to disable
# NEPTUNE_PROJECT="kjb961013/Retinosolver"  # Change to your Neptune project
# NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZWNlMzk4Ny1hOTVlLTRjMzgtOWI4ZS1hY2FkYTY4MzNhYzMifQ=="  # Set your API token or use environment variable

# Predictions to run
PREDICTIONS=("eccentricity" "polarAngle" "pRFsize") # "eccentricity" "polarAngle" "pRFsize"

# Hemispheres to run
HEMISPHERES=("Left" "Right") # "Right" 

# Model types to run
MODEL_TYPES=("transolver_optionC") #"baseline" "transolver_optionA" "transolver_optionB") # ) # "transolver_optionA"  "transolver_optionB" "baseline"

# Myelination to run
USE_MYELINATION="True"

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
    # Pass Wandb API key as environment variable if provided
    if [ ! -z "$WANDB_API_KEY" ]; then
        DOCKER_CMD="$DOCKER_CMD -e WANDB_API_KEY=$WANDB_API_KEY"
    fi
    DOCKER_CMD="$DOCKER_CMD $DOCKER_IMAGE"
    DOCKER_CMD="$DOCKER_CMD tail -f /dev/null"  # Keep container running
    
    eval "$DOCKER_CMD" > /dev/null 2>&1
    CONTAINER_RUNNING=true
fi

# Install required packages once (wandb and einops)
echo "Installing required packages (wandb, einops) in container..."
INSTALL_CMD="pip install --quiet --no-cache-dir wandb einops"
# Install required packages once (neptune and einops) - COMMENTED OUT
# echo "Installing required packages (neptune, einops) in container..."
# INSTALL_CMD="pip install --quiet --no-cache-dir neptune einops wandb"
docker exec "$CONTAINER_NAME" bash -c "$INSTALL_CMD" > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "Warning: Failed to install packages. Continuing anyway..."
fi

# Setup Wandb authentication if API key is provided
if [ "$USE_WANDB" = "true" ]; then
    if [ ! -z "$WANDB_API_KEY" ]; then
        echo "Setting up Wandb authentication in container..."
        docker exec "$CONTAINER_NAME" bash -c "wandb login $WANDB_API_KEY" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ Wandb authentication successful"
        else
            echo "Warning: Wandb authentication failed. You may need to login manually."
            echo "  Run: docker exec -it $CONTAINER_NAME wandb login"
        fi
    else
        echo "Warning: WANDB_API_KEY not set. Wandb may prompt for login."
        echo "  Set it via: export WANDB_API_KEY=your_api_key"
        echo "  Or add it to the script: WANDB_API_KEY=\"your_api_key\""
        echo "  Get your API key from: https://wandb.ai/authorize"
    fi
fi

# Maximum number of concurrent jobs
MAX_CONCURRENT_JOBS=12

# Array to track running job PIDs
declare -a running_jobs=()

# Function to wait for a job slot to become available
wait_for_slot() {
    while [ ${#running_jobs[@]} -ge $MAX_CONCURRENT_JOBS ]; do
        # Check which jobs are still running
        local new_jobs=()
        for pid in "${running_jobs[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                # Job is still running
                new_jobs+=("$pid")
            fi
        done
        running_jobs=("${new_jobs[@]}")
        
        # If still at max capacity, wait a bit
        if [ ${#running_jobs[@]} -ge $MAX_CONCURRENT_JOBS ]; then
            sleep 2
        fi
    done
}

# Function to run a single experiment
run_experiment() {
    local model_type=$1
    local prediction=$2
    local hemisphere=$3
    
    # Wait for an available slot
    wait_for_slot
    
    echo "=========================================="
    echo "Running experiment in Docker:"
    echo "  Model: $model_type"
    echo "  Prediction: $prediction"
    echo "  Hemisphere: $hemisphere"
    echo "  Running jobs: ${#running_jobs[@]}/$MAX_CONCURRENT_JOBS"
    echo "=========================================="
    
    # Check if this is transolver_optionC and use appropriate parameters
    if [ "$model_type" = "transolver_optionC" ]; then
        # Use transolver_optionC specific parameters
        USE_N_EPOCHS=$N_EPOCHS_OPTIONC
        USE_LR_INIT=$LR_INIT_OPTIONC
        USE_LR_DECAY_EPOCH=$LR_DECAY_EPOCH_OPTIONC
        USE_LR_DECAY=$LR_DECAY_OPTIONC
        
        # Build Python command with transolver_optionC parameters
        PYTHON_CMD="cd Models && python train_unified.py \
            --model_type $model_type \
            --prediction $prediction \
            --hemisphere $hemisphere \
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
    else
        # Use default parameters for other model types
        USE_N_EPOCHS=$N_EPOCHS
        USE_LR_INIT=$LR_INIT
        USE_LR_DECAY_EPOCH=$LR_DECAY_EPOCH
        USE_LR_DECAY=$LR_DECAY
        
        # Build Python command with default parameters
        PYTHON_CMD="cd Models && python train_unified.py \
            --model_type $model_type \
            --prediction $prediction \
            --hemisphere $hemisphere \
            --n_epochs $USE_N_EPOCHS \
            --lr_init $USE_LR_INIT \
            --lr_decay_epoch $USE_LR_DECAY_EPOCH \
            --lr_decay $USE_LR_DECAY \
            --interm_save_every $INTERM_SAVE_EVERY \
            --batch_size $BATCH_SIZE \
            --n_examples $N_EXAMPLES \
            --output_dir $OUTPUT_DIR \
            --myelination $USE_MYELINATION"
    fi
    
    # Add Wandb options if enabled
    if [ "$USE_WANDB" = "true" ]; then
        PYTHON_CMD="$PYTHON_CMD --use_wandb --wandb_project $WANDB_PROJECT"
        if [ ! -z "$WANDB_ENTITY" ]; then
            PYTHON_CMD="$PYTHON_CMD --wandb_entity $WANDB_ENTITY"
        fi
    fi
    
    # Add Neptune options if enabled - COMMENTED OUT
    # if [ "$USE_NEPTUNE" = "true" ]; then
    #     PYTHON_CMD="$PYTHON_CMD --use_neptune --project $NEPTUNE_PROJECT"
    #     if [ ! -z "$NEPTUNE_API_TOKEN" ]; then
    #         PYTHON_CMD="$PYTHON_CMD --api_token $NEPTUNE_API_TOKEN"
    #     fi
    # fi
    
    # Run the command in Docker container using exec in background
    # Pass WANDB_API_KEY as environment variable if provided
    local job_pid
    if [ ! -z "$WANDB_API_KEY" ]; then
        docker exec -e WANDB_API_KEY="$WANDB_API_KEY" "$CONTAINER_NAME" bash -c "$PYTHON_CMD" > /dev/null 2>&1 &
        job_pid=$!
    else
        docker exec "$CONTAINER_NAME" bash -c "$PYTHON_CMD" > /dev/null 2>&1 &
        job_pid=$!
    fi
    
    # Add PID to running jobs array
    running_jobs+=("$job_pid")
    
    echo "Started experiment (PID: $job_pid)"
    echo ""
}

# Main execution
echo "=========================================="
echo "Starting all experiments in Docker..."
echo "  Docker Image: $DOCKER_IMAGE"
echo "  GPU: $USE_GPU"
echo "  Wandb: $USE_WANDB"
if [ "$USE_WANDB" = "true" ]; then
    echo "  Wandb Project: $WANDB_PROJECT"
    if [ ! -z "$WANDB_ENTITY" ]; then
        echo "  Wandb Entity: $WANDB_ENTITY"
    fi
    if [ ! -z "$WANDB_API_KEY" ]; then
        echo "  Wandb API Key: *** (set)"
    else
        echo "  Wandb API Key: Not set (will prompt for login if needed)"
    fi
fi
# echo "  Neptune: $USE_NEPTUNE"  # COMMENTED OUT
echo "Total experiments: $((${#MODEL_TYPES[@]} * ${#PREDICTIONS[@]} * ${#HEMISPHERES[@]}))"
echo "Max concurrent jobs: $MAX_CONCURRENT_JOBS"
echo "=========================================="
echo ""

# Run all combinations (PREDICTIONS -> HEMISPHERES -> MODEL_TYPES)
for prediction in "${PREDICTIONS[@]}"; do
    for hemisphere in "${HEMISPHERES[@]}"; do
        for model_type in "${MODEL_TYPES[@]}"; do
            run_experiment $model_type $prediction $hemisphere
            sleep 5
        done
    done
done

# Wait for all remaining jobs to complete
echo "Waiting for all experiments to complete..."
while [ ${#running_jobs[@]} -gt 0 ]; do
    new_jobs=()
    for pid in "${running_jobs[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            # Job is still running
            new_jobs+=("$pid")
        else
            # Job completed
            wait "$pid" 2>/dev/null
            echo "Experiment (PID: $pid) completed"
        fi
    done
    running_jobs=("${new_jobs[@]}")
    
    if [ ${#running_jobs[@]} -gt 0 ]; then
        echo "Waiting for ${#running_jobs[@]} experiment(s) to complete..."
        sleep 5
    fi
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Note: Container '$CONTAINER_NAME' is still running."
echo "To stop it, run: docker stop $CONTAINER_NAME"
echo "To remove it, run: docker rm -f $CONTAINER_NAME"
echo "=========================================="

