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

# Default parameters (for baseline and other standard models)
N_EPOCHS=200
LR_INIT=0.01
LR_DECAY_EPOCH=100
LR_DECAY=0.005
OPTIMIZER="Adam"  # Adam for baseline (matching deepRetinotopy_PA_LH.py)
SCHEDULER="step"  # Step scheduler for baseline (matching deepRetinotopy_PA_LH.py)
WEIGHT_DECAY=0  # No weight decay for baseline (matching deepRetinotopy_PA_LH.py)
INTERM_SAVE_EVERY=25
BATCH_SIZE=1
N_EXAMPLES=181
OUTPUT_DIR="./output_from_fs_curv"

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
WANDB_MODE="offline"  # Set to "offline" to run wandb in offline mode
# WANDB_API_KEY=${WANDB_API_KEY:-"c3775da3e8a4df79fc7cd2a26025023d557c14ca"}  # Wandb API key (get from https://wandb.ai/authorize) - COMMENTED OUT (using offline mode)
                                      # Or set environment variable: export WANDB_API_KEY=your_api_key

# Predicted map saving (optional)
SAVE_PREDICTED_MAP=${SAVE_PREDICTED_MAP:-"false"}  # Set to "true" to save predicted maps as numpy files

# R2 scaling in loss calculation (optional)
R2_SCALING=${R2_SCALING:-"true"}  # Set to "true" to use R2 scaling in loss, "false" to disable (default: true)

# Neptune settings (optional) - COMMENTED OUT
# USE_NEPTUNE=${USE_NEPTUNE:-"true"}  # Set to "true" to enable, "false" to disable
# NEPTUNE_PROJECT="kjb961013/Retinosolver"  # Change to your Neptune project
# NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZWNlMzk4Ny1hOTVlLTRjMzgtOWI4ZS1hY2FkYTY4MzNhYzMifQ=="  # Set your API token or use environment variable

# Predictions to run
PREDICTIONS=("pRFsize") #"polarAngle") # "pRFsize" "polarAngle" "pRFsize") # "eccentricity" "polarAngle" "pRFsize"

# Hemispheres to run
HEMISPHERES=("Right") # "Right") # "Right" 

# Model types to run
MODEL_TYPES=("transolver_optionC") # "transolver_optionA" "transolver_optionC") # "baseline" "transolver_optionA" "transolver_optionC")  #" "transolver_optionB") # ) # "transolver_optionA"  "transolver_optionB" "baseline"

# Myelination to run
USE_MYELINATION="False"

# Seeds to run (for subject splitting)
SEEDS=(2)  # Run experiments with seeds 0, 1, 2. Set to (0) to run only seed 0, or () to use default seed=1

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

# Function to check if GPU is available in container
check_gpu_available() {
    local container_name=$1
    # Try to run nvidia-smi in the container
    if docker exec "$container_name" nvidia-smi > /dev/null 2>&1; then
        return 0  # GPU is available
    else
        return 1  # GPU is not available
    fi
}

# Function to create a new container
create_container() {
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
    # Pass Wandb API key as environment variable if provided - COMMENTED OUT (using offline mode)
    # if [ ! -z "$WANDB_API_KEY" ]; then
    #     DOCKER_CMD="$DOCKER_CMD -e WANDB_API_KEY=$WANDB_API_KEY"
    # fi
    DOCKER_CMD="$DOCKER_CMD $DOCKER_IMAGE"
    DOCKER_CMD="$DOCKER_CMD tail -f /dev/null"  # Keep container running
    
    eval "$DOCKER_CMD" > /dev/null 2>&1
}

# Check if container is already running, or create/start it
CONTAINER_RUNNING=false
RECREATE_CONTAINER=false

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' is already running."
    # Check if GPU is needed and available
    if [ "$USE_GPU" = "true" ]; then
        echo "Checking GPU availability in container..."
        if check_gpu_available "$CONTAINER_NAME"; then
            echo "✓ GPU is available in container. Using existing container."
            CONTAINER_RUNNING=true
        else
            echo "✗ GPU is NOT available in container, but USE_GPU=true."
            echo "  Recreating container with GPU support..."
            docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1
            RECREATE_CONTAINER=true
        fi
    else
        echo "GPU not required. Using existing container."
        CONTAINER_RUNNING=true
    fi
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' exists but is stopped."
    # Check if GPU is needed
    if [ "$USE_GPU" = "true" ]; then
        echo "Starting container and checking GPU availability..."
        docker start "$CONTAINER_NAME" > /dev/null 2>&1
        if check_gpu_available "$CONTAINER_NAME"; then
            echo "✓ GPU is available in container. Using existing container."
            CONTAINER_RUNNING=true
        else
            echo "✗ GPU is NOT available in container, but USE_GPU=true."
            echo "  Recreating container with GPU support..."
            docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1
            RECREATE_CONTAINER=true
        fi
    else
        echo "Starting container (GPU not required)..."
        docker start "$CONTAINER_NAME" > /dev/null 2>&1
        CONTAINER_RUNNING=true
    fi
else
    RECREATE_CONTAINER=true
fi

# Create container if needed
if [ "$RECREATE_CONTAINER" = "true" ]; then
    create_container
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

# Setup Wandb for offline mode
if [ "$USE_WANDB" = "true" ]; then
    if [ "$WANDB_MODE" = "offline" ]; then
        echo "Wandb will run in offline mode (WANDB_MODE=offline)"
    fi
fi

# Setup Wandb authentication if API key is provided - COMMENTED OUT (using offline mode)
# if [ "$USE_WANDB" = "true" ]; then
#     if [ ! -z "$WANDB_API_KEY" ]; then
#         echo "Setting up Wandb authentication in container..."
#         docker exec "$CONTAINER_NAME" bash -c "wandb login $WANDB_API_KEY" > /dev/null 2>&1
#         if [ $? -eq 0 ]; then
#             echo "✓ Wandb authentication successful"
#         else
#             echo "Warning: Wandb authentication failed. You may need to login manually."
#             echo "  Run: docker exec -it $CONTAINER_NAME wandb login"
#         fi
#     else
#         echo "Warning: WANDB_API_KEY not set. Wandb may prompt for login."
#         echo "  Set it via: export WANDB_API_KEY=your_api_key"
#         echo "  Or add it to the script: WANDB_API_KEY=\"your_api_key\""
#         echo "  Get your API key from: https://wandb.ai/authorize"
#     fi
# fi

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
    local seed=$4  # Seed for subject splitting (optional, can be empty)
    local single_mode=${5:-false}  # If true, run directly without background and show output
    
    # Wait for an available slot only if not in single mode
    if [ "$single_mode" != "true" ]; then
        wait_for_slot
    fi
    
    echo "=========================================="
    echo "Running experiment in Docker:"
    echo "  Model: $model_type"
    echo "  Prediction: $prediction"
    echo "  Hemisphere: $hemisphere"
    if [ ! -z "$seed" ]; then
        echo "  Seed: $seed"
    fi
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
            --myelination $USE_MYELINATION \
            --r2_scaling $R2_SCALING \
            --use_freesurfer_curv True"
        
        # Add seed parameter if provided
        if [ ! -z "$seed" ]; then
            PYTHON_CMD="$PYTHON_CMD --seed $seed"
        fi
        
        # Add predicted map saving if enabled
        if [ "$SAVE_PREDICTED_MAP" = "true" ]; then
            PYTHON_CMD="$PYTHON_CMD --save_predicted_map True"
        fi
    else
        # Use default parameters for other model types (baseline, transolver_optionA, etc.)
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
            --scheduler $SCHEDULER \
            --optimizer $OPTIMIZER \
            --weight_decay $WEIGHT_DECAY \
            --interm_save_every $INTERM_SAVE_EVERY \
            --batch_size $BATCH_SIZE \
            --n_examples $N_EXAMPLES \
            --output_dir $OUTPUT_DIR \
            --myelination $USE_MYELINATION \
            --r2_scaling $R2_SCALING \
            --use_freesurfer_curv True"
        
        # Add seed parameter if provided
        if [ ! -z "$seed" ]; then
            PYTHON_CMD="$PYTHON_CMD --seed $seed"
        fi
        
        # Add predicted map saving if enabled
        if [ "$SAVE_PREDICTED_MAP" = "true" ]; then
            PYTHON_CMD="$PYTHON_CMD --save_predicted_map True"
        fi
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
    
    # Run the command based on single mode or parallel mode
    if [ "$single_mode" = "true" ]; then
        # Single mode: run directly and show output immediately
        echo "Running experiment directly (single mode, output shown below)..."
        echo ""
        
        if [ "$USE_WANDB" = "true" ] && [ ! -z "$WANDB_MODE" ]; then
            docker exec -e WANDB_MODE="$WANDB_MODE" "$CONTAINER_NAME" bash -c "$PYTHON_CMD"
        else
            docker exec "$CONTAINER_NAME" bash -c "$PYTHON_CMD"
        fi
        
        echo ""
        echo "Experiment completed!"
        echo ""
    else
        # Parallel mode: run in background and save to log file
        # Create log file name for this experiment
        local log_filename="${model_type}_${prediction}_${hemisphere}"
        if [ ! -z "$seed" ]; then
            log_filename="${log_filename}_seed${seed}"
        fi
        log_filename="${log_filename}.log"
        local log_file="$PROJECT_ROOT/logs/$log_filename"
        
        # Create logs directory if it doesn't exist
        mkdir -p "$PROJECT_ROOT/logs"
        
        # Run the command in Docker container using exec in background
        # Set WANDB_MODE environment variable for offline mode
        # Save output to log file
        local job_pid
        if [ "$USE_WANDB" = "true" ] && [ ! -z "$WANDB_MODE" ]; then
            docker exec -e WANDB_MODE="$WANDB_MODE" "$CONTAINER_NAME" bash -c "$PYTHON_CMD" > "$log_file" 2>&1 &
            job_pid=$!
        else
            docker exec "$CONTAINER_NAME" bash -c "$PYTHON_CMD" > "$log_file" 2>&1 &
            job_pid=$!
        fi
        
        echo "  Log file: $log_file"
        echo "  View log: tail -f $log_file"
        
        # Add PID to running jobs array
        running_jobs+=("$job_pid")
        
        echo "Started experiment (PID: $job_pid)"
        echo ""
    fi
}

# Main execution
echo "=========================================="
echo "Starting all experiments in Docker..."
echo "  Docker Image: $DOCKER_IMAGE"
echo "  GPU: $USE_GPU"
echo "  Wandb: $USE_WANDB"
echo "  R2 Scaling: $R2_SCALING"
    if [ "$USE_WANDB" = "true" ]; then
        echo "  Wandb Project: $WANDB_PROJECT"
        if [ ! -z "$WANDB_ENTITY" ]; then
            echo "  Wandb Entity: $WANDB_ENTITY"
        fi
        if [ ! -z "$WANDB_MODE" ]; then
            echo "  Wandb Mode: $WANDB_MODE"
        fi
        # if [ ! -z "$WANDB_API_KEY" ]; then
        #     echo "  Wandb API Key: *** (set)"
        # else
        #     echo "  Wandb API Key: Not set (will prompt for login if needed)"
        # fi
    fi
# echo "  Neptune: $USE_NEPTUNE"  # COMMENTED OUT
# Calculate total experiments (including seeds if specified)
if [ ${#SEEDS[@]} -gt 0 ]; then
    TOTAL_EXPERIMENTS=$((${#MODEL_TYPES[@]} * ${#PREDICTIONS[@]} * ${#HEMISPHERES[@]} * ${#SEEDS[@]}))
    echo "Seeds: ${SEEDS[@]}"
else
    TOTAL_EXPERIMENTS=$((${#MODEL_TYPES[@]} * ${#PREDICTIONS[@]} * ${#HEMISPHERES[@]}))
    echo "Seeds: (default seed=1)"
fi
echo "Total experiments: $TOTAL_EXPERIMENTS"
if [ $TOTAL_EXPERIMENTS -eq 1 ]; then
    echo "Single experiment mode: output will be shown directly (no log file)"
else
    echo "Max concurrent jobs: $MAX_CONCURRENT_JOBS"
fi
echo "=========================================="
echo ""

# Check if single experiment mode
if [ $TOTAL_EXPERIMENTS -eq 1 ]; then
    # Single experiment mode: run directly without background
    for prediction in "${PREDICTIONS[@]}"; do
        for hemisphere in "${HEMISPHERES[@]}"; do
            for model_type in "${MODEL_TYPES[@]}"; do
                if [ ${#SEEDS[@]} -gt 0 ]; then
                    # Run with each seed
                    for seed in "${SEEDS[@]}"; do
                        run_experiment $model_type $prediction $hemisphere $seed "true"
                    done
                else
                    # Run without seed (uses default seed=1)
                    run_experiment $model_type $prediction $hemisphere "" "true"
                fi
            done
        done
    done
    
    echo ""
    echo "=========================================="
    echo "Experiment completed!"
    echo ""
    echo "Note: Container '$CONTAINER_NAME' is still running."
    echo "To stop it, run: docker stop $CONTAINER_NAME"
    echo "To remove it, run: docker rm -f $CONTAINER_NAME"
    echo "=========================================="
else
    # Multiple experiments mode: run in parallel with log files
    # Run all combinations (PREDICTIONS -> HEMISPHERES -> MODEL_TYPES -> SEEDS)
    for prediction in "${PREDICTIONS[@]}"; do
        for hemisphere in "${HEMISPHERES[@]}"; do
            for model_type in "${MODEL_TYPES[@]}"; do
                if [ ${#SEEDS[@]} -gt 0 ]; then
                    # Run with each seed
                    for seed in "${SEEDS[@]}"; do
                        run_experiment $model_type $prediction $hemisphere $seed "false"
                        sleep 5
                    done
                else
                    # Run without seed (uses default seed=1)
                    run_experiment $model_type $prediction $hemisphere "" "false"
                    sleep 5
                fi
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
    echo ""
    echo "Log files are saved in: $PROJECT_ROOT/logs/"
    echo "To view all log files: ls -lh $PROJECT_ROOT/logs/"
    echo "To view a specific log: tail -f $PROJECT_ROOT/logs/<log_file_name>"
    echo ""
    echo "Note: Container '$CONTAINER_NAME' is still running."
    echo "To stop it, run: docker stop $CONTAINER_NAME"
    echo "To remove it, run: docker rm -f $CONTAINER_NAME"
    echo "=========================================="
fi

