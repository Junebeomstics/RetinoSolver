#!/bin/bash

# Script to load checkpoint and run inference only on test set
# This script loads a checkpoint and runs test set evaluation only (no training)

# Set base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Docker settings
DOCKER_IMAGE=${DOCKER_IMAGE:-"vnmd/deepretinotopy_1.0.18:latest"}  # Change to your Docker image name
CONTAINER_NAME="deepretinotopy_test"
USE_GPU=${USE_GPU:-"true"}  # Set to "false" to disable GPU

# Default parameters
BATCH_SIZE=1
N_EXAMPLES=181
OUTPUT_DIR="./output_wandb"
MYELINATION=${MYELINATION:-"True"}  # Set to "True" or "False"

# Default checkpoint base directory (Docker environment)
# Normalize OUTPUT_DIR (remove ./ prefix if present)
OUTPUT_DIR_NORMALIZED="${OUTPUT_DIR#./}"
DEFAULT_CHECKPOINT_BASE="/mnt/scratch/junb/deepRetinotopy/Models/$OUTPUT_DIR_NORMALIZED"
CONTAINER_CHECKPOINT_BASE="/workspace/Models/$OUTPUT_DIR_NORMALIZED"


# Model configuration (required if not in checkpoint metadata)
# Define all combinations to test
PREDICTIONS=("eccentricity" "polarAngle" "pRFsize")  # eccentricity, polarAngle, pRFsize
HEMISPHERES=("Left" "Right")  # Left, Right
MODEL_TYPES=("baseline" "transolver_optionA" "transolver_optionB")  # baseline, transolver_optionA, transolver_optionB, transolver_optionC

# Default values (used if not running in loop mode)
PREDICTION=${PREDICTION:-"eccentricity"}
HEMISPHERE=${HEMISPHERE:-"Left"}
MODEL_TYPE=${MODEL_TYPE:-"transolver_optionA"}

# Flag to enable/disable loop mode (set RUN_ALL=true to run all combinations)
RUN_ALL=${RUN_ALL:-"false"}

# Function to determine prediction suffix
get_pred_suffix() {
    local pred=$1
    case "$pred" in
        eccentricity)
            echo "ecc"
            ;;
        polarAngle)
            echo "PA"
            ;;
        pRFsize)
            echo "size"
            ;;
        *)
            echo "Unknown PREDICTION: $pred" >&2
            return 1
            ;;
    esac
}

# Function to get checkpoint path for given parameters
get_checkpoint_path() {
    local pred=$1
    local hemi=$2
    local model=$3
    local myel=$4
    local pred_suffix=$(get_pred_suffix "$pred")
    
    if [[ "$myel" == "False" ]]; then
        echo "${DEFAULT_CHECKPOINT_BASE}/${pred}_${hemi}_${model}_noMyelin/${pred_suffix}_${hemi}_${model}_noMyelin_best_model_epoch*.pt"
    else
        echo "${DEFAULT_CHECKPOINT_BASE}/${pred}_${hemi}_${model}/${pred_suffix}_${hemi}_${model}_best_model_epoch*.pt"
    fi
}

# Checkpoint path (required) - will be set per iteration if RUN_ALL=true
CHECKPOINT_PATH=${CHECKPOINT_PATH:-""}


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --prediction)
            PREDICTION="$2"
            shift 2
            ;;
        --hemisphere)
            HEMISPHERE="$2"
            shift 2
            ;;
        --n_examples)
            N_EXAMPLES="$2"
            shift 2
            ;;
        --myelination)
            MYELINATION="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --docker_image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --use_gpu)
            USE_GPU="$2"
            shift 2
            ;;
        --run_all)
            RUN_ALL="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint_path PATH    Path to checkpoint file (.pt) [REQUIRED if not using --run_all]"
            echo "  --model_type TYPE         Model type: baseline, transolver_optionA, transolver_optionB, transolver_optionC"
            echo "  --prediction TYPE         Prediction type: eccentricity, polarAngle, pRFsize"
            echo "  --hemisphere HEMI         Hemisphere: Left, Right"
            echo "  --n_examples NUM          Number of examples (default: 181)"
            echo "  --myelination BOOL        Use myelination: True, False (default: True)"
            echo "  --output_dir DIR          Output directory (default: ./output_wandb)"
            echo "  --docker_image IMAGE      Docker image name (default: vnmd/deepretinotopy_1.0.18:latest)"
            echo "  --use_gpu BOOL            Use GPU: true, false (default: true)"
            echo "  --run_all                Run all combinations of PREDICTION, HEMISPHERE, MODEL_TYPE"
            echo ""
            echo "Environment variables can also be used:"
            echo "  CHECKPOINT_PATH, MODEL_TYPE, PREDICTION, HEMISPHERE, RUN_ALL, etc."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Update checkpoint base paths after parsing arguments (in case OUTPUT_DIR was changed)
OUTPUT_DIR_NORMALIZED="${OUTPUT_DIR#./}"
DEFAULT_CHECKPOINT_BASE="/mnt/scratch/junb/deepRetinotopy/Models/$OUTPUT_DIR_NORMALIZED"
CONTAINER_CHECKPOINT_BASE="/workspace/Models/$OUTPUT_DIR_NORMALIZED"

# Function to run test inference for a single configuration
run_single_test() {
    local pred=$1
    local hemi=$2
    local model=$3
    local checkpoint_path=$4
    
    echo ""
    echo "=========================================="
    echo "Running test set inference:"
    echo "  Checkpoint: $checkpoint_path"
    echo "  Model: $model"
    echo "  Prediction: $pred"
    echo "  Hemisphere: $hemi"
    echo "  Myelination: $MYELINATION"
    echo "=========================================="
    
    # Convert checkpoint path to container path
    local container_checkpoint_path
    if [[ "$checkpoint_path" == "$DEFAULT_CHECKPOINT_BASE"* ]]; then
        # Convert default base to container base
        container_checkpoint_path="${checkpoint_path/$DEFAULT_CHECKPOINT_BASE/$CONTAINER_CHECKPOINT_BASE}"
    elif [[ "$checkpoint_path" == "$PROJECT_ROOT"* ]]; then
        # Convert project root path to container path
        container_checkpoint_path="/workspace${checkpoint_path#$PROJECT_ROOT}"
    elif [[ "$checkpoint_path" == /* ]]; then
        # Absolute path - assume it's already a container path or use as is
        container_checkpoint_path="$checkpoint_path"
    else
        # Relative path - assume relative to /workspace
        container_checkpoint_path="/workspace/$checkpoint_path"
    fi
    
    echo "  Host checkpoint path: $checkpoint_path"
    echo "  Container checkpoint path: $container_checkpoint_path"
    
    # Build Python command
    local python_cmd="cd Models && python train_unified.py \
        --checkpoint_path $container_checkpoint_path \
        --run_test True \
        --model_type $model \
        --prediction $pred \
        --hemisphere $hemi \
        --n_examples $N_EXAMPLES \
        --myelination $MYELINATION \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE"
    
    # Run the command in Docker container
    echo "Executing test inference..."
    docker exec "$CONTAINER_NAME" bash -c "$python_cmd"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✓ Test inference completed successfully"
        echo "  Model: $model, Prediction: $pred, Hemisphere: $hemi"
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "✗ Test inference failed"
        echo "  Model: $model, Prediction: $pred, Hemisphere: $hemi"
        echo "=========================================="
    fi
    
    return $exit_code
}

# Check if checkpoint path is provided (only if not running all combinations)
if [[ "$RUN_ALL" != "true" ]]; then
    if [ -z "$CHECKPOINT_PATH" ]; then
        # Try to generate default checkpoint path
        CHECKPOINT_PATH=$(get_checkpoint_path "$PREDICTION" "$HEMISPHERE" "$MODEL_TYPE" "$MYELINATION")
    fi
    
    # Prepend default checkpoint base if checkpoint path is relative or just filename
    if [[ "$CHECKPOINT_PATH" != /* ]]; then
        # Relative path or filename - prepend default base directory
        CHECKPOINT_PATH="$DEFAULT_CHECKPOINT_BASE/$CHECKPOINT_PATH"
        echo "Using default checkpoint base directory: $CHECKPOINT_PATH"
    fi
    
    # Check if checkpoint file exists (if it's an absolute path or relative to project root)
    if [ ! -f "$CHECKPOINT_PATH" ] && [ ! -f "$PROJECT_ROOT/$CHECKPOINT_PATH" ]; then
        echo "Warning: Checkpoint file not found at: $CHECKPOINT_PATH"
        echo "Will try to load from Docker container path"
    fi
fi

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

# Run test inference
if [[ "$RUN_ALL" == "true" ]]; then
    # Run all combinations
    echo "=========================================="
    echo "Running test inference for all combinations"
    echo "  Predictions: ${PREDICTIONS[@]}"
    echo "  Hemispheres: ${HEMISPHERES[@]}"
    echo "  Model Types: ${MODEL_TYPES[@]}"
    echo "  Myelination: $MYELINATION"
    echo "=========================================="
    
    total_combinations=$((${#PREDICTIONS[@]} * ${#HEMISPHERES[@]} * ${#MODEL_TYPES[@]}))
    current=0
    success_count=0
    fail_count=0
    
    for PREDICTION in "${PREDICTIONS[@]}"; do
        for HEMISPHERE in "${HEMISPHERES[@]}"; do
            for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
                current=$((current + 1))
                echo ""
                echo "[$current/$total_combinations] Processing: $MODEL_TYPE - $PREDICTION - $HEMISPHERE"
                
                # Get checkpoint path for this combination
                CHECKPOINT_PATH=$(get_checkpoint_path "$PREDICTION" "$HEMISPHERE" "$MODEL_TYPE" "$MYELINATION")
                
                # Check if checkpoint file exists (use first match if wildcard)
                if [[ "$CHECKPOINT_PATH" == *"*"* ]]; then
                    # Expand wildcard and take first match
                    CHECKPOINT_PATH=$(ls $CHECKPOINT_PATH 2>/dev/null | head -1)
                fi
                
                echo "  Checkpoint path: $CHECKPOINT_PATH"
                
                if [ -z "$CHECKPOINT_PATH" ] || [ ! -f "$CHECKPOINT_PATH" ]; then
                    echo "  ⚠ Skipping: Checkpoint not found for $MODEL_TYPE - $PREDICTION - $HEMISPHERE"
                    echo "    Expected path: $(get_checkpoint_path "$PREDICTION" "$HEMISPHERE" "$MODEL_TYPE" "$MYELINATION")"
                    fail_count=$((fail_count + 1))
                    continue
                fi
                
                # Run test inference
                if run_single_test "$PREDICTION" "$HEMISPHERE" "$MODEL_TYPE" "$CHECKPOINT_PATH"; then
                    success_count=$((success_count + 1))
                else
                    fail_count=$((fail_count + 1))
                fi
            done
        done
    done
    
    echo ""
    echo "=========================================="
    echo "Summary:"
    echo "  Total combinations: $total_combinations"
    echo "  Successful: $success_count"
    echo "  Failed/Skipped: $fail_count"
    echo "=========================================="
    
    if [ $fail_count -gt 0 ]; then
        exit 1
    fi
else
    # Run single configuration
    run_single_test "$PREDICTION" "$HEMISPHERE" "$MODEL_TYPE" "$CHECKPOINT_PATH"
    
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

echo ""
echo "Note: Container '$CONTAINER_NAME' is still running."
echo "To stop it, run: docker stop $CONTAINER_NAME"
echo "To remove it, run: docker rm -f $CONTAINER_NAME"

