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
OUTPUT_DIR="./output"
MYELINATION=${MYELINATION:-"True"}  # Set to "True" or "False"

# Default checkpoint base directory (Docker environment)
DEFAULT_CHECKPOINT_BASE="/mnt/scratch/junb/deepRetinotopy/Models/output"
CONTAINER_CHECKPOINT_BASE="/workspace/Models/output"

# Checkpoint path (required)
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"RET-45/transolver_optionA_ecc_Left_best_model_epoch115.pt"}

# Model configuration (required if not in checkpoint metadata)
MODEL_TYPE=${MODEL_TYPE:-"transolver_optionA"}  # baseline, transolver_optionA, transolver_optionB, transolver_optionC
PREDICTION=${PREDICTION:-"eccentricity"}  # eccentricity, polarAngle
HEMISPHERE=${HEMISPHERE:-"Left"}  # Left, Right

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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint_path PATH    Path to checkpoint file (.pt) [REQUIRED]"
            echo "  --model_type TYPE         Model type: baseline, transolver_optionA, transolver_optionB, transolver_optionC"
            echo "  --prediction TYPE         Prediction type: eccentricity, polarAngle"
            echo "  --hemisphere HEMI         Hemisphere: Left, Right"
            echo "  --n_examples NUM          Number of examples (default: 181)"
            echo "  --myelination BOOL        Use myelination: True, False (default: True)"
            echo "  --output_dir DIR          Output directory (default: ./output)"
            echo "  --docker_image IMAGE      Docker image name (default: vnmd/deepretinotopy_1.0.18:latest)"
            echo "  --use_gpu BOOL            Use GPU: true, false (default: true)"
            echo ""
            echo "Environment variables can also be used:"
            echo "  CHECKPOINT_PATH, MODEL_TYPE, PREDICTION, HEMISPHERE, etc."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if checkpoint path is provided
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: --checkpoint_path is required"
    echo "Use --help for usage information"
    exit 1
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

# Build Python command for test-only inference
echo "=========================================="
echo "Running test set inference:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Model: $MODEL_TYPE"
echo "  Prediction: $PREDICTION"
echo "  Hemisphere: $HEMISPHERE"
echo "=========================================="

# Convert checkpoint path to container path
# Check if it's the default checkpoint base path
if [[ "$CHECKPOINT_PATH" == "$DEFAULT_CHECKPOINT_BASE"* ]]; then
    # Convert default base to container base
    CONTAINER_CHECKPOINT_PATH="${CHECKPOINT_PATH/$DEFAULT_CHECKPOINT_BASE/$CONTAINER_CHECKPOINT_BASE}"
elif [[ "$CHECKPOINT_PATH" == "$PROJECT_ROOT"* ]]; then
    # Convert project root path to container path
    CONTAINER_CHECKPOINT_PATH="/workspace${CHECKPOINT_PATH#$PROJECT_ROOT}"
elif [[ "$CHECKPOINT_PATH" == /* ]]; then
    # Absolute path - assume it's already a container path or use as is
    CONTAINER_CHECKPOINT_PATH="$CHECKPOINT_PATH"
else
    # Relative path - assume relative to /workspace
    CONTAINER_CHECKPOINT_PATH="/workspace/$CHECKPOINT_PATH"
fi

# Build Python command
PYTHON_CMD="cd Models && python train_unified.py \
    --checkpoint_path $CONTAINER_CHECKPOINT_PATH \
    --run_test True \
    --model_type $MODEL_TYPE \
    --prediction $PREDICTION \
    --hemisphere $HEMISPHERE \
    --n_examples $N_EXAMPLES \
    --myelination $MYELINATION \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE"

# Run the command in Docker container
echo "Executing test inference..."
docker exec "$CONTAINER_NAME" bash -c "$PYTHON_CMD"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Test inference completed successfully"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Test inference failed"
    echo "=========================================="
    exit 1
fi

echo ""
echo "Note: Container '$CONTAINER_NAME' is still running."
echo "To stop it, run: docker stop $CONTAINER_NAME"
echo "To remove it, run: docker rm -f $CONTAINER_NAME"

