#!/bin/bash

# Transolver OptionC hyperparameter search experiment script
# Limits concurrent executions and runs sequentially considering 10GB memory constraints

# Set base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Docker settings
DOCKER_IMAGE=${DOCKER_IMAGE:-"vnmd/deepretinotopy_1.0.18:latest"}
CONTAINER_NAME="deepretinotopy_train"
USE_GPU=${USE_GPU:-"true"}

# Training hyperparameters (fixed)
N_EPOCHS=500
LR_INIT=0.001
LR_DECAY_EPOCH=250
LR_DECAY=0.0001
INTERM_SAVE_EVERY=25
BATCH_SIZE=1
N_EXAMPLES=181
OUTPUT_DIR="./output/hyperparameter_search"

# Optimizer and scheduler settings (fixed)
OPTIMIZER="AdamW"
SCHEDULER="cosine"
WEIGHT_DECAY=1e-5
MAX_GRAD_NORM=0.1

# Fixed architecture parameters
REF=8
UNIFIED_POS=0
DROPOUT=0.0

# Neptune settings
USE_NEPTUNE=${USE_NEPTUNE:-"true"}
NEPTUNE_PROJECT="kjb961013/Retinosolver"
NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZWNlMzk4Ny1hOTVlLTRjMzgtOWI4ZS1hY2FkYTY4MzNhYzMifQ=="

# Model and task settings
MODEL_TYPE="transolver_optionC"
PREDICTION="polarAngle"
HEMISPHERE="Left"

# Memory management: Maximum concurrent experiments (reduced for 10GB memory)
# Conservative setting: 2-3 concurrent experiments to avoid OOM
MAX_CONCURRENT=2  # Reduced to 2 for 10GB memory constraint (can increase to 3 if stable)

# Experiment configurations
# Format: "experiment_name:n_layers:n_hidden:slice_num:mlp_ratio"
# Ordered by estimated memory usage (smaller models first)
declare -a EXPERIMENTS=(
    "small:6:96:48:1"           # Smallest model (lowest memory)
    "shallow:6:128:64:1"        # Shallow model
    "narrow:8:96:64:1"          # Narrow model
    "low_slice:8:128:48:1"      # Low slice
    "baseline:8:128:64:1"       # Baseline (reference)
    "mlp2:8:128:64:2"           # MLP ratio 2
    "high_slice:8:128:80:1"     # High slice
    "wide:8:160:64:1"           # Wide model
    "deep:10:128:64:1"         # Deep model
    "large:10:160:80:2"         # Largest model (highest memory)
)

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
    DOCKER_CMD="$DOCKER_CMD tail -f /dev/null"
    
    eval "$DOCKER_CMD" > /dev/null 2>&1
    CONTAINER_RUNNING=true
fi

# Install required packages once
echo "Installing required packages (neptune, einops) in container..."
INSTALL_CMD="pip install --quiet --no-cache-dir neptune einops"
docker exec "$CONTAINER_NAME" bash -c "$INSTALL_CMD" > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "Warning: Failed to install packages. Continuing anyway..."
fi

# Function to check running experiments in container
check_running_experiments() {
    # Count running Python processes in container
    docker exec "$CONTAINER_NAME" bash -c "ps aux | grep 'python train_unified.py' | grep -v grep | wc -l" 2>/dev/null | tr -d ' '
}

# Function to check memory usage (optional monitoring)
check_memory_usage() {
    if [ "$USE_GPU" = "true" ]; then
        # Check GPU memory if available
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "N/A"
    else
        # Check system memory
        free -m | awk 'NR==2{printf "%.1f", $3/1024}' 2>/dev/null || echo "N/A"
    fi
}

# Function to wait until a slot is available
wait_for_slot() {
    while [ $(check_running_experiments) -ge $MAX_CONCURRENT ]; do
        running=$(check_running_experiments)
        mem_usage=$(check_memory_usage)
        echo "  [Queue] Waiting for slot (running: $running/$MAX_CONCURRENT, memory: ${mem_usage}GB)..."
        sleep 30  # Check every 30 seconds
    done
}

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local n_layers=$2
    local n_hidden=$3
    local slice_num=$4
    local mlp_ratio=$5
    
    echo "=========================================="
    echo "Starting experiment: $exp_name"
    echo "  n_layers=$n_layers, n_hidden=$n_hidden, slice_num=$slice_num, mlp_ratio=$mlp_ratio"
    echo "=========================================="
    
    # Wait for available slot
    wait_for_slot
    
    # Build Python command
    PYTHON_CMD="cd Models && python train_unified.py \
        --model_type $MODEL_TYPE \
        --prediction $PREDICTION \
        --hemisphere $HEMISPHERE \
        --n_epochs $N_EPOCHS \
        --lr_init $LR_INIT \
        --lr_decay_epoch $LR_DECAY_EPOCH \
        --lr_decay $LR_DECAY \
        --scheduler $SCHEDULER \
        --optimizer $OPTIMIZER \
        --weight_decay $WEIGHT_DECAY \
        --max_grad_norm $MAX_GRAD_NORM \
        --n_layers $n_layers \
        --n_hidden $n_hidden \
        --n_heads 8 \
        --slice_num $slice_num \
        --mlp_ratio $mlp_ratio \
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
    
    # Run the command in Docker container in background
    # Store PID for tracking
    docker exec -d "$CONTAINER_NAME" bash -c "$PYTHON_CMD" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        running=$(check_running_experiments)
        mem_usage=$(check_memory_usage)
        echo "  ✓ Experiment '$exp_name' started (running: $running/$MAX_CONCURRENT, memory: ${mem_usage}GB)"
    else
        echo "  ✗ Experiment '$exp_name' failed to start"
        return 1
    fi
    
    # Small delay to avoid race conditions
    sleep 2
    echo ""
}

# Main execution
echo "=========================================="
echo "Transolver OptionC Hyperparameter Search"
echo "=========================================="
echo "  Docker Image: $DOCKER_IMAGE"
echo "  GPU: $USE_GPU"
echo "  Neptune: $USE_NEPTUNE"
echo "  Max Concurrent: $MAX_CONCURRENT"
echo "  Total Experiments: ${#EXPERIMENTS[@]}"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$PROJECT_ROOT/Models/$OUTPUT_DIR"

# Run experiments
for exp_config in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name n_layers n_hidden slice_num mlp_ratio <<< "$exp_config"
    
    run_experiment "$exp_name" "$n_layers" "$n_hidden" "$slice_num" "$mlp_ratio"
done

# Wait for all experiments to complete
echo "=========================================="
echo "All experiments started. Waiting for completion..."
echo "=========================================="

while [ $(check_running_experiments) -gt 0 ]; do
    running=$(check_running_experiments)
    mem_usage=$(check_memory_usage)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running experiments: $running/$MAX_CONCURRENT, Memory: ${mem_usage}GB"
    sleep 60  # Check every minute
done

echo "=========================================="
echo "All experiments completed!"
echo "Note: Container '$CONTAINER_NAME' is still running."
echo "To stop it, run: docker stop $CONTAINER_NAME"
echo "To remove it, run: docker rm -f $CONTAINER_NAME"
echo "=========================================="

