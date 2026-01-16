#!/bin/bash
#
# Run deepRetinotopy inference on NSD dataset using Docker
#
# Usage:
#   ./run_nsd_inference_docker.sh [options]
#
# Options:
#   -s SUBJECT      Subject ID (default: subj01)
#   -h HEMISPHERE   Hemisphere: lh or rh (default: lh)
#   -p PREDICTION   Prediction target: eccentricity, polarAngle, pRFsize (default: eccentricity)
#   -m MODEL        Model type (default: baseline)
#   -y MYELINATION  Use myelination: True or False (default: False)
#   -r R2_THRESHOLD R2 threshold for evaluation (default: 0.1)
#   -j N_JOBS       Number of parallel jobs (default: auto-detect)
#   -o OUTPUT_DIR   Output directory (default: ./nsd_evaluation)
#   -g USE_GPU      Use GPU: true or false (default: true)

set -e

# Default values
SUBJECT="subj01"
HEMISPHERE="lh"
PREDICTION="eccentricity"
MODEL_TYPE="baseline"
MYELINATION="False"
R2_THRESHOLD="0.1"
OUTPUT_DIR="./nsd_evaluation"
NSD_DIR="/mnt/external_storage1/natural-scenes-dataset/nsddata/freesurfer"
USE_GPU="true"

# Auto-detect number of cores
if command -v nproc >/dev/null 2>&1; then
    N_JOBS=$(($(nproc) - 1))
else
    N_JOBS=4
fi
[ $N_JOBS -lt 1 ] && N_JOBS=1

# Docker settings
DOCKER_IMAGE=${DOCKER_IMAGE:-"vnmd/deepretinotopy_1.0.18:latest"}
CONTAINER_NAME="deepretinotopy_nsd_eval"

# Parse command line arguments
while getopts "s:h:p:m:y:r:j:o:g:" opt; do
    case $opt in
        s) SUBJECT="$OPTARG" ;;
        h) HEMISPHERE="$OPTARG" ;;
        p) PREDICTION="$OPTARG" ;;
        m) MODEL_TYPE="$OPTARG" ;;
        y) MYELINATION="$OPTARG" ;;
        r) R2_THRESHOLD="$OPTARG" ;;
        j) N_JOBS="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        g) USE_GPU="$OPTARG" ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            echo "Usage: $0 [-s subject] [-h hemisphere] [-p prediction] [-m model] [-y myelination] [-r r2_threshold] [-j n_jobs] [-o output_dir] [-g use_gpu]"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# HCP surface directory (will be created if needed)
HCP_SURFACE_DIR="$PROJECT_ROOT/surface"
mkdir -p "$HCP_SURFACE_DIR"

echo "==============================================="
echo "deepRetinotopy Inference on NSD Dataset (Docker)"
echo "==============================================="
echo "Subject: $SUBJECT"
echo "Hemisphere: $HEMISPHERE"
echo "Prediction: $PREDICTION"
echo "Model: $MODEL_TYPE"
echo "Myelination: $MYELINATION"
echo "R2 Threshold: $R2_THRESHOLD"
echo "Parallel Jobs: $N_JOBS"
echo "Output Directory: $OUTPUT_DIR"
echo "Docker Image: $DOCKER_IMAGE"
echo "GPU: $USE_GPU"
echo "==============================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker image exists
if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
    echo "Docker image '$DOCKER_IMAGE' not found. Attempting to pull..."
    if ! docker pull "$DOCKER_IMAGE"; then
        echo "ERROR: Failed to pull Docker image '$DOCKER_IMAGE'."
        exit 1
    fi
fi

# Check if subject directory exists
if [ ! -d "$NSD_DIR/$SUBJECT" ]; then
    echo "ERROR: Subject directory not found: $NSD_DIR/$SUBJECT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

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
    DOCKER_CMD="$DOCKER_CMD -v $NSD_DIR:/mnt/nsd_freesurfer"
    DOCKER_CMD="$DOCKER_CMD -v $HCP_SURFACE_DIR:/mnt/hcp_surface"
    DOCKER_CMD="$DOCKER_CMD -w /workspace"
    DOCKER_CMD="$DOCKER_CMD $DOCKER_IMAGE"
    DOCKER_CMD="$DOCKER_CMD tail -f /dev/null"  # Keep container running
    
    eval "$DOCKER_CMD" > /dev/null 2>&1
    CONTAINER_RUNNING=true
    echo "Container created successfully."
fi

# Map hemisphere to long name
if [ "$HEMISPHERE" == "lh" ]; then
    HEMISPHERE_LONG="Left"
elif [ "$HEMISPHERE" == "rh" ]; then
    HEMISPHERE_LONG="Right"
else
    echo "ERROR: Invalid hemisphere: $HEMISPHERE"
    exit 1
fi

# Determine checkpoint path
if [[ "${MYELINATION}" == "True" ]]; then
    NO_MYELIN_SUFFIX=""
else
    NO_MYELIN_SUFFIX="_noMyelin"
fi

# Convert prediction to short name
if [[ "${PREDICTION}" == "eccentricity" ]]; then
    PRED_SHORT="ecc"
elif [[ "${PREDICTION}" == "polarAngle" ]]; then
    PRED_SHORT="PA"
elif [[ "${PREDICTION}" == "pRFsize" ]]; then
    PRED_SHORT="size"
else
    PRED_SHORT="${PREDICTION}"
fi

# Find checkpoint (inside container path)
CHECKPOINT_PATTERN="/workspace/Models/checkpoints/${PREDICTION}_${HEMISPHERE_LONG}_${MODEL_TYPE}${NO_MYELIN_SUFFIX}/${PRED_SHORT}_${HEMISPHERE_LONG}_${MODEL_TYPE}${NO_MYELIN_SUFFIX}_best_model_epoch*.pt"

echo ""
echo "Searching for checkpoint with pattern: $CHECKPOINT_PATTERN"

# Find checkpoint in container
CHECKPOINT_PATH=$(docker exec "$CONTAINER_NAME" bash -c "ls -1 $CHECKPOINT_PATTERN 2>/dev/null | sort -V | tail -n 1")

if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: No checkpoint file found with pattern: ${CHECKPOINT_PATTERN}"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT_PATH"
echo ""

# Step 1: Native to fsaverage conversion
echo "==============================================="
echo "[Step 1] Native to fsaverage Conversion"
echo "==============================================="

STEP1_CMD="cd /workspace/run_from_freesurfer && ./1_native2fsaverage.sh \
    -s /mnt/nsd_freesurfer \
    -t /mnt/hcp_surface \
    -h $HEMISPHERE \
    -j $N_JOBS \
    -i $SUBJECT"

docker exec "$CONTAINER_NAME" bash -c "$STEP1_CMD"

if [ $? -ne 0 ]; then
    echo "ERROR: Native to fsaverage conversion failed"
    exit 1
fi

echo "[Step 1] Completed!"

# Step 2: Inference
echo ""
echo "==============================================="
echo "[Step 2] Running Inference"
echo "==============================================="

STEP2_CMD="cd /workspace && python Models/run_inference_freesurfer.py \
    --freesurfer_dir /mnt/nsd_freesurfer \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_type $MODEL_TYPE \
    --prediction $PREDICTION \
    --hemisphere $HEMISPHERE_LONG \
    --myelination $MYELINATION \
    --subject_id $SUBJECT"

docker exec "$CONTAINER_NAME" bash -c "$STEP2_CMD"

if [ $? -ne 0 ]; then
    echo "ERROR: Inference failed"
    exit 1
fi

echo "[Step 2] Completed!"

# Step 3: Fsaverage to native conversion
echo ""
echo "==============================================="
echo "[Step 3] Fsaverage to Native Space Conversion"
echo "==============================================="

STEP3_CMD="cd /workspace/run_from_freesurfer && ./2_fsaverage2native.sh \
    -s /mnt/nsd_freesurfer \
    -t /mnt/hcp_surface \
    -h $HEMISPHERE \
    -r $PREDICTION \
    -m $MODEL_TYPE \
    -y $MYELINATION \
    -j $N_JOBS \
    -i $SUBJECT"

docker exec "$CONTAINER_NAME" bash -c "$STEP3_CMD"

if [ $? -ne 0 ]; then
    echo "ERROR: Fsaverage to native conversion failed"
    exit 1
fi

echo "[Step 3] Completed!"

# Step 4: Copy results from container to host and run evaluation
echo ""
echo "==============================================="
echo "[Step 4] Evaluating Against Ground Truth"
echo "==============================================="

# Check if prediction files were generated in the container
echo "Checking for prediction files in container..."
DEEPRET_DIR="/mnt/nsd_freesurfer/$SUBJECT/deepRetinotopy"
MODEL_NAME="model"
if [ "$MODEL_TYPE" != "baseline" ]; then
    MODEL_NAME="$MODEL_TYPE"
fi

PRED_FILE_NATIVE="${DEEPRET_DIR}/${SUBJECT}.predicted_${PREDICTION}_${MODEL_NAME}.${HEMISPHERE}.native.func.gii"
docker exec "$CONTAINER_NAME" bash -c "ls -lh $PRED_FILE_NATIVE" || echo "Native space file not found (will try FSLR space)"

# Run evaluation (Python can access NSD data directly on host)
cd "$PROJECT_ROOT"

python evaluate_nsd_prf.py \
    --nsd_dir "$NSD_DIR" \
    --subject "$SUBJECT" \
    --hemisphere "$HEMISPHERE" \
    --prediction "$PREDICTION" \
    --model_type "$MODEL_TYPE" \
    --myelination "$MYELINATION" \
    --output_dir "$OUTPUT_DIR" \
    --r2_threshold "$R2_THRESHOLD"

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi

echo "[Step 4] Completed!"

echo ""
echo "==============================================="
echo "Pipeline Completed Successfully!"
echo "==============================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Note: Container '$CONTAINER_NAME' is still running."
echo "To stop it: docker stop $CONTAINER_NAME"
echo "To remove it: docker rm -f $CONTAINER_NAME"
echo ""
