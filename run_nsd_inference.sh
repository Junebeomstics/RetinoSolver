#!/bin/bash
#
# Run deepRetinotopy inference on NSD dataset and evaluate against ground truth
#
# Usage:
#   ./run_nsd_inference.sh [options]
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

# Auto-detect number of cores
if command -v nproc >/dev/null 2>&1; then
    N_JOBS=$(($(nproc) - 1))
else
    N_JOBS=4
fi
[ $N_JOBS -lt 1 ] && N_JOBS=1

# Parse command line arguments
while getopts "s:h:p:m:y:r:j:o:" opt; do
    case $opt in
        s) SUBJECT="$OPTARG" ;;
        h) HEMISPHERE="$OPTARG" ;;
        p) PREDICTION="$OPTARG" ;;
        m) MODEL_TYPE="$OPTARG" ;;
        y) MYELINATION="$OPTARG" ;;
        r) R2_THRESHOLD="$OPTARG" ;;
        j) N_JOBS="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            echo "Usage: $0 [-s subject] [-h hemisphere] [-p prediction] [-m model] [-y myelination] [-r r2_threshold] [-j n_jobs] [-o output_dir]"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# HCP surface directory
HCP_SURFACE_DIR="$PROJECT_ROOT/surface"
mkdir -p "$HCP_SURFACE_DIR"

echo "==============================================="
echo "deepRetinotopy Inference on NSD Dataset"
echo "==============================================="
echo "Subject: $SUBJECT"
echo "Hemisphere: $HEMISPHERE"
echo "Prediction: $PREDICTION"
echo "Model: $MODEL_TYPE"
echo "Myelination: $MYELINATION"
echo "R2 Threshold: $R2_THRESHOLD"
echo "Parallel Jobs: $N_JOBS"
echo "Output Directory: $OUTPUT_DIR"
echo "==============================================="

# Check if subject directory exists
if [ ! -d "$NSD_DIR/$SUBJECT" ]; then
    echo "ERROR: Subject directory not found: $NSD_DIR/$SUBJECT"
    exit 1
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

# Find checkpoint
DIRNAME="$PROJECT_ROOT/Models/checkpoints/${PREDICTION}_${HEMISPHERE_LONG}_${MODEL_TYPE}${NO_MYELIN_SUFFIX}"
FILENAME="${PRED_SHORT}_${HEMISPHERE_LONG}_${MODEL_TYPE}${NO_MYELIN_SUFFIX}_best_model_epoch*.pt"
CHECKPOINT_SEARCH="${DIRNAME}/${FILENAME}"

CHECKPOINT_PATH=$(ls -1 ${CHECKPOINT_SEARCH} 2>/dev/null | sort -V | tail -n 1)

if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: No checkpoint file found with pattern: ${CHECKPOINT_SEARCH}"
    exit 1
fi

echo ""
echo "Using checkpoint: $CHECKPOINT_PATH"
echo ""

# Step 1: Native to fsaverage conversion
echo "==============================================="
echo "[Step 1] Native to fsaverage Conversion"
echo "==============================================="

cd "$PROJECT_ROOT/run_from_freesurfer"

./1_native2fsaverage.sh \
    -s "$NSD_DIR" \
    -t "$HCP_SURFACE_DIR" \
    -h "$HEMISPHERE" \
    -j "$N_JOBS" \
    -i "$SUBJECT"

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

cd "$PROJECT_ROOT"

python Models/run_inference_freesurfer.py \
    --freesurfer_dir "$NSD_DIR" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --model_type "$MODEL_TYPE" \
    --prediction "$PREDICTION" \
    --hemisphere "$HEMISPHERE_LONG" \
    --myelination "$MYELINATION" \
    --subject_id "$SUBJECT"

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

cd "$PROJECT_ROOT/run_from_freesurfer"

./2_fsaverage2native.sh \
    -s "$NSD_DIR" \
    -t "$HCP_SURFACE_DIR" \
    -h "$HEMISPHERE" \
    -r "$PREDICTION" \
    -m "$MODEL_TYPE" \
    -y "$MYELINATION" \
    -j "$N_JOBS" \
    -i "$SUBJECT"

if [ $? -ne 0 ]; then
    echo "ERROR: Fsaverage to native conversion failed"
    exit 1
fi

echo "[Step 3] Completed!"

# Step 4: Evaluation
echo ""
echo "==============================================="
echo "[Step 4] Evaluating Against Ground Truth"
echo "==============================================="

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
