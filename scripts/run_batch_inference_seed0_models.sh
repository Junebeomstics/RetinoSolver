#!/bin/bash
"""
Batch inference script for seed0 test subjects with all specified model checkpoints.

This script runs inference for all test subjects from seed0 using the following models:
- eccentricity_Left_baseline_noMyelin_seed0
- eccentricity_Left_transolver_optionA_noMyelin_seed0
- eccentricity_Left_transolver_optionC_noMyelin_seed0
- polarAngle_Left_baseline_noMyelin_seed0
- polarAngle_Left_transolver_optionA_noMyelin_seed0
- polarAngle_Left_transolver_optionC_noMyelin_seed0

Usage:
    # Sequential execution (safer, easier to debug)
    ./run_batch_inference_seed0_models.sh

    # Parallel execution (faster, uses 4 workers)
    ./run_batch_inference_seed0_models.sh --parallel

    # Parallel execution with custom number of workers
    ./run_batch_inference_seed0_models.sh --parallel --num_workers 8
"""

# Configuration
SUBJECT_LIST="Retinotopy/data/subject_splits/seed0/test_subjects.txt"
OUTPUT_DIR="inference_output_batch_seed0"
DATA_DIR="Retinotopy/data/raw/converted"

# List of checkpoint directories
CHECKPOINT_DIRS=(
    "Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed0"
    "Models/output_wandb/eccentricity_Left_transolver_optionA_noMyelin_seed0"
    "Models/output_wandb/eccentricity_Left_transolver_optionC_noMyelin_seed0"
    "Models/output_wandb/polarAngle_Left_baseline_noMyelin_seed0"
    "Models/output_wandb/polarAngle_Left_transolver_optionA_noMyelin_seed0"
    "Models/output_wandb/polarAngle_Left_transolver_optionC_noMyelin_seed0"
)

# Parse command line arguments
PARALLEL_FLAG=""
NUM_WORKERS_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL_FLAG="--parallel"
            shift
            ;;
        --num_workers)
            NUM_WORKERS_FLAG="--num_workers $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--parallel] [--num_workers N]"
            exit 1
            ;;
    esac
done

# Print configuration
echo "================================================================================"
echo "Batch Inference Configuration"
echo "================================================================================"
echo "Subject list: $SUBJECT_LIST"
echo "Output directory: $OUTPUT_DIR"
echo "Data directory: $DATA_DIR"
echo "Number of checkpoint directories: ${#CHECKPOINT_DIRS[@]}"
echo ""
echo "Checkpoint directories:"
for dir in "${CHECKPOINT_DIRS[@]}"; do
    echo "  - $dir"
done
echo ""
if [ -n "$PARALLEL_FLAG" ]; then
    echo "Mode: PARALLEL"
    if [ -n "$NUM_WORKERS_FLAG" ]; then
        echo "Workers: $NUM_WORKERS_FLAG"
    else
        echo "Workers: 4 (default)"
    fi
else
    echo "Mode: SEQUENTIAL"
fi
echo "================================================================================"
echo ""

# Check if subject list exists
if [ ! -f "$SUBJECT_LIST" ]; then
    echo "Error: Subject list file not found: $SUBJECT_LIST"
    exit 1
fi

# Check if checkpoint directories exist
for dir in "${CHECKPOINT_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Error: Checkpoint directory not found: $dir"
        exit 1
    fi
done

# Run batch inference
python3 run_batch_inference_from_fslr_curv_using_checkpoint.py \
    --subject_list "$SUBJECT_LIST" \
    --checkpoint_dirs "${CHECKPOINT_DIRS[@]}" \
    --output_base_dir "$OUTPUT_DIR" \
    --data_dir "$DATA_DIR" \
    $PARALLEL_FLAG \
    $NUM_WORKERS_FLAG

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "Batch inference completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "Batch inference failed!"
    echo "Check the error messages above for details."
    echo "================================================================================"
    exit 1
fi
