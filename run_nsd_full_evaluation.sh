#!/bin/bash
#
# Run complete deepRetinotopy evaluation on NSD dataset
# This script runs inference and evaluation for all prediction targets and both hemispheres
#
# Usage:
#   ./run_nsd_full_evaluation.sh [options]
#
# Options:
#   -s SUBJECT      Subject ID (default: subj01)
#   -m MODEL        Model type (default: baseline)
#   -y MYELINATION  Use myelination: True or False (default: False)
#   -r R2_THRESHOLD R2 threshold for evaluation (default: 0.1)
#   -o OUTPUT_DIR   Output directory (default: ./nsd_evaluation)

set -e

# Default values
SUBJECT="subj01"
MODEL_TYPE="baseline"
MYELINATION="False"
R2_THRESHOLD="0.1"
OUTPUT_DIR="./nsd_evaluation"

# Parse command line arguments
while getopts "s:m:y:r:o:" opt; do
    case $opt in
        s) SUBJECT="$OPTARG" ;;
        m) MODEL_TYPE="$OPTARG" ;;
        y) MYELINATION="$OPTARG" ;;
        r) R2_THRESHOLD="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            echo "Usage: $0 [-s subject] [-m model] [-y myelination] [-r r2_threshold] [-o output_dir]"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==============================================="
echo "Full deepRetinotopy Evaluation on NSD Dataset"
echo "==============================================="
echo "Subject: $SUBJECT"
echo "Model: $MODEL_TYPE"
echo "Myelination: $MYELINATION"
echo "R2 Threshold: $R2_THRESHOLD"
echo "Output Directory: $OUTPUT_DIR"
echo "==============================================="
echo ""

# Start timing
TOTAL_START=$(date +%s)

# Arrays of predictions and hemispheres
PREDICTIONS=("eccentricity" "polarAngle" "pRFsize")
HEMISPHERES=("lh" "rh")

# Counter for completed tasks
COMPLETED=0
TOTAL=$((${#PREDICTIONS[@]} * ${#HEMISPHERES[@]}))

# Run inference and evaluation for each combination
for PREDICTION in "${PREDICTIONS[@]}"; do
    for HEMISPHERE in "${HEMISPHERES[@]}"; do
        echo ""
        echo "###############################################"
        echo "Processing: $PREDICTION - $HEMISPHERE"
        echo "Progress: $((COMPLETED + 1))/$TOTAL"
        echo "###############################################"
        echo ""
        
        START=$(date +%s)
        
        # Run the inference and evaluation pipeline
        bash "$SCRIPT_DIR/run_nsd_inference.sh" \
            -s "$SUBJECT" \
            -h "$HEMISPHERE" \
            -p "$PREDICTION" \
            -m "$MODEL_TYPE" \
            -y "$MYELINATION" \
            -r "$R2_THRESHOLD" \
            -o "$OUTPUT_DIR"
        
        if [ $? -eq 0 ]; then
            END=$(date +%s)
            DURATION=$((END - START))
            echo ""
            echo "✓ Completed $PREDICTION - $HEMISPHERE in ${DURATION}s"
            COMPLETED=$((COMPLETED + 1))
        else
            echo ""
            echo "✗ Failed: $PREDICTION - $HEMISPHERE"
            echo "Continuing with remaining tasks..."
        fi
        
        echo ""
    done
done

# Calculate total time
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "==============================================="
echo "Full Evaluation Complete!"
echo "==============================================="
echo "Completed: $COMPLETED/$TOTAL tasks"
echo "Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Generate summary report
echo "Generating summary report..."
python "$SCRIPT_DIR/summarize_nsd_results.py" \
    --results_dir "$OUTPUT_DIR/results" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Summary report saved to: $OUTPUT_DIR/summary_report.txt"
echo "==============================================="
