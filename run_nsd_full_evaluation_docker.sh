#!/bin/bash
#
# Run complete deepRetinotopy evaluation on NSD dataset using Docker
# This script runs inference and evaluation for all prediction targets and both hemispheres
#
# Usage:
#   ./run_nsd_full_evaluation_docker.sh [options]
#
# Options:
#   -s SUBJECT      Subject ID (default: subj01)
#   -m MODEL        Model type (default: baseline)
#   -y MYELINATION  Use myelination: True or False (default: False)
#   -r R2_THRESHOLD R2 threshold for evaluation (default: 0.1)
#   -o OUTPUT_DIR   Output directory (default: ./nsd_evaluation)
#   -g USE_GPU      Use GPU: true or false (default: true)
#   -c CONCURRENT   Max concurrent jobs (default: 2)

set -e

# Default values
SUBJECT="subj01"
MODEL_TYPE="baseline"
MYELINATION="False"
R2_THRESHOLD="0.1"
OUTPUT_DIR="./nsd_evaluation"
USE_GPU="true"
MAX_CONCURRENT=2

# Parse command line arguments
while getopts "s:m:y:r:o:g:c:" opt; do
    case $opt in
        s) SUBJECT="$OPTARG" ;;
        m) MODEL_TYPE="$OPTARG" ;;
        y) MYELINATION="$OPTARG" ;;
        r) R2_THRESHOLD="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        g) USE_GPU="$OPTARG" ;;
        c) MAX_CONCURRENT="$OPTARG" ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            echo "Usage: $0 [-s subject] [-m model] [-y myelination] [-r r2_threshold] [-o output_dir] [-g use_gpu] [-c concurrent]"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==============================================="
echo "Full deepRetinotopy Evaluation on NSD (Docker)"
echo "==============================================="
echo "Subject: $SUBJECT"
echo "Model: $MODEL_TYPE"
echo "Myelination: $MYELINATION"
echo "R2 Threshold: $R2_THRESHOLD"
echo "Output Directory: $OUTPUT_DIR"
echo "GPU: $USE_GPU"
echo "Max Concurrent Jobs: $MAX_CONCURRENT"
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

# Array to track running job PIDs
declare -a running_jobs=()

# Function to wait for a job slot to become available
wait_for_slot() {
    while [ ${#running_jobs[@]} -ge $MAX_CONCURRENT ]; do
        # Check which jobs are still running
        local new_jobs=()
        for pid in "${running_jobs[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                # Job is still running
                new_jobs+=("$pid")
            else
                # Job completed
                wait "$pid" 2>/dev/null
                COMPLETED=$((COMPLETED + 1))
                echo ""
                echo "✓ Job completed (PID: $pid). Progress: $COMPLETED/$TOTAL"
                echo ""
            fi
        done
        running_jobs=("${new_jobs[@]}")
        
        # If still at max capacity, wait a bit
        if [ ${#running_jobs[@]} -ge $MAX_CONCURRENT ]; then
            sleep 5
        fi
    done
}

# Run inference and evaluation for each combination
for PREDICTION in "${PREDICTIONS[@]}"; do
    for HEMISPHERE in "${HEMISPHERES[@]}"; do
        # Wait for available slot
        wait_for_slot
        
        echo ""
        echo "###############################################"
        echo "Starting: $PREDICTION - $HEMISPHERE"
        echo "Progress: $((COMPLETED + 1))/$TOTAL"
        echo "Running jobs: ${#running_jobs[@]}/$MAX_CONCURRENT"
        echo "###############################################"
        echo ""
        
        # Run the inference and evaluation pipeline in background
        (
            bash "$SCRIPT_DIR/run_nsd_inference_docker.sh" \
                -s "$SUBJECT" \
                -h "$HEMISPHERE" \
                -p "$PREDICTION" \
                -m "$MODEL_TYPE" \
                -y "$MYELINATION" \
                -r "$R2_THRESHOLD" \
                -o "$OUTPUT_DIR" \
                -g "$USE_GPU"
        ) > "$OUTPUT_DIR/${SUBJECT}_${HEMISPHERE}_${PREDICTION}.log" 2>&1 &
        
        # Add PID to running jobs
        running_jobs+=($!)
        echo "Started job (PID: $!)"
        
        # Small delay between starting jobs
        sleep 2
    done
done

# Wait for all remaining jobs to complete
echo ""
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
            COMPLETED=$((COMPLETED + 1))
            echo "✓ Job completed (PID: $pid). Progress: $COMPLETED/$TOTAL"
        fi
    done
    running_jobs=("${new_jobs[@]}")
    
    if [ ${#running_jobs[@]} -gt 0 ]; then
        echo "Waiting for ${#running_jobs[@]} job(s) to complete..."
        sleep 5
    fi
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

# Check for any failed jobs by looking at log files
echo "Checking for errors in log files..."
FAILED=0
for log in "$OUTPUT_DIR"/*.log; do
    if [ -f "$log" ]; then
        if grep -q "ERROR:" "$log" || grep -q "failed" "$log"; then
            echo "⚠ Found errors in: $(basename $log)"
            FAILED=$((FAILED + 1))
        fi
    fi
done

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Warning: $FAILED task(s) may have failed. Check log files in $OUTPUT_DIR/"
    echo ""
fi

# Generate summary report
echo "Generating summary report..."
if [ -d "$OUTPUT_DIR/results" ] && [ "$(ls -A "$OUTPUT_DIR/results"/*.json 2>/dev/null)" ]; then
    python "$SCRIPT_DIR/summarize_nsd_results.py" \
        --results_dir "$OUTPUT_DIR/results" \
        --output_dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Summary report saved to: $OUTPUT_DIR/summary_report.txt"
    else
        echo "Warning: Failed to generate summary report"
    fi
else
    echo "Warning: No result files found. Summary report not generated."
fi

echo ""
echo "==============================================="
echo "View individual logs: ls $OUTPUT_DIR/*.log"
echo "View summary: cat $OUTPUT_DIR/summary_report.txt"
echo "==============================================="
