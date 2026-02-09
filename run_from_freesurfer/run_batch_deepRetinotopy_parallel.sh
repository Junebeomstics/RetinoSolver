#!/bin/bash

# Wrapper script to run deepRetinotopy inference for multiple subjects, predictions, and model types in parallel
#
# This script generates all combinations of subjects, predictions, and model types,
# then executes them in parallel using GNU parallel or background jobs.
#
# USAGE EXAMPLES:
#
# 1. Process all subjects with default settings:
#    ./run_batch_deepRetinotopy_parallel.sh \
#        --freesurfer_dir /path/to/freesurfer/subjects \
#        --hemisphere lh \
#        --hcp_surface_dir /path/to/surface
#
# 2. Process specific subjects with specific predictions and models:
#    ./run_batch_deepRetinotopy_parallel.sh \
#        --freesurfer_dir /path/to/freesurfer/subjects \
#        --subject_list sub-191033 sub-157336 \
#        --hemisphere lh \
#        --predictions eccentricity polarAngle \
#        --model_types baseline transolver_optionC \
#        --hcp_surface_dir /path/to/surface \
#        --parallel_jobs 4
#
# 3. Process all combinations with specific seeds:
#    ./run_batch_deepRetinotopy_parallel.sh \
#        --freesurfer_dir /path/to/freesurfer/subjects \
#        --hemisphere rh \
#        --seeds 0 1 2 \
#        --hcp_surface_dir /path/to/surface \
#        --parallel_jobs 8
#
# 4. Process with single seed (backward compatibility):
#    ./run_batch_deepRetinotopy_parallel.sh \
#        --freesurfer_dir /path/to/freesurfer/subjects \
#        --hemisphere lh \
#        --seed 0 \
#        --hcp_surface_dir /path/to/surface

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER_SCRIPT="$SCRIPT_DIR/run_deepRetinotopy_freesurfer_with_docker.sh"

# Default parameters
FREESURFER_DIR="/mnt/storage/junb/proj-5dceb267c4ae281d2c297b92/"
HEMISPHERE="rh"
PREDICTIONS=("polarAngle" "eccentricity" "pRFsize") #) #"eccentricity"  "pRFsize") # )
MODEL_TYPES=("baseline" "transolver_optionA" "transolver_optionC") #  "transolver_optionA" "transolver_optionC") #)
SEEDS=("0" "1" "2") # "1" "2" # Default: process seeds 0, 1, 2
SUBJECT_LIST=()
MYELINATION="False"
HCP_SURFACE_DIR="surface"
OUTPUT_DIR=""
PARALLEL_JOBS=12
USE_GNU_PARALLEL=true
SKIP_PREPROCESSING=false
SKIP_NATIVE_CONVERSION=false
LOG_DIR=""
CONTINUE_ON_ERROR=false
DOCKER_IMAGE=""
USE_WANDB="false"
WANDB_API_KEY=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --freesurfer_dir)
            FREESURFER_DIR="$2"
            shift 2
            ;;
        --subject_list)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                SUBJECT_LIST+=("$1")
                shift
            done
            ;;
        --subject_file)
            if [ -f "$2" ]; then
                while IFS= read -r line || [ -n "$line" ]; do
                    # Skip empty lines and comments
                    if [[ -n "$line" ]] && [[ ! "$line" =~ ^[[:space:]]*# ]]; then
                        SUBJECT_LIST+=(sub-"$line")
                    fi
                done < "$2"
            else
                echo "ERROR: Subject file not found: $2"
                exit 1
            fi
            shift 2
            ;;
        --hemisphere)
            HEMISPHERE="$2"
            shift 2
            ;;
        --predictions)
            shift
            PREDICTIONS=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                PREDICTIONS+=("$1")
                shift
            done
            ;;
        --model_types)
            shift
            MODEL_TYPES=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                MODEL_TYPES+=("$1")
                shift
            done
            ;;
        --seed)
            # Single seed (for backward compatibility)
            SEEDS=("$2")
            shift 2
            ;;
        --seeds)
            shift
            SEEDS=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                SEEDS+=("$1")
                shift
            done
            ;;
        --myelination)
            MYELINATION="$2"
            shift 2
            ;;
        --hcp_surface_dir)
            HCP_SURFACE_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --parallel_jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --no_gnu_parallel)
            USE_GNU_PARALLEL=false
            shift
            ;;
        --skip_preprocessing)
            SKIP_PREPROCESSING=true
            shift
            ;;
        --skip_native_conversion)
            SKIP_NATIVE_CONVERSION=true
            shift
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --continue_on_error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --docker_image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB="$2"
            shift 2
            ;;
        --wandb_api_key)
            WANDB_API_KEY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required options:"
            echo "  --freesurfer_dir PATH       Path to FreeSurfer subjects directory"
            echo "  --hcp_surface_dir PATH      Path to HCP surface templates"
            echo ""
            echo "Optional options:"
            echo "  --subject_list SUB1 SUB2    List of subject IDs to process (space-separated)"
            echo "  --subject_file FILE         File containing subject IDs (one per line)"
            echo "                             If neither is provided, processes all subjects in directory"
            echo "  --hemisphere lh|rh         Hemisphere to process (default: lh)"
            echo "  --predictions PRED1 PRED2  Prediction types (default: eccentricity polarAngle pRFsize)"
            echo "  --model_types TYPE1 TYPE2   Model types (default: baseline transolver_optionA transolver_optionB transolver_optionC)"
            echo "  --seed SEED                Single seed number (for backward compatibility)"
            echo "  --seeds SEED1 SEED2        Seed numbers to process (default: 0 1 2)"
            echo "  --myelination True|False   Whether myelination was used (default: False)"
            echo "  --output_dir PATH         Output directory (default: in-place)"
            echo "  --parallel_jobs N         Number of parallel jobs (default: 4)"
            echo "  --no_gnu_parallel         Use bash background jobs instead of GNU parallel"
            echo "  --skip_preprocessing       Skip preprocessing steps"
            echo "  --skip_native_conversion  Skip fsaverage to native conversion"
            echo "  --log_dir PATH            Directory for log files (default: ./logs_batch_<timestamp>)"
            echo "  --continue_on_error       Continue processing remaining jobs if one fails"
            echo "  --docker_image IMAGE       Docker image to use"
            echo "  --use_wandb True|False    Enable Wandb logging (default: false)"
            echo "  --wandb_api_key KEY       Wandb API key"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$FREESURFER_DIR" ]; then
    echo "ERROR: --freesurfer_dir is required"
    exit 1
fi

if [ -z "$HCP_SURFACE_DIR" ] && [ "$SKIP_PREPROCESSING" = false ] && [ "$SKIP_NATIVE_CONVERSION" = false ]; then
    echo "ERROR: --hcp_surface_dir is required when preprocessing or native conversion is enabled"
    exit 1
fi

# Check if wrapper script exists
if [ ! -f "$WRAPPER_SCRIPT" ]; then
    echo "ERROR: Wrapper script not found: $WRAPPER_SCRIPT"
    exit 1
fi

# Make wrapper script executable
chmod +x "$WRAPPER_SCRIPT"

# Discover subjects if not provided
if [ ${#SUBJECT_LIST[@]} -eq 0 ]; then
    echo "No subject list provided. Discovering subjects in $FREESURFER_DIR..."
    
    # Look for sub-* directories or dt-neuro-freesurfer.* directories
    if [ -d "$FREESURFER_DIR" ]; then
        # Check if this is a base directory with sub-* subdirectories
        for subdir in "$FREESURFER_DIR"/sub-*; do
            if [ -d "$subdir" ]; then
                SUBJECT_ID=$(basename "$subdir")
                SUBJECT_LIST+=("$SUBJECT_ID")
            fi
        done
        
        # If no sub-* directories found, check for dt-neuro-freesurfer.* directories
        if [ ${#SUBJECT_LIST[@]} -eq 0 ]; then
            for subdir in "$FREESURFER_DIR"/dt-neuro-freesurfer.*; do
                if [ -d "$subdir" ]; then
                    # Extract subject ID from parent directory
                    PARENT_DIR=$(dirname "$subdir")
                    SUBJECT_ID=$(basename "$PARENT_DIR")
                    if [[ "$SUBJECT_ID" =~ ^sub- ]]; then
                        SUBJECT_LIST+=("$SUBJECT_ID")
                    fi
                fi
            done
        fi
    fi
    
    if [ ${#SUBJECT_LIST[@]} -eq 0 ]; then
        echo "ERROR: No subjects found in $FREESURFER_DIR"
        echo "Please provide --subject_list or --subject_file"
        exit 1
    fi
    
    echo "Found ${#SUBJECT_LIST[@]} subjects: ${SUBJECT_LIST[*]}"
fi

# Setup log directory
if [ -z "$LOG_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./logs_batch_${TIMESTAMP}"
fi
mkdir -p "$LOG_DIR"

# Calculate total combinations
TOTAL_COMBINATIONS=$((${#SUBJECT_LIST[@]} * ${#PREDICTIONS[@]} * ${#MODEL_TYPES[@]} * ${#SEEDS[@]}))
COUNTER=0

# Display configuration
echo ""
echo "================================================================================"
echo "Batch Parallel Execution Configuration"
echo "================================================================================"
echo "  FreeSurfer Directory: $FREESURFER_DIR"
echo "  HCP Surface Directory: $HCP_SURFACE_DIR"
echo "  Hemisphere: $HEMISPHERE"
echo "  Subjects: ${#SUBJECT_LIST[@]}"
for subj in "${SUBJECT_LIST[@]}"; do
    echo "    - $subj"
done
echo "  Predictions: ${#PREDICTIONS[@]}"
for pred in "${PREDICTIONS[@]}"; do
    echo "    - $pred"
done
echo "  Model Types: ${#MODEL_TYPES[@]}"
for model in "${MODEL_TYPES[@]}"; do
    echo "    - $model"
done
echo "  Seeds: ${#SEEDS[@]}"
for seed in "${SEEDS[@]}"; do
    echo "    - $seed"
done
echo "  Myelination: $MYELINATION"
if [ ! -z "$OUTPUT_DIR" ]; then
    echo "  Output Directory: $OUTPUT_DIR"
fi
echo "  Parallel Jobs: $PARALLEL_JOBS"
echo "  Use GNU Parallel: $USE_GNU_PARALLEL"
echo "  Log Directory: $LOG_DIR"
echo "  Continue on Error: $CONTINUE_ON_ERROR"
echo "  Total Combinations: $TOTAL_COMBINATIONS"
echo "================================================================================"
echo ""

# Function to process a single combination
process_combination() {
    local subject_id="$1"
    local prediction="$2"
    local model_type="$3"
    local seed="$4"
    local job_num="$5"
    local total_jobs="$6"
    
    local log_file="$LOG_DIR/${subject_id}_${prediction}_${model_type}_seed${seed}.log"
    local start_time=$(date +%s)
    
    echo "[$job_num/$total_jobs] Starting: $subject_id | $prediction | $model_type | seed=$seed"
    echo "  Log: $log_file"
    
    # Build command
    local cmd="$WRAPPER_SCRIPT"
    cmd="$cmd --freesurfer_dir \"$FREESURFER_DIR\""
    cmd="$cmd --subject_id \"$subject_id\""
    cmd="$cmd --hemisphere \"$HEMISPHERE\""
    cmd="$cmd --prediction \"$prediction\""
    cmd="$cmd --model_type \"$model_type\""
    cmd="$cmd --myelination \"$MYELINATION\""
    cmd="$cmd --seed \"$seed\""
    
    if [ ! -z "$HCP_SURFACE_DIR" ]; then
        cmd="$cmd --hcp_surface_dir \"$HCP_SURFACE_DIR\""
    fi
    
    if [ ! -z "$OUTPUT_DIR" ]; then
        cmd="$cmd --output_dir \"$OUTPUT_DIR\""
    fi
    
    if [ "$SKIP_PREPROCESSING" = true ]; then
        cmd="$cmd --skip_preprocessing"
    fi
    
    if [ "$SKIP_NATIVE_CONVERSION" = true ]; then
        cmd="$cmd --skip_native_conversion"
    fi
    
    if [ ! -z "$DOCKER_IMAGE" ]; then
        cmd="$cmd --docker_image \"$DOCKER_IMAGE\""
    fi
    
    if [ ! -z "$USE_WANDB" ]; then
        cmd="$cmd --use_wandb \"$USE_WANDB\""
    fi
    
    if [ ! -z "$WANDB_API_KEY" ]; then
        cmd="$cmd --wandb_api_key \"$WANDB_API_KEY\""
    fi
    
    # Execute command and log output
    if eval "$cmd" > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        
        echo "[$job_num/$total_jobs] ✓ SUCCESS: $subject_id | $prediction | $model_type | seed=$seed (${minutes}m ${seconds}s)"
        echo "success:$subject_id:$prediction:$model_type:$seed:$duration" >> "$LOG_DIR/results.txt"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        
        echo "[$job_num/$total_jobs] ✗ FAILED: $subject_id | $prediction | $model_type | seed=$seed (${minutes}m ${seconds}s)"
        echo "  Check log: $log_file"
        echo "failed:$subject_id:$prediction:$model_type:$seed:$duration" >> "$LOG_DIR/results.txt"
        
        if [ "$CONTINUE_ON_ERROR" = false ]; then
            return 1
        else
            return 0
        fi
    fi
}

# Export function for parallel execution
export -f process_combination
export WRAPPER_SCRIPT FREESURFER_DIR HEMISPHERE MYELINATION HCP_SURFACE_DIR
export OUTPUT_DIR SKIP_PREPROCESSING SKIP_NATIVE_CONVERSION
export DOCKER_IMAGE USE_WANDB WANDB_API_KEY LOG_DIR CONTINUE_ON_ERROR

# Initialize results file
echo "# Results: subject:prediction:model_type:seed:duration_seconds" > "$LOG_DIR/results.txt"

# Generate all combinations and execute
START_TIME=$(date +%s)

if [ "$USE_GNU_PARALLEL" = true ] && command -v parallel &> /dev/null; then
    echo "Using GNU Parallel for execution..."
    
    # Create job list for GNU parallel
    JOB_LIST_FILE="$LOG_DIR/job_list.txt"
    > "$JOB_LIST_FILE"
    
    JOB_NUM=0
    for subject_id in "${SUBJECT_LIST[@]}"; do
        for prediction in "${PREDICTIONS[@]}"; do
            for model_type in "${MODEL_TYPES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    ((JOB_NUM++))
                    echo "$subject_id|$prediction|$model_type|$seed|$JOB_NUM|$TOTAL_COMBINATIONS" >> "$JOB_LIST_FILE"
                done
            done
        done
    done
    
    # Execute with GNU parallel
    cat "$JOB_LIST_FILE" | parallel -j "$PARALLEL_JOBS" --colsep '|' \
        process_combination {1} {2} {3} {4} {5} {6}
    
    PARALLEL_EXIT=$?
    
    if [ $PARALLEL_EXIT -ne 0 ]; then
        echo ""
        echo "WARNING: Some jobs failed. Check logs in $LOG_DIR"
    fi
else
    if [ "$USE_GNU_PARALLEL" = true ]; then
        echo "WARNING: GNU Parallel not found. Falling back to bash background jobs..."
    fi
    
    echo "Using bash background jobs for execution..."
    
    # Track running jobs
    declare -a PIDS=()
    RUNNING_JOBS=0
    
    JOB_NUM=0
    for subject_id in "${SUBJECT_LIST[@]}"; do
        for prediction in "${PREDICTIONS[@]}"; do
            for model_type in "${MODEL_TYPES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    ((JOB_NUM++))
                    
                    # Wait if we have too many running jobs
                    while [ $RUNNING_JOBS -ge $PARALLEL_JOBS ]; do
                        sleep 1
                        # Check which jobs are still running
                        RUNNING_JOBS=0
                        for pid in "${PIDS[@]}"; do
                            if kill -0 "$pid" 2>/dev/null; then
                                ((RUNNING_JOBS++))
                            fi
                        done
                    done
                    
                    # Start job in background
                    process_combination "$subject_id" "$prediction" "$model_type" "$seed" "$JOB_NUM" "$TOTAL_COMBINATIONS" &
                    PID=$!
                    PIDS+=("$PID")
                    ((RUNNING_JOBS++))
                    
                    # Small delay to avoid race conditions
                    sleep 0.5
                done
            done
        done
    done
    
    # Wait for all jobs to complete
    echo ""
    echo "Waiting for all jobs to complete..."
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done
fi

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

# Summarize results
echo ""
echo "================================================================================"
echo "Execution Summary"
echo "================================================================================"
echo "  Total Duration: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo ""

if [ -f "$LOG_DIR/results.txt" ]; then
    SUCCESSFUL=$(grep -c "^success:" "$LOG_DIR/results.txt" 2>/dev/null || echo "0")
    FAILED=$(grep -c "^failed:" "$LOG_DIR/results.txt" 2>/dev/null || echo "0")
    
    echo "  Successful: $SUCCESSFUL"
    echo "  Failed: $FAILED"
    echo ""
    
    if [ "$FAILED" -gt 0 ]; then
        echo "Failed combinations:"
        grep "^failed:" "$LOG_DIR/results.txt" | while IFS=: read -r status subject pred model seed duration; do
            echo "  - $subject | $pred | $model | seed=$seed"
        done
        echo ""
    fi
fi

echo "  Log Directory: $LOG_DIR"
echo "================================================================================"

# Exit with error if any jobs failed (unless continue_on_error was set)
if [ "$FAILED" -gt 0 ] && [ "$CONTINUE_ON_ERROR" = false ]; then
    exit 1
else
    exit 0
fi
