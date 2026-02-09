#!/bin/bash

# Batch inference script for running deepRetinotopy inference pipeline
# across multiple combinations of subjects, models, hemispheres, predictions, and seeds

# Set base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../" && pwd)"
cd "$PROJECT_ROOT"

# Path to the main inference script
MAIN_SCRIPT="$SCRIPT_DIR/run_deepRetinotopy_freesurfer_with_docker.sh"

# Default parameter lists (modify these arrays as needed)
SUBJECT_IDS=("sub-157336")
MODEL_TYPES=("baseline" "transolver_optionA")
HEMISPHERES=("lh" "rh")
PREDICTIONS=("eccentricity" "polarAngle" "pRFsize")
SEEDS=("1")

# Common parameters (shared across all runs)
FREESURFER_BASE_DIR=""  # Will be set via --freesurfer_base_dir or auto-detected
HCP_SURFACE_DIR="surface"
MYELINATION="False"
SKIP_MYELIN=false
SKIP_PREPROCESSING=false
SKIP_NATIVE_CONVERSION=false
OUTPUT_DIR=""
N_JOBS=$(($(nproc) - 1))
[ $N_JOBS -lt 1 ] && N_JOBS=1

# Docker settings
DOCKER_IMAGE=${DOCKER_IMAGE:-"vnmd/deepretinotopy_1.0.18:latest"}
USE_GPU=${USE_GPU:-"true"}

# Wandb settings
USE_WANDB=${USE_WANDB:-"false"}
WANDB_API_KEY=${WANDB_API_KEY:-""}

# Execution options
CONTINUE_ON_ERROR=false
DRY_RUN=false
LOG_DIR=""
FORCE_RECREATE_CONTAINER=false
SUBJECT_LIST_FILE=""  # Will be set if using auto-detection
PARALLEL_INFERENCE=1  # Number of parallel inference processes (default: 1 = sequential)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --freesurfer_base_dir)
            FREESURFER_BASE_DIR="$2"
            shift 2
            ;;
        --hcp_surface_dir)
            HCP_SURFACE_DIR="$2"
            shift 2
            ;;
        --subject_ids)
            # Parse comma-separated list, "all" for auto-detection, or path to txt file
            if [ "$2" = "all" ] || [ "$2" = "ALL" ]; then
                SUBJECT_IDS=("all")  # Special marker to detect all subjects later
            else
                # Check if it's a file path - try multiple possible locations
                SUBJECT_FILE=""
                if [ -f "$2" ]; then
                    # File exists at given path
                    SUBJECT_FILE="$2"
                elif [ -f "$(dirname "$0")/$2" ]; then
                    # File exists relative to script directory
                    SUBJECT_FILE="$(dirname "$0")/$2"
                elif [ -f "$(dirname "$0")/../$2" ]; then
                    # File exists relative to project root
                    SUBJECT_FILE="$(dirname "$0")/../$2"
                fi
                
                if [ -n "$SUBJECT_FILE" ]; then
                    # Read subjects from file
                    echo "Reading subject IDs from file: $SUBJECT_FILE"
                    SUBJECT_IDS=()
                    while IFS= read -r line || [ -n "$line" ]; do
                        # Skip empty lines and comments
                        line=$(echo "$line" | sed 's/#.*//' | xargs)  # Remove comments and trim
                        if [ -n "$line" ]; then
                            # Add sub- prefix if not present
                            if [[ ! "$line" =~ ^sub- ]]; then
                                line="sub-${line}"
                            fi
                            SUBJECT_IDS+=("$line")
                        fi
                    done < "$SUBJECT_FILE"
                    
                    if [ ${#SUBJECT_IDS[@]} -eq 0 ]; then
                        echo "ERROR: No valid subject IDs found in file: $SUBJECT_FILE"
                        exit 1
                    fi
                    
                    echo "Loaded ${#SUBJECT_IDS[@]} subject(s) from file"
                else
                    # Treat as comma-separated list
                    IFS=',' read -ra SUBJECT_IDS <<< "$2"
                    # Add sub- prefix if not present for comma-separated list
                    for i in "${!SUBJECT_IDS[@]}"; do
                        if [[ ! "${SUBJECT_IDS[$i]}" =~ ^sub- ]]; then
                            SUBJECT_IDS[$i]="sub-${SUBJECT_IDS[$i]}"
                        fi
                    done
                fi
            fi
            shift 2
            ;;
        --model_types)
            IFS=',' read -ra MODEL_TYPES <<< "$2"
            shift 2
            ;;
        --hemispheres)
            IFS=',' read -ra HEMISPHERES <<< "$2"
            shift 2
            ;;
        --predictions)
            IFS=',' read -ra PREDICTIONS <<< "$2"
            shift 2
            ;;
        --seeds)
            IFS=',' read -ra SEEDS <<< "$2"
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
        --skip_myelin)
            SKIP_MYELIN=true
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
        --n_jobs)
            N_JOBS="$2"
            shift 2
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
        --continue_on_error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --force_recreate_container)
            FORCE_RECREATE_CONTAINER=true
            shift
            ;;
        --parallel_inference)
            PARALLEL_INFERENCE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Batch inference runner for deepRetinotopy pipeline"
            echo ""
            echo "Required options:"
            echo "  --freesurfer_base_dir PATH    Base directory containing FreeSurfer subjects"
            echo ""
            echo "Parameter lists (comma-separated or space-separated):"
            echo "  --subject_ids LIST            Subject IDs: comma-separated (e.g., sub-100307,sub-100408)"
            echo "                                or 'all' to auto-detect all subjects"
            echo "                                or path to txt file (one subject per line)"
            echo "                                Note: 'sub-' prefix is auto-added if missing"
            echo "                                (default: sub-157336)"
            echo "  --model_types LIST            Model types: baseline,transolver_optionA,... (default: baseline,transolver_optionA)"
            echo "  --hemispheres LIST            Hemispheres: lh,rh (default: lh,rh)"
            echo "  --predictions LIST            Predictions: eccentricity,polarAngle,pRFsize (default: all three)"
            echo "  --seeds LIST                  Seeds: 1,2,3,... (default: 1)"
            echo ""
            echo "Common options (applied to all runs):"
            echo "  --hcp_surface_dir PATH       Path to HCP surface templates (default: surface)"
            echo "  --myelination True|False      Whether myelination was used (default: False)"
            echo "  --output_dir PATH            Output directory (default: in-place)"
            echo "  --skip_myelin                Skip myelin map generation"
            echo "  --skip_preprocessing         Skip preprocessing steps"
            echo "  --skip_native_conversion     Skip fsaverage to native conversion"
            echo "  --n_jobs N                   Number of parallel jobs (default: auto-detect)"
            echo ""
            echo "Docker options:"
            echo "  --docker_image IMAGE         Docker image to use (default: vnmd/deepretinotopy_1.0.18:latest)"
            echo ""
            echo "Wandb options:"
            echo "  --use_wandb True|False       Enable Wandb logging (default: false)"
            echo "  --wandb_api_key KEY         Wandb API key"
            echo ""
            echo "Execution options:"
            echo "  --continue_on_error          Continue to next combination if error occurs"
            echo "  --dry_run                    Print commands without executing"
            echo "  --log_dir PATH              Directory to save logs for each run"
            echo "  --force_recreate_container   Force recreation of Docker container (useful when mounts change)"
            echo "  --parallel_inference N       Number of subjects to process in parallel (default: 1, recommended: 8-10)"
            echo "                               Note: Each subject's combinations are processed sequentially"
            echo ""
            echo "Examples:"
            echo "  # Run with default parameters"
            echo "  $0 --freesurfer_base_dir ./HCP_freesurfer/proj-5dceb267c4ae281d2c297b92"
            echo ""
            echo "  # Run all subjects in the FreeSurfer directory with parallel processing"
            echo "  $0 --freesurfer_base_dir /mnt/storage/junb/proj-xxx \\"
            echo "     --subject_ids all \\"
            echo "     --model_types baseline \\"
            echo "     --predictions polarAngle \\"
            echo "     --parallel_inference 8 \\"
            echo "     --continue_on_error"
            echo ""
            echo "  # Run specific combinations"
            echo "  $0 --freesurfer_base_dir ./HCP_freesurfer/proj-5dceb267c4ae281d2c297b92 \\"
            echo "     --subject_ids sub-157336,sub-157337 \\"
            echo "     --model_types baseline,transolver_optionC \\"
            echo "     --hemispheres lh \\"
            echo "     --predictions polarAngle \\"
            echo "     --seeds 1,2,3"
            echo ""
            echo "  # Run subjects from txt file (sub- prefix auto-added if missing)"
            echo "  $0 --freesurfer_base_dir /mnt/storage/junb/proj-xxx \\"
            echo "     --subject_ids Retinotopy/data/subject_splits/seed0/test_subjects.txt \\"
            echo "     --model_types baseline \\"
            echo "     --hemispheres lh,rh \\"
            echo "     --predictions polarAngle,eccentricity \\"
            echo "     --parallel_inference 8"
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
if [ -z "$FREESURFER_BASE_DIR" ]; then
    echo "ERROR: --freesurfer_base_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

# Convert to absolute path and verify it exists
if [[ "$FREESURFER_BASE_DIR" != /* ]]; then
    FREESURFER_BASE_DIR="$(cd "$(dirname "$FREESURFER_BASE_DIR")" && pwd)/$(basename "$FREESURFER_BASE_DIR")"
fi

if [ ! -d "$FREESURFER_BASE_DIR" ]; then
    echo "ERROR: FreeSurfer base directory does not exist: $FREESURFER_BASE_DIR"
    exit 1
fi

echo "Using FreeSurfer base directory: $FREESURFER_BASE_DIR"

# Helper function to find FreeSurfer directory for a subject (defined early for auto-detection)
find_freesurfer_dir() {
    local subject_id="$1"
    local base_dir="$2"
    
    # Normalize base_dir path (remove trailing slash)
    base_dir="${base_dir%/}"
    
    # First try: look for dt-neuro-freesurfer.* subdirectory
    local freesurfer_subdir=$(find "$base_dir/$subject_id/" -maxdepth 1 -type d -name "dt-neuro-freesurfer.*" 2>/dev/null | head -n1)
    if [[ ! -z "$freesurfer_subdir" ]]; then
        echo "$freesurfer_subdir"
        return 0
    fi
    
    # Second try: check if the subject directory itself is a valid FreeSurfer directory
    if [ -d "$base_dir/$subject_id/surf" ]; then
        echo "$base_dir/$subject_id"
        return 0
    fi
    
    # Third try: maybe the base_dir directly contains subject directories with surf subdirs
    if [ -d "$base_dir/$subject_id/surf" ]; then
        echo "$base_dir/$subject_id"
        return 0
    fi
    
    echo ""
    return 1
}

# Auto-detect all subjects if requested
if [ "${SUBJECT_IDS[0]}" = "all" ]; then
    echo ""
    echo "Auto-detecting all subjects in FreeSurfer base directory..."
    
    # Find all subject directories (typically named sub-* or similar)
    DETECTED_SUBJECTS=()
    for item in "$FREESURFER_BASE_DIR"/*; do
        if [ -d "$item" ]; then
            subject_name=$(basename "$item")
            # Check if this looks like a valid FreeSurfer subject directory
            # Try to find a FreeSurfer directory for this subject
            test_fs_dir=$(find_freesurfer_dir "$subject_name" "$FREESURFER_BASE_DIR" 2>/dev/null)
            if [ ! -z "$test_fs_dir" ]; then
                DETECTED_SUBJECTS+=("$subject_name")
            fi
        fi
    done
    
    if [ ${#DETECTED_SUBJECTS[@]} -eq 0 ]; then
        echo "ERROR: No valid subjects found in $FREESURFER_BASE_DIR"
        exit 1
    fi
    
    # Replace SUBJECT_IDS with detected subjects
    SUBJECT_IDS=("${DETECTED_SUBJECTS[@]}")
    
    # Create timestamp for log file
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SUBJECT_LIST_FILE="processed_subjects_${TIMESTAMP}.txt"
    
    # Save detected subjects to file
    echo "# Subjects processed on $(date)" > "$SUBJECT_LIST_FILE"
    echo "# FreeSurfer base directory: $FREESURFER_BASE_DIR" >> "$SUBJECT_LIST_FILE"
    echo "# Total subjects: ${#SUBJECT_IDS[@]}" >> "$SUBJECT_LIST_FILE"
    echo "" >> "$SUBJECT_LIST_FILE"
    for subject in "${SUBJECT_IDS[@]}"; do
        echo "$subject" >> "$SUBJECT_LIST_FILE"
    done
    
    echo "Found ${#SUBJECT_IDS[@]} subjects: ${SUBJECT_IDS[@]}"
    echo "Subject list saved to: $SUBJECT_LIST_FILE"
    echo ""
fi

# Check if main script exists
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "ERROR: Main script not found: $MAIN_SCRIPT"
    exit 1
fi

# Make main script executable
chmod +x "$MAIN_SCRIPT"

# Create log directory if specified
if [ ! -z "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    echo "Log directory: $LOG_DIR"
fi

# Calculate total number of combinations
TOTAL_COMBINATIONS=$((${#SUBJECT_IDS[@]} * ${#MODEL_TYPES[@]} * ${#HEMISPHERES[@]} * ${#PREDICTIONS[@]} * ${#SEEDS[@]}))

# Display configuration
echo ""
echo "=========================================="
echo "Batch Inference Configuration"
echo "=========================================="
echo "  FreeSurfer Base Dir: $FREESURFER_BASE_DIR"
echo "  HCP Surface Dir: $HCP_SURFACE_DIR"
# Display subjects (show count if many, otherwise list them)
if [ ${#SUBJECT_IDS[@]} -le 5 ]; then
    echo "  Subject IDs: ${SUBJECT_IDS[@]}"
else
    echo "  Subject IDs: ${#SUBJECT_IDS[@]} subjects (see $SUBJECT_LIST_FILE)"
fi
echo "  Model Types: ${MODEL_TYPES[@]}"
echo "  Hemispheres: ${HEMISPHERES[@]}"
echo "  Predictions: ${PREDICTIONS[@]}"
echo "  Seeds: ${SEEDS[@]}"
echo "  Myelination: $MYELINATION"
echo "  Total Combinations: $TOTAL_COMBINATIONS"
echo "  Parallel Inference: $PARALLEL_INFERENCE"
echo "  Continue on Error: $CONTINUE_ON_ERROR"
echo "  Dry Run: $DRY_RUN"
if [ ! -z "$LOG_DIR" ]; then
    echo "  Log Directory: $LOG_DIR"
fi
echo "=========================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - Commands will be printed but not executed"
    echo ""
fi

# Counters for tracking progress
CURRENT_COMBINATION=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
FAILED_COMBINATIONS=()

# Start time tracking
START_TIME=$(date +%s)

# Create temporary directory for parallel execution tracking
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Function to process all combinations for a single subject
process_subject() {
    local SUBJECT_ID="$1"
    local FREESURFER_DIR="$2"
    local SUBJECT_NUM="$3"
    local TOTAL_SUBJECTS="$4"
    
    local SUBJECT_SUCCESS=0
    local SUBJECT_FAILED=0
    local COMBINATIONS_FOR_SUBJECT=$((${#MODEL_TYPES[@]} * ${#HEMISPHERES[@]} * ${#PREDICTIONS[@]} * ${#SEEDS[@]}))
    local COMBO_NUM=0
    
    echo ""
    echo "=========================================="
    echo "Processing Subject [$SUBJECT_NUM/$TOTAL_SUBJECTS]: $SUBJECT_ID"
    echo "  FreeSurfer Dir: $FREESURFER_DIR"
    echo "  Combinations: $COMBINATIONS_FOR_SUBJECT"
    echo "=========================================="
    
    # Process all combinations for this subject sequentially
    for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
        for HEMISPHERE in "${HEMISPHERES[@]}"; do
            for PREDICTION in "${PREDICTIONS[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    ((COMBO_NUM++))
                    
                    # Build log file path if log directory is specified
                    LOG_FILE=""
                    if [ ! -z "$LOG_DIR" ]; then
                        LOG_FILE="$LOG_DIR/${SUBJECT_ID}_${MODEL_TYPE}_${HEMISPHERE}_${PREDICTION}_seed${SEED}.log"
                    fi
                    
                    # Display progress
                    echo ""
                    echo "[$SUBJECT_NUM/$TOTAL_SUBJECTS | $COMBO_NUM/$COMBINATIONS_FOR_SUBJECT] Processing:"
                    echo "  Subject: $SUBJECT_ID"
                    echo "  Model: $MODEL_TYPE | Hemisphere: $HEMISPHERE"
                    echo "  Prediction: $PREDICTION | Seed: $SEED"
                    
                    # Build command
                    CMD="$MAIN_SCRIPT"
                    CMD="$CMD --freesurfer_dir \"$FREESURFER_DIR\""
                    CMD="$CMD --subject_id \"$SUBJECT_ID\""
                    CMD="$CMD --hemisphere \"$HEMISPHERE\""
                    CMD="$CMD --model_type \"$MODEL_TYPE\""
                    CMD="$CMD --prediction \"$PREDICTION\""
                    CMD="$CMD --seed \"$SEED\""
                    CMD="$CMD --myelination \"$MYELINATION\""
                    CMD="$CMD --hcp_surface_dir \"$HCP_SURFACE_DIR\""
                    CMD="$CMD --docker_image \"$DOCKER_IMAGE\""
                    CMD="$CMD --n_jobs $N_JOBS"
                    
                    if [ "$SKIP_MYELIN" = true ]; then
                        CMD="$CMD --skip_myelin"
                    fi
                    
                    if [ "$SKIP_PREPROCESSING" = true ]; then
                        CMD="$CMD --skip_preprocessing"
                    fi
                    
                    if [ "$SKIP_NATIVE_CONVERSION" = true ]; then
                        CMD="$CMD --skip_native_conversion"
                    fi
                    
                    if [ ! -z "$OUTPUT_DIR" ]; then
                        CMD="$CMD --output_dir \"$OUTPUT_DIR\""
                    fi
                    
                    if [ "$USE_WANDB" = "true" ]; then
                        CMD="$CMD --use_wandb \"$USE_WANDB\""
                    fi
                    
                    if [ ! -z "$WANDB_API_KEY" ]; then
                        CMD="$CMD --wandb_api_key \"$WANDB_API_KEY\""
                    fi
                    
                    if [ "$FORCE_RECREATE_CONTAINER" = true ]; then
                        CMD="$CMD --force_recreate_container"
                    fi
                    
                    # Execute or print command
                    local EXIT_CODE=0
                    if [ "$DRY_RUN" = true ]; then
                        echo "Command: $CMD"
                        if [ ! -z "$LOG_FILE" ]; then
                            echo "  Log file: $LOG_FILE"
                        fi
                    else
                        # Execute command with optional logging
                        if [ ! -z "$LOG_FILE" ]; then
                            eval "$CMD" > "$LOG_FILE" 2>&1
                            EXIT_CODE=$?
                        else
                            eval "$CMD"
                            EXIT_CODE=$?
                        fi
                        
                        # Save result
                        local COMBO_KEY="${SUBJECT_ID}:${MODEL_TYPE}:${HEMISPHERE}:${PREDICTION}:${SEED}"
                        if [ $EXIT_CODE -eq 0 ]; then
                            echo "success:$COMBO_KEY" >> "$TEMP_DIR/results_${SUBJECT_ID}.txt"
                            ((SUBJECT_SUCCESS++))
                            echo "  ✓ Success"
                        else
                            echo "failed:$COMBO_KEY" >> "$TEMP_DIR/results_${SUBJECT_ID}.txt"
                            ((SUBJECT_FAILED++))
                            echo "  ✗ Failed (exit code: $EXIT_CODE)"
                            
                            if [ "$CONTINUE_ON_ERROR" = false ]; then
                                echo ""
                                echo "ERROR: Execution failed. Use --continue_on_error to continue."
                                return $EXIT_CODE
                            fi
                        fi
                    fi
                done
            done
        done
    done
    
    # Subject summary
    echo ""
    echo "=========================================="
    echo "Subject $SUBJECT_ID Complete:"
    echo "  Success: $SUBJECT_SUCCESS"
    echo "  Failed: $SUBJECT_FAILED"
    echo "=========================================="
    
    return 0
}

# Export function and variables for parallel execution
export -f process_subject find_freesurfer_dir
export MAIN_SCRIPT MYELINATION HCP_SURFACE_DIR DOCKER_IMAGE N_JOBS
export SKIP_MYELIN SKIP_PREPROCESSING SKIP_NATIVE_CONVERSION OUTPUT_DIR
export USE_WANDB WANDB_API_KEY DRY_RUN LOG_DIR TEMP_DIR
export MODEL_TYPES HEMISPHERES PREDICTIONS SEEDS CONTINUE_ON_ERROR
export FREESURFER_BASE_DIR

# Important: Force container recreation on first run to set up base directory mount
# After that, reuse the same container for all subjects
if [ "${SUBJECT_IDS[0]}" != "all" ] || [ ${#SUBJECT_IDS[@]} -gt 1 ]; then
    # Multiple subjects will be processed - force recreate once at the start
    FORCE_RECREATE_CONTAINER=true
fi
export FORCE_RECREATE_CONTAINER

# Build list of subjects to process
SUBJECT_LIST=()
SUBJECT_FS_DIRS=()

for SUBJECT_ID in "${SUBJECT_IDS[@]}"; do
    # Find FreeSurfer directory for this subject
    FREESURFER_DIR=$(find_freesurfer_dir "$SUBJECT_ID" "$FREESURFER_BASE_DIR")
    if [ -z "$FREESURFER_DIR" ]; then
        echo "WARNING: FreeSurfer directory not found for subject $SUBJECT_ID, skipping..."
        echo "failed:${SUBJECT_ID}:*:*:*:*" >> "$TEMP_DIR/results_${SUBJECT_ID}.txt"
        if [ "$CONTINUE_ON_ERROR" = false ]; then
            echo "ERROR: Stopping execution. Use --continue_on_error to continue with remaining subjects."
            exit 1
        fi
        continue
    fi
    
    echo "Found FreeSurfer directory for $SUBJECT_ID: $FREESURFER_DIR"
    SUBJECT_LIST+=("$SUBJECT_ID")
    SUBJECT_FS_DIRS+=("$FREESURFER_DIR")
done

TOTAL_SUBJECTS=${#SUBJECT_LIST[@]}

echo ""
echo "=========================================="
echo "FreeSurfer Base Directory Mount Strategy"
echo "=========================================="
echo "  Base Directory: $FREESURFER_BASE_DIR"
echo "  Mount Strategy: Single base directory mount (shared across all subjects)"
echo "  This allows container reuse and faster parallel processing"
echo "=========================================="

echo ""
echo "=========================================="
echo "Starting Subject-Wise Parallel Execution"
echo "=========================================="
echo "  Total Subjects: $TOTAL_SUBJECTS"
echo "  Parallel Subjects: $PARALLEL_INFERENCE"
echo "  Combinations per Subject: $((${#MODEL_TYPES[@]} * ${#HEMISPHERES[@]} * ${#PREDICTIONS[@]} * ${#SEEDS[@]}))"
echo "  Total Combinations: $TOTAL_COMBINATIONS"
echo "=========================================="
echo ""

# Execute subjects in parallel or sequentially
if [ "$PARALLEL_INFERENCE" -gt 1 ] && [ "$TOTAL_SUBJECTS" -gt 1 ]; then
    # Check if GNU parallel is available
    if command -v parallel &> /dev/null; then
        echo "Using GNU Parallel for subject-wise execution..."
        # Create job list for GNU parallel
        for i in "${!SUBJECT_LIST[@]}"; do
            echo "${SUBJECT_LIST[$i]}|${SUBJECT_FS_DIRS[$i]}|$((i+1))|$TOTAL_SUBJECTS"
        done | parallel -j "$PARALLEL_INFERENCE" --colsep '|' \
            process_subject {1} {2} {3} {4}
    else
        echo "GNU Parallel not found. Using bash background jobs for subjects..."
        
        # First, create container with base directory mount using first subject
        echo "Initializing container with base directory mount..."
        FIRST_SUBJECT_ID="${SUBJECT_LIST[0]}"
        FIRST_FREESURFER_DIR="${SUBJECT_FS_DIRS[0]}"
        export FORCE_RECREATE_CONTAINER=true
        process_subject "$FIRST_SUBJECT_ID" "$FIRST_FREESURFER_DIR" "1" "$TOTAL_SUBJECTS"
        FIRST_EXIT=$?
        
        # Now container is ready with base mount, process remaining subjects in parallel
        export FORCE_RECREATE_CONTAINER=false
        RUNNING_JOBS=0
        
        for i in "${!SUBJECT_LIST[@]}"; do
            # Skip first subject (already processed)
            if [ $i -eq 0 ]; then
                continue
            fi
            
            SUBJECT_ID="${SUBJECT_LIST[$i]}"
            FREESURFER_DIR="${SUBJECT_FS_DIRS[$i]}"
            SUBJECT_NUM=$((i+1))
            
            # Wait if we have too many running jobs
            while [ $RUNNING_JOBS -ge $((PARALLEL_INFERENCE - 1)) ]; do
                sleep 1
                # Count running background jobs
                RUNNING_JOBS=$(jobs -r | wc -l)
            done
            
            # Start subject processing in background
            process_subject "$SUBJECT_ID" "$FREESURFER_DIR" "$SUBJECT_NUM" "$TOTAL_SUBJECTS" &
            ((RUNNING_JOBS++))
            
            # Small delay to avoid race conditions
            sleep 0.5
        done
        
        # Wait for all background jobs to complete
        echo ""
        echo "Waiting for all subjects to complete..."
        wait
    fi
else
    # Sequential execution (original behavior)
    echo "Running sequentially (parallel_inference=1 or only 1 subject)..."
    for i in "${!SUBJECT_LIST[@]}"; do
        SUBJECT_ID="${SUBJECT_LIST[$i]}"
        FREESURFER_DIR="${SUBJECT_FS_DIRS[$i]}"
        SUBJECT_NUM=$((i+1))
        
        # Only force recreate container for the first subject
        # After that, reuse the same container with the base directory mount
        if [ $i -gt 0 ]; then
            export FORCE_RECREATE_CONTAINER=false
        fi
        
        process_subject "$SUBJECT_ID" "$FREESURFER_DIR" "$SUBJECT_NUM" "$TOTAL_SUBJECTS"
        
        if [ $? -ne 0 ] && [ "$CONTINUE_ON_ERROR" = false ]; then
            echo ""
            echo "ERROR: Subject processing failed. Use --continue_on_error to continue with remaining subjects."
            exit 1
        fi
    done
fi

# Aggregate results from all subject temp files
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
FAILED_COMBINATIONS=()

for result_file in "$TEMP_DIR"/results_*.txt; do
    if [ -f "$result_file" ]; then
        SUCCESS_COUNT=$(grep -c "^success:" "$result_file" 2>/dev/null || echo 0)
        FAILED_COUNT=$(grep -c "^failed:" "$result_file" 2>/dev/null || echo 0)
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + SUCCESS_COUNT))
        FAILED_RUNS=$((FAILED_RUNS + FAILED_COUNT))
        
        # Extract failed combinations
        while IFS=':' read -r status combo; do
            if [ "$status" = "failed" ]; then
                FAILED_COMBINATIONS+=("$combo")
            fi
        done < "$result_file"
    fi
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_MINUTES=$((ELAPSED_TIME / 60))
ELAPSED_SECONDS=$((ELAPSED_TIME % 60))

# Display summary
echo ""
echo "==============================================="
echo "Batch Inference Summary"
echo "==============================================="
echo "  Total Combinations: $TOTAL_COMBINATIONS"
echo "  Successful Runs: $SUCCESSFUL_RUNS"
echo "  Failed Runs: $FAILED_RUNS"
echo "  Elapsed Time: ${ELAPSED_MINUTES}m ${ELAPSED_SECONDS}s"
if [ $ELAPSED_TIME -gt 0 ]; then
    THROUGHPUT=$(echo "scale=2; $SUCCESSFUL_RUNS * 60 / $ELAPSED_TIME" | bc 2>/dev/null || echo "N/A")
    if [ "$THROUGHPUT" != "N/A" ]; then
        echo "  Throughput: ${THROUGHPUT} jobs/hour"
    fi
fi
echo "  Parallel Processes Used: $PARALLEL_INFERENCE"
if [ ! -z "$SUBJECT_LIST_FILE" ] && [ -f "$SUBJECT_LIST_FILE" ]; then
    echo "  Subject List File: $SUBJECT_LIST_FILE"
fi
echo ""

if [ ${#FAILED_COMBINATIONS[@]} -gt 0 ]; then
    echo "Failed Combinations:"
    for failed in "${FAILED_COMBINATIONS[@]}"; do
        echo "  - $failed"
    done
    echo ""
fi

if [ "$DRY_RUN" = false ]; then
    if [ $FAILED_RUNS -eq 0 ]; then
        echo "✓ All combinations completed successfully!"
        exit 0
    else
        echo "⚠ Some combinations failed. Check logs for details."
        exit 1
    fi
else
    echo "Dry run completed. No commands were executed."
    exit 0
fi
