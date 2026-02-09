#!/bin/bash

# Script to run FreeSurfer preprocessing and inference pipeline using Docker container
# This script sets up Docker environment and runs the full pipeline for FreeSurfer subjects
#
# USAGE EXAMPLES:
#
# 1. Basic usage with default settings (single subject):
#    ./run_deepRetinotopy_freesurfer_with_docker.sh \
#        --freesurfer_dir /path/to/freesurfer/subjects \
#        --hemisphere lh \
#        --checkpoint_path /path/to/checkpoint.pt \
#        --hcp_surface_dir /path/to/surface
#
# 2. Process specific subject with custom model:
#    ./run_deepRetinotopy_freesurfer_with_docker.sh \
#        --freesurfer_dir /mnt/storage/junb/proj-5dceb267c4ae281d2c297b92/ \
#        --subject_id sub-191033 \
#        --hemisphere lh \
#        --model_type transolver_optionC \
#        --prediction eccentricity \
#        --checkpoint_path Models/checkpoints/eccentricity_Left_transolver_optionC/best_model.pt \
#        --hcp_surface_dir surface \
#        --myelination False \
#        --seed 0
#
# 3. Process all subjects in directory with custom output:
#    ./run_deepRetinotopy_freesurfer_with_docker.sh \
#        --freesurfer_dir /path/to/freesurfer/subjects \
#        --hemisphere rh \
#        --checkpoint_path /path/to/checkpoint.pt \
#        --hcp_surface_dir /path/to/surface \
#        --output_dir /path/to/output \
#        --model_type baseline \
#        --prediction polarAngle
#
# 4. Skip preprocessing (already done) and use wb_command for curvature:
#    ./run_deepRetinotopy_freesurfer_with_docker.sh \
#        --freesurfer_dir /path/to/freesurfer/subjects \
#        --hemisphere lh \
#        --checkpoint_path /path/to/checkpoint.pt \
#        --hcp_surface_dir /path/to/surface \
#        --skip_preprocessing \
#        --calculate_curv_using_wb yes
#
# 5. Fast mode with custom Docker image:
#    ./run_deepRetinotopy_freesurfer_with_docker.sh \
#        --freesurfer_dir /path/to/freesurfer/subjects \
#        --hemisphere lh \
#        --checkpoint_path /path/to/checkpoint.pt \
#        --hcp_surface_dir /path/to/surface \
#        --fast yes \
#        --docker_image custom_image:tag
#
# NOTES:
# - If --freesurfer_dir is not specified, default is used: /mnt/storage/junb/proj-5dceb267c4ae281d2c297b92/
# - If --subject_id is provided, the script will automatically search for dt-neuro-freesurfer.* 
#   subdirectory within $FREESURFER_DIR/$SUBJECT_ID/ and use it if found
# - If --output_dir is not specified, results are saved in-place within FreeSurfer directory structure
# - The script processes three steps: (1) native to fsaverage, (2) inference, (3) fsaverage to native
# - Use --skip_preprocessing or --skip_native_conversion to skip specific steps if already done

# Set base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../" && pwd)"
cd "$PROJECT_ROOT"

# Docker settings
DOCKER_IMAGE=${DOCKER_IMAGE:-"vnmd/deepretinotopy_1.0.18:latest"}  # Change to your Docker image name
CONTAINER_NAME="deepretinotopy_pipeline_all_subjects"
USE_GPU=${USE_GPU:-"true"}  # Set to "false" to disable GPU

# Default parameters

SUBJECT_ID="sub-191033" #"sub-157336"
# Default FreeSurfer subjects directory (base directory containing all subjects)
FREESURFER_DIR="/mnt/storage/junb/proj-5dceb267c4ae281d2c297b92/"

HEMISPHERE="rh"
MODEL_TYPE="baseline" # "baseline" "transolver_optionA" "transolver_optionB" "transolver_optionC"
PREDICTION="polarAngle" # "eccentricity"  "pRFsize" "eccentricity"
MYELINATION="False"
SEED="0"  # Seed name (optional, will be added to checkpoint folder name if provided)
CHECKPOINT_PATH=""  # Will be set automatically or via --checkpoint_path argument
HCP_SURFACE_DIR="surface"
T1_PATH=""
T2_PATH=""
OUTPUT_DIR=""
SKIP_MYELIN=false
SKIP_PREPROCESSING=false
SKIP_NATIVE_CONVERSION=false
FORCE_RECREATE_CONTAINER=false
FAST_MODE="no"  # "yes" or "no" - fast mode for midthickness surface generation
CALCULATE_CURV_USING_WB="no"  # "yes" or "no" - use wb_command for curvature calculation
N_JOBS=$(($(nproc) - 1))
[ $N_JOBS -lt 1 ] && N_JOBS=1

# Wandb settings (optional)
USE_WANDB=${USE_WANDB:-"false"}  # Set to "true" to enable, "false" to disable
WANDB_API_KEY=${WANDB_API_KEY:-""}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --freesurfer_dir)
            FREESURFER_DIR="$2"
            shift 2
            ;;
        --subject_id)
            SUBJECT_ID="$2"
            shift 2
            ;;
        --hemisphere)
            HEMISPHERE="$2"
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
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --myelination)
            MYELINATION="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --hcp_surface_dir)
            HCP_SURFACE_DIR="$2"
            shift 2
            ;;
        --t1_path)
            T1_PATH="$2"
            shift 2
            ;;
        --t2_path)
            T2_PATH="$2"
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
        --force_recreate_container)
            FORCE_RECREATE_CONTAINER=true
            shift
            ;;
        --fast)
            FAST_MODE="$2"
            case "$FAST_MODE" in
                'yes'|'no') ;;
                *) echo "Invalid fast mode argument: $FAST_MODE. Must be 'yes' or 'no'"; exit 1;;
            esac
            shift 2
            ;;
        --calculate_curv_using_wb)
            CALCULATE_CURV_USING_WB="$2"
            case "$CALCULATE_CURV_USING_WB" in
                'yes'|'no') ;;
                *) echo "Invalid calculate_curv_using_wb argument: $CALCULATE_CURV_USING_WB. Must be 'yes' or 'no'"; exit 1;;
            esac
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required options:"
            echo "  --freesurfer_dir PATH       Path to FreeSurfer subjects directory (base directory containing all subjects)"
            echo "                             Default: /mnt/storage/junb/proj-5dceb267c4ae281d2c297b92/"
            echo "  --hemisphere lh|rh         Hemisphere to process"
            echo "  --checkpoint_path PATH     Path to pre-trained model checkpoint"
            echo "  --hcp_surface_dir PATH     Path to HCP surface templates (if not skipping preprocessing)"
            echo ""
            echo "Optional options:"
            echo "  --subject_id ID            Single subject ID (if not provided, processes all)"
            echo "  --model_type TYPE          Model type: baseline, transolver_optionA, transolver_optionB, transolver_optionC (default: baseline)"
            echo "  --prediction TYPE          Prediction type: eccentricity, polarAngle, pRFsize (default: eccentricity)"
            echo "  --myelination True|False   Whether myelination was used during training (default: True)"
            echo "  --seed SEED                Seed name to add to checkpoint folder name (optional, e.g., 0, 1, 2)"
            echo "  --t1_path PATH            Path to T1 image (required for myelin generation)"
            echo "  --t2_path PATH            Path to T2 image (required for myelin generation)"
            echo "  --output_dir PATH         Output directory (default: in-place)"
            echo "  --skip_myelin             Skip myelin map generation step"
            echo "  --skip_preprocessing       Skip preprocessing steps (assume already done)"
            echo "  --skip_native_conversion  Skip fsaverage to native space conversion"
            echo "  --fast yes|no             Fast mode for midthickness surface generation (default: no)"
            echo "                             When 'yes', uses Python script instead of mris_expand, output files have '_fast' suffix"
            echo "  --calculate_curv_using_wb yes|no  Use wb_command for curvature calculation (default: no)"
            echo "                             When 'yes', uses wb_command -surface-curvature instead of mris_curvature"
            echo "  --n_jobs N                Number of parallel jobs (default: auto-detect)"
            echo "  --docker_image IMAGE      Docker image to use (default: vnmd/deepretinotopy_1.0.18:latest)"
            echo "  --use_wandb True|False    Enable Wandb logging (default: false)"
            echo "  --wandb_api_key KEY       Wandb API key"
            echo "  --force_recreate_container Force recreation of Docker container (useful when mounts change)"
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

# Validate FreeSurfer directory exists
if [ ! -d "$FREESURFER_DIR" ]; then
    echo "ERROR: FreeSurfer directory not found: $FREESURFER_DIR"
    exit 1
fi

# If SUBJECT_ID is provided, find dt-neuro-freesurfer.* subdirectory and update FREESURFER_DIR
if [ ! -z "$SUBJECT_ID" ]; then
    SUBJECT_DIR="$FREESURFER_DIR/$SUBJECT_ID"
    if [ ! -d "$SUBJECT_DIR" ]; then
        echo "WARNING: Subject directory not found: $SUBJECT_DIR"
        echo "Continuing anyway (will process all subjects in $FREESURFER_DIR)"
    else
        # Look for dt-neuro-freesurfer.* subdirectory within subject directory
        FREESURFER_SUBDIR=$(find "$SUBJECT_DIR/" -maxdepth 1 -type d -name "dt-neuro-freesurfer.*" | head -n1)
        if [ ! -z "$FREESURFER_SUBDIR" ]; then
            echo "INFO: Found FreeSurfer subdirectory: $FREESURFER_SUBDIR"
            FREESURFER_DIR="$FREESURFER_SUBDIR"
        else
            echo "INFO: No dt-neuro-freesurfer.* subdirectory found in $SUBJECT_DIR"
            echo "INFO: Using subject directory directly: $SUBJECT_DIR"
            FREESURFER_DIR="$SUBJECT_DIR"
        fi
    fi
fi

if [ -z "$HEMISPHERE" ]; then
    echo "ERROR: --hemisphere is required (must be 'lh' or 'rh')"
    exit 1
fi

# Automatically find checkpoint if not provided via --checkpoint_path
if [ -z "$CHECKPOINT_PATH" ]; then
    # First, map HEMISPHERE to the correct name for checkpoint search
    if [[ "${HEMISPHERE,,}" == "lh" ]]; then
        HEMI_CHECK="Left"
    elif [[ "${HEMISPHERE,,}" == "rh" ]]; then
        HEMI_CHECK="Right"
    else
        HEMI_CHECK="${HEMISPHERE}"
    fi

    # Compose directory and filename patterns
    if [[ "${MYELINATION,,}" == "true" || "${MYELINATION}" == "True" || "${MYELINATION}" == "1" ]]; then
        NO_MYELIN_SUFFIX=""
    else
        NO_MYELIN_SUFFIX="_noMyelin"
    fi

    # Convert to short name for prediction if needed
    if [[ "${PREDICTION}" == "eccentricity" ]]; then
        PRED_SHORT="ecc"
    elif [[ "${PREDICTION}" == "polarAngle" ]]; then
        PRED_SHORT="PA"
    elif [[ "${PREDICTION}" == "pRFsize" ]]; then
        PRED_SHORT="size"
    else
        PRED_SHORT="${PREDICTION}"
    fi

    # Compose seed suffix for checkpoint search pattern
    # If seed is provided, add _seed{SEED} suffix (matching train_unified.py convention)
    if [ ! -z "$SEED" ]; then
        SEED_SUFFIX="_seed${SEED}"
    else
        SEED_SUFFIX=""
    fi

    # Compose possible checkpoint search pattern
    DIRNAME="Models/output_wandb/${PREDICTION}_${HEMI_CHECK}_${MODEL_TYPE}${NO_MYELIN_SUFFIX}${SEED_SUFFIX}"
    FILENAME="${PRED_SHORT}_${HEMI_CHECK}_${MODEL_TYPE}${NO_MYELIN_SUFFIX}${SEED_SUFFIX}_best_model_epoch*.pt"

    CHECKPOINT_SEARCH="${DIRNAME}/${FILENAME}"

    # Find the latest matching checkpoint file if exists
    CHECKPOINT_PATH=$(ls -1 ${CHECKPOINT_SEARCH} 2>/dev/null | sort -V | tail -n 1)

    if [[ -z "$CHECKPOINT_PATH" ]]; then
        echo "ERROR: No checkpoint file found with pattern: ${CHECKPOINT_SEARCH}"
        echo "Please provide --checkpoint_path or ensure checkpoint exists in the expected location."
        exit 1
    fi
    echo "INFO: Automatically found checkpoint: $CHECKPOINT_PATH"
fi

if [ "$SKIP_PREPROCESSING" = false ] && [ -z "$HCP_SURFACE_DIR" ]; then
    echo "ERROR: --hcp_surface_dir is required for preprocessing"
    exit 1
fi

if [ "$SKIP_NATIVE_CONVERSION" = false ] && [ -z "$HCP_SURFACE_DIR" ]; then
    echo "ERROR: --hcp_surface_dir is required for native space conversion"
    exit 1
fi

# Convert paths to absolute paths with error handling
# Helper function to convert to absolute path
to_abs_path() {
    local path="$1"
    if [ -z "$path" ]; then
        echo ""
        return 1
    fi
    # If path is relative, make it relative to current directory
    if [[ "$path" != /* ]]; then
        path="$(pwd)/$path"
    fi
    # Use realpath if available, otherwise use readlink -f or just return the path
    if command -v realpath &> /dev/null; then
        realpath "$path" 2>/dev/null || echo "$path"
    elif command -v readlink &> /dev/null; then
        readlink -f "$path" 2>/dev/null || echo "$path"
    else
        echo "$path"
    fi
}

# Helper function to convert absolute path to relative path from PROJECT_ROOT
to_rel_path() {
    local abs_path="$1"
    if [ -z "$abs_path" ]; then
        echo ""
        return 1
    fi
    # Convert PROJECT_ROOT to absolute path
    local project_root_abs=$(to_abs_path "$PROJECT_ROOT")
    # Normalize paths (remove trailing slashes)
    project_root_abs="${project_root_abs%/}"
    abs_path="${abs_path%/}"
    
    # If the path is inside PROJECT_ROOT, return relative path
    if [[ "$abs_path" == "$project_root_abs"* ]]; then
        # If paths are exactly the same, return "."
        if [ "$abs_path" = "$project_root_abs" ]; then
            echo "."
        else
            # Remove PROJECT_ROOT prefix and leading slash
            local rel_path="${abs_path#$project_root_abs/}"
            echo "$rel_path"
        fi
    else
        # Path is outside PROJECT_ROOT, return as-is (will need to handle separately)
        echo "$abs_path"
    fi
}

# Convert all paths to absolute paths and track external mounts
# Array to store external mounts: format "host_path:container_path"
declare -a EXTERNAL_MOUNTS=()

# Helper function to add external mount if path is outside PROJECT_ROOT
# This function modifies EXTERNAL_MOUNTS array and prints the container path to stdout
# Usage: container_path=$(add_external_mount_get_path "$abs_path" "$mount_name" "$mount_as_dir")
add_external_mount_get_path() {
    local abs_path="$1"
    local mount_name="$2"  # e.g., "freesurfer", "t1", "t2"
    local mount_as_dir="${3:-auto}"  # "auto", "true", or "false"
    
    if [ -z "$abs_path" ]; then
        return 1
    fi
    
    local project_root_abs=$(to_abs_path "$PROJECT_ROOT")
    project_root_abs="${project_root_abs%/}"
    abs_path="${abs_path%/}"
    
    # If path is outside PROJECT_ROOT, add to mounts
    if [[ "$abs_path" != "$project_root_abs"* ]]; then
        local container_mount="/mnt/external_${mount_name}"
        
        # Determine if we should mount the directory itself or its parent
        local should_mount_dir=false
        if [ "$mount_as_dir" = "true" ]; then
            should_mount_dir=true
        elif [ "$mount_as_dir" = "auto" ]; then
            # Auto-detect: if path is a directory, mount it directly
            if [ -d "$abs_path" ]; then
                should_mount_dir=true
            fi
        fi
        
        if [ "$should_mount_dir" = true ]; then
            # Mount the directory itself directly
            # Check if this mount already exists (avoid duplicates)
            local mount_exists=false
            for existing_mount in "${EXTERNAL_MOUNTS[@]}"; do
                if [[ "$existing_mount" == "$abs_path:"* ]]; then
                    # Extract container path from existing mount
                    local existing_container="${existing_mount#*:}"
                    container_mount="$existing_container"
                    mount_exists=true
                    break
                fi
            done
            
            # Add mount if it doesn't exist (must be done before echo to ensure it persists)
            if [ "$mount_exists" = false ]; then
                EXTERNAL_MOUNTS+=("$abs_path:$container_mount")
            fi
            
            echo "$container_mount"
        else
            # Mount parent directory (for files)
            local parent_dir=$(dirname "$abs_path")
            
            # Check if this mount already exists (avoid duplicates)
            local mount_exists=false
            for existing_mount in "${EXTERNAL_MOUNTS[@]}"; do
                if [[ "$existing_mount" == "$parent_dir:"* ]]; then
                    # Extract container path from existing mount
                    local existing_container="${existing_mount#*:}"
                    container_mount="$existing_container"
                    mount_exists=true
                    break
                fi
            done
            
            # Add mount if it doesn't exist (must be done before echo to ensure it persists)
            if [ "$mount_exists" = false ]; then
                EXTERNAL_MOUNTS+=("$parent_dir:$container_mount")
            fi
            
            echo "$container_mount/$(basename "$abs_path")"
        fi
        return 0
    fi
    return 1
}

# Wrapper that directly adds external mount and sets a variable
# Usage: add_external_mount "$abs_path" "$mount_name" "result_var" "$mount_as_dir"
add_external_mount() {
    local abs_path="$1"
    local mount_name="$2"
    local result_var="$3"
    local mount_as_dir="${4:-auto}"
    
    if [ -z "$abs_path" ]; then
        return 1
    fi
    
    local project_root_abs=$(to_abs_path "$PROJECT_ROOT")
    project_root_abs="${project_root_abs%/}"
    abs_path="${abs_path%/}"
    
    # If path is outside PROJECT_ROOT, add to mounts
    if [[ "$abs_path" != "$project_root_abs"* ]]; then
        local container_mount="/mnt/external_${mount_name}"
        
        # Determine if we should mount the directory itself or its parent
        local should_mount_dir=false
        if [ "$mount_as_dir" = "true" ]; then
            should_mount_dir=true
        elif [ "$mount_as_dir" = "auto" ]; then
            if [ -d "$abs_path" ]; then
                should_mount_dir=true
            fi
        fi
        
        if [ "$should_mount_dir" = true ]; then
            # Mount the directory itself directly
            local mount_exists=false
            for existing_mount in "${EXTERNAL_MOUNTS[@]}"; do
                if [[ "$existing_mount" == "$abs_path:"* ]]; then
                    local existing_container="${existing_mount#*:}"
                    container_mount="$existing_container"
                    mount_exists=true
                    break
                fi
            done
            
            if [ "$mount_exists" = false ]; then
                EXTERNAL_MOUNTS+=("$abs_path:$container_mount")
            fi
            
            eval "$result_var=\"$container_mount\""
        else
            # Mount parent directory (for files)
            local parent_dir=$(dirname "$abs_path")
            
            local mount_exists=false
            for existing_mount in "${EXTERNAL_MOUNTS[@]}"; do
                if [[ "$existing_mount" == "$parent_dir:"* ]]; then
                    local existing_container="${existing_mount#*:}"
                    container_mount="$existing_container"
                    mount_exists=true
                    break
                fi
            done
            
            if [ "$mount_exists" = false ]; then
                EXTERNAL_MOUNTS+=("$parent_dir:$container_mount")
            fi
            
            eval "$result_var=\"$container_mount/$(basename "$abs_path")\""
        fi
        return 0
    fi
    return 1
}

# Convert FREESURFER_DIR
# Strategy: If processing multiple subjects, mount the base directory once
# and use relative paths for each subject
FREESURFER_DIR_CONTAINER=""
FREESURFER_BASE_MOUNT=""

if [ ! -z "$FREESURFER_DIR" ]; then
    FREESURFER_DIR_ABS=$(to_abs_path "$FREESURFER_DIR")
    if [ -z "$FREESURFER_DIR_ABS" ] || [ ! -d "$FREESURFER_DIR_ABS" ]; then
        echo "ERROR: FreeSurfer directory not found: $FREESURFER_DIR"
        echo "Current directory: $(pwd)"
        exit 1
    fi
    FREESURFER_DIR_REL=$(to_rel_path "$FREESURFER_DIR_ABS")
    
    if [[ "$FREESURFER_DIR_REL" == /* ]]; then
        # Path is outside PROJECT_ROOT
        # Check if this is part of a base directory that should be mounted
        
        # Try to find a common base directory by going up the path
        # Look for pattern: /path/to/base/sub-XXXXX/dt-neuro-freesurfer.*
        FREESURFER_BASE_CANDIDATE=""
        
        # If path contains /sub-*/, extract the base directory
        if [[ "$FREESURFER_DIR_ABS" =~ (.*/)(sub-[^/]+)(/.*) ]]; then
            FREESURFER_BASE_CANDIDATE="${BASH_REMATCH[1]}"
            FREESURFER_BASE_CANDIDATE="${FREESURFER_BASE_CANDIDATE%/}"  # Remove trailing slash
            
            echo "INFO: Detected FreeSurfer base directory: $FREESURFER_BASE_CANDIDATE"
            echo "INFO: Will mount base directory to allow multiple subjects"
            
            # Mount the base directory
            add_external_mount "$FREESURFER_BASE_CANDIDATE" "freesurfer_base" FREESURFER_BASE_MOUNT "auto"
            
            if [ ! -z "$FREESURFER_BASE_MOUNT" ]; then
                # Calculate relative path from base to actual FreeSurfer directory
                RELATIVE_FROM_BASE="${FREESURFER_DIR_ABS#$FREESURFER_BASE_CANDIDATE/}"
                FREESURFER_DIR_CONTAINER="$FREESURFER_BASE_MOUNT/$RELATIVE_FROM_BASE"
                
                echo "INFO: Base directory mounted as: $FREESURFER_BASE_MOUNT"
                echo "INFO: Subject directory will be: $FREESURFER_DIR_CONTAINER"
                
                FREESURFER_DIR="$FREESURFER_DIR_CONTAINER"
            else
                echo "ERROR: Failed to set up external mount for FreeSurfer base directory"
                exit 1
            fi
        else
            # Fallback: mount individual directory
            echo "INFO: Could not detect base directory pattern, mounting individual FreeSurfer directory"
            add_external_mount "$FREESURFER_DIR_ABS" "freesurfer" FREESURFER_DIR_CONTAINER "auto"
            if [ ! -z "$FREESURFER_DIR_CONTAINER" ]; then
                echo "INFO: FreeSurfer directory mounted as: $FREESURFER_DIR_CONTAINER"
                FREESURFER_DIR="$FREESURFER_DIR_CONTAINER"
            else
                echo "ERROR: Failed to set up external mount for FreeSurfer directory"
                exit 1
            fi
        fi
    else
        # Path is inside PROJECT_ROOT, use relative path
        FREESURFER_DIR="$FREESURFER_DIR_REL"
        FREESURFER_DIR_CONTAINER="/workspace/$FREESURFER_DIR"
    fi
fi

# Convert CHECKPOINT_PATH
CHECKPOINT_PATH_CONTAINER=""
if [ ! -z "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_PATH_ABS=$(to_abs_path "$CHECKPOINT_PATH")
    if [ -z "$CHECKPOINT_PATH_ABS" ] || [ ! -f "$CHECKPOINT_PATH_ABS" ]; then
        echo "ERROR: Checkpoint file not found: $CHECKPOINT_PATH"
        echo "Current directory: $(pwd)"
        exit 1
    fi
    CHECKPOINT_PATH_REL=$(to_rel_path "$CHECKPOINT_PATH_ABS")
    if [[ "$CHECKPOINT_PATH_REL" == /* ]]; then
        # Checkpoint outside PROJECT_ROOT - use add_external_mount with file path
        # It will mount the parent directory and return the container path
        add_external_mount "$CHECKPOINT_PATH_ABS" "checkpoint" CHECKPOINT_PATH_CONTAINER "false"
        if [ ! -z "$CHECKPOINT_PATH_CONTAINER" ]; then
            echo "INFO: Checkpoint is outside PROJECT_ROOT, will mount as: $CHECKPOINT_PATH_CONTAINER"
            CHECKPOINT_PATH="$CHECKPOINT_PATH_CONTAINER"
        else
            echo "ERROR: Failed to set up external mount for checkpoint"
            exit 1
        fi
    else
        CHECKPOINT_PATH="$CHECKPOINT_PATH_REL"
        CHECKPOINT_PATH_CONTAINER="/workspace/$CHECKPOINT_PATH"
    fi
fi

# Convert HCP_SURFACE_DIR
HCP_SURFACE_DIR_CONTAINER=""
if [ ! -z "$HCP_SURFACE_DIR" ]; then
    HCP_SURFACE_DIR_ABS=$(to_abs_path "$HCP_SURFACE_DIR")
    if [ -z "$HCP_SURFACE_DIR_ABS" ] || [ ! -d "$HCP_SURFACE_DIR_ABS" ]; then
        echo "ERROR: HCP surface directory not found: $HCP_SURFACE_DIR"
        echo "Current directory: $(pwd)"
        exit 1
    fi
    HCP_SURFACE_DIR_REL=$(to_rel_path "$HCP_SURFACE_DIR_ABS")
    if [[ "$HCP_SURFACE_DIR_REL" == /* ]]; then
        # Path is outside PROJECT_ROOT, add external mount
        add_external_mount "$HCP_SURFACE_DIR_ABS" "hcp_surface" HCP_SURFACE_DIR_CONTAINER "auto"
        if [ ! -z "$HCP_SURFACE_DIR_CONTAINER" ]; then
            echo "INFO: HCP surface directory is outside PROJECT_ROOT, will mount as: $HCP_SURFACE_DIR_CONTAINER"
            HCP_SURFACE_DIR="$HCP_SURFACE_DIR_CONTAINER"
        else
            echo "ERROR: Failed to set up external mount for HCP surface directory"
            exit 1
        fi
    else
        HCP_SURFACE_DIR="$HCP_SURFACE_DIR_REL"
        HCP_SURFACE_DIR_CONTAINER="/workspace/$HCP_SURFACE_DIR"
    fi
fi

# Convert T1_PATH
T1_PATH_CONTAINER=""
if [ ! -z "$T1_PATH" ]; then
    T1_PATH_ABS=$(to_abs_path "$T1_PATH")
    if [ -z "$T1_PATH_ABS" ] || [ ! -f "$T1_PATH_ABS" ]; then
        echo "WARNING: T1 file not found: $T1_PATH (will skip if not needed)"
    else
        T1_PATH_REL=$(to_rel_path "$T1_PATH_ABS")
        if [[ "$T1_PATH_REL" == /* ]]; then
            # Path is outside PROJECT_ROOT, add external mount
            add_external_mount "$T1_PATH_ABS" "t1" T1_PATH_CONTAINER "false"
            if [ ! -z "$T1_PATH_CONTAINER" ]; then
                echo "INFO: T1 path is outside PROJECT_ROOT, will mount as: $T1_PATH_CONTAINER"
                T1_PATH="$T1_PATH_CONTAINER"
            else
                echo "WARNING: Failed to set up external mount for T1, may not be accessible"
            fi
        else
            T1_PATH="/workspace/$T1_PATH_REL"
            T1_PATH_CONTAINER="$T1_PATH"
        fi
    fi
fi

# Convert T2_PATH
T2_PATH_CONTAINER=""
if [ ! -z "$T2_PATH" ]; then
    T2_PATH_ABS=$(to_abs_path "$T2_PATH")
    if [ -z "$T2_PATH_ABS" ] || [ ! -f "$T2_PATH_ABS" ]; then
        echo "WARNING: T2 file not found: $T2_PATH (will skip if not needed)"
    else
        T2_PATH_REL=$(to_rel_path "$T2_PATH_ABS")
        if [[ "$T2_PATH_REL" == /* ]]; then
            # Path is outside PROJECT_ROOT, add external mount
            add_external_mount "$T2_PATH_ABS" "t2" T2_PATH_CONTAINER "false"
            if [ ! -z "$T2_PATH_CONTAINER" ]; then
                echo "INFO: T2 path is outside PROJECT_ROOT, will mount as: $T2_PATH_CONTAINER"
                T2_PATH="$T2_PATH_CONTAINER"
            else
                echo "WARNING: Failed to set up external mount for T2, may not be accessible"
            fi
        else
            T2_PATH="/workspace/$T2_PATH_REL"
            T2_PATH_CONTAINER="$T2_PATH"
        fi
    fi
fi

# Convert OUTPUT_DIR
OUTPUT_DIR_CONTAINER=""
if [ ! -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR_ABS=$(to_abs_path "$OUTPUT_DIR")
    if [ -z "$OUTPUT_DIR_ABS" ]; then
        OUTPUT_DIR_ABS="$(pwd)/$OUTPUT_DIR"
    fi
    mkdir -p "$OUTPUT_DIR_ABS"
    OUTPUT_DIR_REL=$(to_rel_path "$OUTPUT_DIR_ABS")
    if [[ "$OUTPUT_DIR_REL" == /* ]]; then
        # Path is outside PROJECT_ROOT, add external mount
        add_external_mount "$OUTPUT_DIR_ABS" "output" OUTPUT_DIR_CONTAINER "auto"
        if [ ! -z "$OUTPUT_DIR_CONTAINER" ]; then
            echo "INFO: Output directory is outside PROJECT_ROOT, will mount as: $OUTPUT_DIR_CONTAINER"
            OUTPUT_DIR="$OUTPUT_DIR_CONTAINER"
        else
            echo "ERROR: Failed to set up external mount for output directory"
            exit 1
        fi
    else
        OUTPUT_DIR="$OUTPUT_DIR_REL"
        OUTPUT_DIR_CONTAINER="/workspace/$OUTPUT_DIR"
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
NEED_NEW_CONTAINER=false

if [ "$FORCE_RECREATE_CONTAINER" = true ]; then
    # Force recreation was requested
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Force recreation requested. Stopping running container '$CONTAINER_NAME'..."
        docker stop "$CONTAINER_NAME" > /dev/null 2>&1
        docker rm "$CONTAINER_NAME" > /dev/null 2>&1
    elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Force recreation requested. Removing existing container '$CONTAINER_NAME'..."
        docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1
    fi
    NEED_NEW_CONTAINER=true
elif docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    # Container is running - check if it has the required mounts
    if [ ${#EXTERNAL_MOUNTS[@]} -gt 0 ]; then
        # Verify all required mounts match the actual container mounts
        MOUNTS_OK=true
        for mount in "${EXTERNAL_MOUNTS[@]}"; do
            # Extract paths from mount (format: host_path:container_path)
            host_path="${mount%%:*}"
            container_path="${mount#*:}"
            
            # Get actual mount source for this container path
            actual_source=$(docker inspect "$CONTAINER_NAME" --format='{{range .Mounts}}{{if eq .Destination "'"$container_path"'"}}{{.Source}}{{end}}{{end}}')
            
            # Normalize paths for comparison (remove trailing slashes)
            host_path="${host_path%/}"
            actual_source="${actual_source%/}"
            
            # Check if path exists and matches
            if [ -z "$actual_source" ]; then
                echo "Container mount for $container_path not found"
                MOUNTS_OK=false
                break
            elif [ "$actual_source" != "$host_path" ]; then
                echo "Container mount mismatch:"
                echo "  Expected: $host_path -> $container_path"
                echo "  Actual:   $actual_source -> $container_path"
                MOUNTS_OK=false
                break
            fi
        done
        
        if [ "$MOUNTS_OK" = false ]; then
            echo "Container '$CONTAINER_NAME' has incorrect mounts."
            echo "  Removing and recreating container with correct mounts..."
            docker stop "$CONTAINER_NAME" > /dev/null 2>&1
            docker rm "$CONTAINER_NAME" > /dev/null 2>&1
            NEED_NEW_CONTAINER=true
        else
            CONTAINER_RUNNING=true
            echo "Container '$CONTAINER_NAME' is already running with correct mounts. Using existing container."
        fi
    else
        CONTAINER_RUNNING=true
        echo "Container '$CONTAINER_NAME' is already running. Using existing container."
    fi
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    # Container exists but is stopped - check if we need to recreate for mounts
    if [ ${#EXTERNAL_MOUNTS[@]} -gt 0 ]; then
        echo "Container '$CONTAINER_NAME' exists but is stopped. External mounts needed."
        echo "  Removing existing container to recreate with proper mounts..."
        docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1
        NEED_NEW_CONTAINER=true
    else
        echo "Container '$CONTAINER_NAME' exists but is stopped. Starting it..."
        docker start "$CONTAINER_NAME" > /dev/null 2>&1
        CONTAINER_RUNNING=true
    fi
else
    NEED_NEW_CONTAINER=true
fi

if [ "$NEED_NEW_CONTAINER" = true ]; then
    echo "Creating new container '$CONTAINER_NAME'..."
    DOCKER_CMD="docker run -d"
    if [ "$USE_GPU" = "true" ]; then
        DOCKER_CMD="$DOCKER_CMD --gpus all"
    fi
    DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"
    # Mount PROJECT_ROOT as /workspace
    DOCKER_CMD="$DOCKER_CMD -v $PROJECT_ROOT:/workspace"
    DOCKER_CMD="$DOCKER_CMD -w /workspace"
    
    # Add external mounts if any
    if [ ${#EXTERNAL_MOUNTS[@]} -gt 0 ]; then
        echo "  Setting up external mounts:"
        for mount in "${EXTERNAL_MOUNTS[@]}"; do
            DOCKER_CMD="$DOCKER_CMD -v $mount"
            echo "    - $mount"
        done
    fi
    
    # Pass Wandb API key as environment variable if provided
    if [ ! -z "$WANDB_API_KEY" ]; then
        DOCKER_CMD="$DOCKER_CMD -e WANDB_API_KEY=$WANDB_API_KEY"
    fi
    
    DOCKER_CMD="$DOCKER_CMD $DOCKER_IMAGE"
    DOCKER_CMD="$DOCKER_CMD tail -f /dev/null"  # Keep container running
    
    echo "Executing: $DOCKER_CMD"
    if eval "$DOCKER_CMD" > /dev/null 2>&1; then
        # Wait a moment for container to start
        sleep 2
        # Verify container was created and is running
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            CONTAINER_RUNNING=true
            echo "✓ Container '$CONTAINER_NAME' created and started successfully"
        else
            echo "ERROR: Container '$CONTAINER_NAME' was not created successfully"
            echo "Trying to see what went wrong..."
            eval "$DOCKER_CMD"
            exit 1
        fi
    else
        echo "ERROR: Failed to create container '$CONTAINER_NAME'"
        echo "Trying to see what went wrong..."
        eval "$DOCKER_CMD"
        exit 1
    fi
fi

# Final verification that container exists and is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "ERROR: Container '$CONTAINER_NAME' is not running"
    echo "Attempting to start existing container..."
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker start "$CONTAINER_NAME" > /dev/null 2>&1
        sleep 2
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "✓ Container '$CONTAINER_NAME' started successfully"
        else
            echo "ERROR: Failed to start container '$CONTAINER_NAME'"
            echo "Please check Docker logs: docker logs $CONTAINER_NAME"
            exit 1
        fi
    else
        echo "ERROR: Container '$CONTAINER_NAME' does not exist"
        echo "Please run the script again to create the container"
        exit 1
    fi
fi

# Check if FreeSurfer and Connectome Workbench are available in container
echo "Checking for required software in container..."
HAS_FREESURFER=false
HAS_WORKBENCH=false

if docker exec "$CONTAINER_NAME" bash -c "command -v mris_convert &> /dev/null" 2>/dev/null; then
    HAS_FREESURFER=true
    echo "✓ FreeSurfer found"
else
    echo "✗ FreeSurfer not found in container"
    echo "  Warning: FreeSurfer tools may not be available. Preprocessing steps may fail."
fi

if docker exec "$CONTAINER_NAME" bash -c "command -v wb_command &> /dev/null" 2>/dev/null; then
    HAS_WORKBENCH=true
    echo "✓ Connectome Workbench found"
else
    echo "✗ Connectome Workbench not found in container"
    echo "  Warning: Connectome Workbench tools may not be available. Preprocessing steps may fail."
fi

# Install required packages once (wandb, einops, and other dependencies)
echo "Installing required packages (wandb, einops, nibabel, scipy) in container..."
INSTALL_CMD="pip install --quiet --no-cache-dir wandb einops nibabel scipy"
docker exec "$CONTAINER_NAME" bash -c "$INSTALL_CMD" > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "Warning: Failed to install packages. Continuing anyway..."
fi

# Setup Wandb authentication if enabled
if [ "$USE_WANDB" = "true" ]; then
    if [ ! -z "$WANDB_API_KEY" ]; then
        echo "Setting up Wandb authentication in container..."
        docker exec "$CONTAINER_NAME" bash -c "wandb login $WANDB_API_KEY" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ Wandb authentication successful"
        else
            echo "Warning: Wandb authentication failed. You may need to login manually."
            echo "  Run: docker exec -it $CONTAINER_NAME wandb login"
        fi
    else
        echo "Warning: WANDB_API_KEY not set. Wandb may prompt for login."
        echo "  Set it via: export WANDB_API_KEY=your_api_key"
        echo "  Or add it to the script: WANDB_API_KEY=\"your_api_key\""
        echo "  Get your API key from: https://wandb.ai/authorize"
    fi
fi

# Convert hemisphere to proper format for pipeline
if [ "$HEMISPHERE" = "lh" ] || [ "$HEMISPHERE" = "LH" ] || [ "$HEMISPHERE" = "left" ]; then
    HEMISPHERE_SHORT="lh"
    HEMISPHERE_LONG="Left"
elif [ "$HEMISPHERE" = "rh" ] || [ "$HEMISPHERE" = "RH" ] || [ "$HEMISPHERE" = "right" ]; then
    HEMISPHERE_SHORT="rh"
    HEMISPHERE_LONG="Right"
else
    echo "ERROR: Invalid hemisphere: $HEMISPHERE (must be 'lh' or 'rh')"
    exit 1
fi

# Determine container paths for use in commands
# Use container paths (either /workspace/relative or /mnt/external_*)
FREESURFER_DIR_FOR_CMD="$FREESURFER_DIR"
if [[ "$FREESURFER_DIR" != /* ]]; then
    FREESURFER_DIR_FOR_CMD="/workspace/$FREESURFER_DIR"
else
    FREESURFER_DIR_FOR_CMD="$FREESURFER_DIR"
fi

HCP_SURFACE_DIR_FOR_CMD="$HCP_SURFACE_DIR"
if [[ "$HCP_SURFACE_DIR" != /* ]]; then
    HCP_SURFACE_DIR_FOR_CMD="/workspace/$HCP_SURFACE_DIR"
else
    HCP_SURFACE_DIR_FOR_CMD="$HCP_SURFACE_DIR"
fi

CHECKPOINT_PATH_FOR_CMD="$CHECKPOINT_PATH"
if [[ "$CHECKPOINT_PATH" != /* ]]; then
    CHECKPOINT_PATH_FOR_CMD="/workspace/$CHECKPOINT_PATH"
else
    CHECKPOINT_PATH_FOR_CMD="$CHECKPOINT_PATH"
fi

OUTPUT_DIR_FOR_CMD=""
if [ ! -z "$OUTPUT_DIR" ]; then
    if [[ "$OUTPUT_DIR" != /* ]]; then
        OUTPUT_DIR_FOR_CMD="/workspace/$OUTPUT_DIR"
    else
        OUTPUT_DIR_FOR_CMD="$OUTPUT_DIR"
    fi
fi

# Determine working directory (use freesurfer_dir if output_dir not specified)
if [ ! -z "$OUTPUT_DIR_FOR_CMD" ]; then
    WORK_DIR="$OUTPUT_DIR_FOR_CMD"
else
    WORK_DIR="$FREESURFER_DIR_FOR_CMD"
fi

# Build pipeline commands - Step 1: Native to fsaverage conversion
STEP1_CMD=""
if [ "$SKIP_PREPROCESSING" = false ]; then
    STEP1_CMD="cd run_from_freesurfer && ./1_native2fsaverage.sh"
    STEP1_CMD="$STEP1_CMD -s $FREESURFER_DIR_FOR_CMD"
    STEP1_CMD="$STEP1_CMD -t $HCP_SURFACE_DIR_FOR_CMD"
    STEP1_CMD="$STEP1_CMD -h $HEMISPHERE_SHORT"
    STEP1_CMD="$STEP1_CMD -g $FAST_MODE"
    STEP1_CMD="$STEP1_CMD -j $N_JOBS"
    STEP1_CMD="$STEP1_CMD -w $CALCULATE_CURV_USING_WB"
    
    # if [ ! -z "$SUBJECT_ID" ]; then
    #     STEP1_CMD="$STEP1_CMD" -i $SUBJECT_ID"
    # fi
    
    if [ ! -z "$OUTPUT_DIR_FOR_CMD" ]; then
        STEP1_CMD="$STEP1_CMD -o $OUTPUT_DIR_FOR_CMD"
    fi
fi

# Build pipeline commands - Step 2: Inference
STEP2_CMD="python run_from_freesurfer/run_inference_freesurfer.py"
STEP2_CMD="$STEP2_CMD --freesurfer_dir $FREESURFER_DIR_FOR_CMD"
STEP2_CMD="$STEP2_CMD --checkpoint_path $CHECKPOINT_PATH_FOR_CMD"
STEP2_CMD="$STEP2_CMD --model_type $MODEL_TYPE"
STEP2_CMD="$STEP2_CMD --prediction $PREDICTION"
STEP2_CMD="$STEP2_CMD --hemisphere $HEMISPHERE_LONG"
STEP2_CMD="$STEP2_CMD --myelination $MYELINATION"

# if [ ! -z "$SUBJECT_ID" ]; then
#     STEP2_CMD="$STEP2_CMD --subject_id \"$SUBJECT_ID\""
# fi

if [ ! -z "$OUTPUT_DIR_FOR_CMD" ]; then
    STEP2_CMD="$STEP2_CMD --output_dir $OUTPUT_DIR_FOR_CMD"
fi

if [ ! -z "$SEED" ]; then
    STEP2_CMD="$STEP2_CMD --seed $SEED"
fi

# Build pipeline commands - Step 3: Fsaverage to native conversion
STEP3_CMD=""
if [ "$SKIP_NATIVE_CONVERSION" = false ]; then
    STEP3_CMD="cd run_from_freesurfer && ./3_fsaverage2native.sh"
    STEP3_CMD="$STEP3_CMD -s $FREESURFER_DIR_FOR_CMD"
    STEP3_CMD="$STEP3_CMD -t $HCP_SURFACE_DIR_FOR_CMD"
    STEP3_CMD="$STEP3_CMD -h $HEMISPHERE_SHORT"
    STEP3_CMD="$STEP3_CMD -r $PREDICTION"
    STEP3_CMD="$STEP3_CMD -m $MODEL_TYPE"
    STEP3_CMD="$STEP3_CMD -y $MYELINATION"
    STEP3_CMD="$STEP3_CMD -j $N_JOBS"
    
    # Add seed parameter if provided
    if [ ! -z "$SEED" ]; then
        STEP3_CMD="$STEP3_CMD -e $SEED"
    fi
    
    # if [ ! -z "$SUBJECT_ID" ]; then
    #     STEP3_CMD="$STEP3_CMD -i $SUBJECT_ID"
    # fi
    
    if [ ! -z "$OUTPUT_DIR_FOR_CMD" ]; then
        STEP3_CMD="$STEP3_CMD -o $OUTPUT_DIR_FOR_CMD"
    fi
fi

# Display configuration
echo ""
echo "=========================================="
echo "Pipeline Configuration:"
echo "=========================================="
echo "  Docker Image: $DOCKER_IMAGE"
echo "  Container: $CONTAINER_NAME"
echo "  GPU: $USE_GPU"
echo "  Wandb: $USE_WANDB"
if [ "$USE_WANDB" = "true" ]; then
    if [ ! -z "$WANDB_API_KEY" ]; then
        echo "  Wandb API Key: *** (set)"
    else
        echo "  Wandb API Key: Not set (will prompt for login if needed)"
    fi
fi
if [[ "$FREESURFER_DIR" == /* ]]; then
    echo "  FreeSurfer Dir: $FREESURFER_DIR (external mount)"
else
    echo "  FreeSurfer Dir: $FREESURFER_DIR (relative to PROJECT_ROOT)"
fi
echo "  Subject ID: ${SUBJECT_ID:-'All subjects'}"
echo "  Hemisphere: $HEMISPHERE"
echo "  Model Type: $MODEL_TYPE"
echo "  Prediction: $PREDICTION"
echo "  Myelination: $MYELINATION"
if [ ! -z "$SEED" ]; then
    echo "  Seed: $SEED"
fi
if [[ "$CHECKPOINT_PATH" == /* ]]; then
    echo "  Checkpoint: $CHECKPOINT_PATH (external mount)"
else
    echo "  Checkpoint: $CHECKPOINT_PATH (relative to PROJECT_ROOT)"
fi
if [ ! -z "$HCP_SURFACE_DIR" ]; then
    if [[ "$HCP_SURFACE_DIR" == /* ]]; then
        echo "  HCP Surface Dir: $HCP_SURFACE_DIR (external mount)"
    else
        echo "  HCP Surface Dir: $HCP_SURFACE_DIR (relative to PROJECT_ROOT)"
    fi
fi
if [ ! -z "$OUTPUT_DIR" ]; then
    if [[ "$OUTPUT_DIR" == /* ]]; then
        echo "  Output Dir: $OUTPUT_DIR (external mount)"
    else
        echo "  Output Dir: $OUTPUT_DIR (relative to PROJECT_ROOT)"
    fi
fi
if [ ${#EXTERNAL_MOUNTS[@]} -gt 0 ]; then
    echo "  External Mounts:"
    for mount in "${EXTERNAL_MOUNTS[@]}"; do
        echo "    - $mount"
    done
fi
echo "  Skip Myelin: $SKIP_MYELIN"
echo "  Skip Preprocessing: $SKIP_PREPROCESSING"
echo "  Skip Native Conversion: $SKIP_NATIVE_CONVERSION"
echo "  Parallel Jobs: $N_JOBS"
echo "=========================================="
echo ""

# Run the pipeline
echo "Running pipeline in Docker container..."
echo ""

# Step 1: Native to fsaverage conversion (if not skipped)
if [ "$SKIP_PREPROCESSING" = false ]; then
    echo ""
    echo "==============================================="
    echo "[Step 1] Native to fsaverage Conversion"
    echo "==============================================="
    
    if [ ! -z "$WANDB_API_KEY" ]; then
        docker exec -e WANDB_API_KEY="$WANDB_API_KEY" "$CONTAINER_NAME" bash -c "$STEP1_CMD"
    else
        docker exec "$CONTAINER_NAME" bash -c "$STEP1_CMD"
    fi
    
    STEP1_EXIT_CODE=$?
    if [ $STEP1_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Native to fsaverage conversion failed with exit code: $STEP1_EXIT_CODE"
        exit $STEP1_EXIT_CODE
    fi
    
    echo "[Step 1] Completed!"
    echo ""
else
    echo "[Step 1] Skipping preprocessing (assumed already done)"
    echo ""
fi

# Step 2: Inference
echo ""
echo "==============================================="
echo "[Step 2] Running Inference"
echo "==============================================="

if [ ! -z "$WANDB_API_KEY" ]; then
    docker exec -e WANDB_API_KEY="$WANDB_API_KEY" "$CONTAINER_NAME" bash -c "$STEP2_CMD"
else
    docker exec "$CONTAINER_NAME" bash -c "$STEP2_CMD"
fi

STEP2_EXIT_CODE=$?

if [ $STEP2_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Inference failed with exit code: $STEP2_EXIT_CODE"
    exit $STEP2_EXIT_CODE
fi

echo "[Step 2] Completed!"
echo ""

# Step 3: Fsaverage to native conversion (if not skipped)
if [ "$SKIP_NATIVE_CONVERSION" = false ]; then
    echo ""
    echo "==============================================="
    echo "[Step 3] Fsaverage to Native Space Conversion"
    echo "==============================================="
    
    if [ ! -z "$WANDB_API_KEY" ]; then
        docker exec -e WANDB_API_KEY="$WANDB_API_KEY" "$CONTAINER_NAME" bash -c "$STEP3_CMD"
    else
        docker exec "$CONTAINER_NAME" bash -c "$STEP3_CMD"
    fi
    
    STEP3_EXIT_CODE=$?
    if [ $STEP3_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Fsaverage to native conversion failed with exit code: $STEP3_EXIT_CODE"
        exit $STEP3_EXIT_CODE
    fi
    
    echo "[Step 3] Completed!"
    echo ""
else
    echo "[Step 3] Skipping native space conversion (assumed already done)"
    echo ""
fi

echo "==============================================="
echo "Full Pipeline Completed Successfully!"
echo "==============================================="
if [ ! -z "$OUTPUT_DIR" ]; then
    if [[ "$OUTPUT_DIR" == /* ]]; then
        echo "Results are available in: $OUTPUT_DIR"
    else
        echo "Results are available in: $PROJECT_ROOT/$OUTPUT_DIR"
    fi
else
    if [[ "$FREESURFER_DIR" == /* ]]; then
        echo "Results are available in: $FREESURFER_DIR"
    else
        echo "Results are available in: $PROJECT_ROOT/$FREESURFER_DIR"
    fi
fi
echo ""
echo "Note: Container '$CONTAINER_NAME' is still running."
echo "To stop it, run: docker stop $CONTAINER_NAME"
echo "To remove it, run: docker rm -f $CONTAINER_NAME"

