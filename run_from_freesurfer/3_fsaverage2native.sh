#!/usr/bin/env bash

# Script to resample predicted maps from fsaverage space to native space
# This script is called from run_deepRetinotopy_freesurfer_with_docker.sh

# Auto-detect number of cores (leave 1 core free)
auto_cores=$(($(nproc) - 1))
[ $auto_cores -lt 1 ] && auto_cores=1  # Ensure at least 1 core

# Default values
n_jobs=$auto_cores
subject_id=""
output_dir=""
freesurfer_dir=""
hcp_surface_dir=""
hemisphere=""
prediction=""
model_type="baseline"
myelination="False"
seed=""

# Get the directory of the current script
script_dir=$(dirname "$(realpath "$0")")

while getopts s:h:t:r:m:j:i:o:y:e: flag
do
  case "${flag}" in
    s) freesurfer_dir=${OPTARG};;
    t) hcp_surface_dir=${OPTARG};;
    h) hemisphere=${OPTARG};
       case "$hemisphere" in
         lh|rh) ;;
         *) echo "Invalid hemisphere argument: $hemisphere"; exit 1;;
       esac;;
    r) prediction=${OPTARG};
       case "$prediction" in
         'polarAngle'|'eccentricity'|'pRFsize') ;;
         *) echo "Invalid prediction argument: $prediction"; exit 1;;
       esac;;
    m) model_type=${OPTARG};;
    j) n_jobs=${OPTARG};;
    i) subject_id=${OPTARG};;
    o) output_dir=${OPTARG};;
    y) myelination=${OPTARG};;
    e) seed=${OPTARG};;
    ?)
      echo "script usage: $(basename "$0") [-s path to freesurfer dir] [-t path to HCP surfaces] [-h hemisphere] [-r prediction] [-m model_type] [-j number of cores] [-i subject ID] [-o output directory] [-y myelination] [-e seed]" >&2
      exit 1;;
  esac
done

echo "Model Type: $model_type"
echo "Prediction: $prediction"
echo "Hemisphere: $hemisphere"
echo "Myelination: $myelination"

# Determine model name for file matching
MODEL_NAME="model"  # Default model name
if [ "$model_type" != "baseline" ]; then
    MODEL_NAME="$model_type"
fi

# Check if processing single subject or multiple subjects
if [ -n "$subject_id" ]; then
    echo "Processing single subject: $subject_id"
else
    echo "Using $n_jobs parallel jobs for multiple subjects"
fi

# Check output directory setup
if [ -n "$output_dir" ]; then
    echo "Output directory: $output_dir"
    mkdir -p "$output_dir"
else
    echo "Output mode: In-place (within FreeSurfer directory structure)"
fi

# Start total timing
total_start_time=$(date +%s)

# Define the processing function
process_subject_step2() {
    local dirSub=$1
    local hemisphere=$2
    local prediction=$3
    local model_name=$4
    local dirSubs=$5
    local dirHCP=$6
    local script_dir=$7
    local output_dir=$8
    local myelination=$9
    local seed=${10}
    
    echo "=== Processing Step 2 (fsaverage2native) for subject: $dirSub ==="
    
    # Determine input and output paths
    if [ -n "$output_dir" ]; then
        local subject_output_dir="$output_dir/$dirSub"
        local deepret_output_dir="$subject_output_dir/deepRetinotopy"
        local surf_output_dir="$subject_output_dir/surf"
        mkdir -p "$deepret_output_dir"
        echo "[$dirSub] Using custom output directory: $subject_output_dir"
        
        # Construct input filename based on inference output
        MYELIN_SUFFIX=""
        if [[ "${myelination,,}" == "true" ]] || [[ "$myelination" == "True" ]] || [[ "$myelination" == "1" ]]; then
            MYELIN_SUFFIX="_myelin"
        fi
        MODEL_SUFFIX=""
        if [ "$model_type" != "baseline" ]; then
            MODEL_SUFFIX="_$model_type"
        fi
        
        # Try seed subfolder first if seed is provided
        if [ -n "$seed" ]; then
            local input_prediction_file="$deepret_output_dir/seed${seed}/$dirSub.predicted_${prediction}_${hemisphere}${MYELIN_SUFFIX}${MODEL_SUFFIX}.func.gii"
        else
            local input_prediction_file="$deepret_output_dir/$dirSub.predicted_${prediction}_${hemisphere}${MYELIN_SUFFIX}${MODEL_SUFFIX}.func.gii"
        fi
        
        # Fallback to original location if not in custom output
        if [ ! -f "$input_prediction_file" ] && [ -n "$seed" ]; then
            input_prediction_file="$dirSubs/$dirSub/deepRetinotopy/seed${seed}/$dirSub.predicted_${prediction}_${hemisphere}${MYELIN_SUFFIX}${MODEL_SUFFIX}.func.gii"
        fi
        
        # Final fallback without seed folder
        if [ ! -f "$input_prediction_file" ]; then
            input_prediction_file="$dirSubs/$dirSub/deepRetinotopy/$dirSub.predicted_${prediction}_${hemisphere}${MYELIN_SUFFIX}${MODEL_SUFFIX}.func.gii"
        fi
        
        # Surface files for resampling
        local surf_midthickness_32k="$surf_output_dir/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii"
        local surf_midthickness_native="$surf_output_dir/$hemisphere.midthickness.surf.gii"
        local surf_sphere_reg="$surf_output_dir/$hemisphere.sphere.reg.surf.gii"
        
        # Fallback to original locations if not in custom output
        if [ ! -f "$surf_midthickness_32k" ]; then
            surf_midthickness_32k="$dirSubs/$dirSub/surf/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii"
        fi
        if [ ! -f "$surf_midthickness_native" ]; then
            surf_midthickness_native="$dirSubs/$dirSub/surf/$hemisphere.midthickness.surf.gii"
        fi
        if [ ! -f "$surf_sphere_reg" ]; then
            surf_sphere_reg="$dirSubs/$dirSub/surf/$hemisphere.sphere.reg.surf.gii"
        fi
    else
        local deepret_output_dir="$dirSubs/$dirSub/deepRetinotopy"
        local surf_output_dir="$dirSubs/$dirSub/surf"
        
        # Construct input filename
        MYELIN_SUFFIX=""
        if [[ "${myelination,,}" == "true" ]] || [[ "$myelination" == "True" ]] || [[ "$myelination" == "1" ]]; then
            MYELIN_SUFFIX="_myelin"
        fi
        MODEL_SUFFIX=""
        if [ "$model_type" != "baseline" ]; then
            MODEL_SUFFIX="_$model_type"
        fi
        
        # Try seed subfolder first if seed is provided
        if [ -n "$seed" ]; then
            local input_prediction_file="$deepret_output_dir/seed${seed}/$dirSub.predicted_${prediction}_${hemisphere}${MYELIN_SUFFIX}${MODEL_SUFFIX}.func.gii"
        else
            local input_prediction_file="$deepret_output_dir/$dirSub.predicted_${prediction}_${hemisphere}${MYELIN_SUFFIX}${MODEL_SUFFIX}.func.gii"
        fi
        
        # Fallback without seed folder if file not found
        if [ ! -f "$input_prediction_file" ]; then
            input_prediction_file="$deepret_output_dir/$dirSub.predicted_${prediction}_${hemisphere}${MYELIN_SUFFIX}${MODEL_SUFFIX}.func.gii"
        fi
        local surf_midthickness_32k="$dirSubs/$dirSub/surf/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii"
        local surf_midthickness_native="$dirSubs/$dirSub/surf/$hemisphere.midthickness.surf.gii"
        local surf_sphere_reg="$dirSubs/$dirSub/surf/$hemisphere.sphere.reg.surf.gii"
        echo "[$dirSub] Using FreeSurfer directory: $deepret_output_dir"
    fi
    
    if [ ! -f "$input_prediction_file" ]; then
        echo "[$dirSub] ERROR: Predicted map is not available: $input_prediction_file"
        return 1
    fi
    
    # Find ROI label file
    # Try multiple possible locations within current project (Docker container uses /workspace as root)
    local roi_label=""
    local project_root="$script_dir/.."
    
    if [ "$hemisphere" == 'lh' ]; then
        # Try different possible paths within current project
        if [ -f "/workspace/Retinotopy/labels/ROI_WangPlusFovea/ROI.fs_lh.label.gii" ]; then
            roi_label="/workspace/Retinotopy/labels/ROI_WangPlusFovea/ROI.fs_lh.label.gii"
        elif [ -f "$project_root/Retinotopy/labels/ROI_WangPlusFovea/ROI.fs_lh.label.gii" ]; then
            roi_label="$project_root/Retinotopy/labels/ROI_WangPlusFovea/ROI.fs_lh.label.gii"
        else
            echo "[$dirSub] ERROR: ROI label file not found. Searched:"
            echo "[$dirSub]   /workspace/Retinotopy/labels/ROI_WangPlusFovea/ROI.fs_lh.label.gii"
            echo "[$dirSub]   $project_root/Retinotopy/labels/ROI_WangPlusFovea/ROI.fs_lh.label.gii"
            return 1
        fi
    else
        # Try different possible paths within current project
        if [ -f "/workspace/Retinotopy/labels/ROI_WangPlusFovea/ROI.fs_rh.label.gii" ]; then
            roi_label="/workspace/Retinotopy/labels/ROI_WangPlusFovea/ROI.fs_rh.label.gii"
        elif [ -f "$project_root/Retinotopy/labels/ROI_WangPlusFovea/ROI.fs_rh.label.gii" ]; then
            roi_label="$project_root/Retinotopy/labels/ROI_WangPlusFovea/ROI.fs_rh.label.gii"
        else
            echo "[$dirSub] ERROR: ROI label file not found. Searched:"
            echo "[$dirSub]   /workspace/Retinotopy/labels/ROI_WangPlusFovea/ROI.fs_rh.label.gii"
            echo "[$dirSub]   $project_root/Retinotopy/labels/ROI_WangPlusFovea/ROI.fs_rh.label.gii"
            return 1
        fi
    fi
    
    echo "[$dirSub] Using ROI label: $roi_label"
    
    # HCP template sphere
    local hcp_sphere=""
    if [ "$hemisphere" == 'lh' ]; then
        hcp_sphere="$dirHCP/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii"
    else
        hcp_sphere="$dirHCP/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii"
    fi
    
    if [ ! -f "$hcp_sphere" ]; then
        echo "[$dirSub] ERROR: HCP sphere template not found: $hcp_sphere"
        return 1
    fi
    
    start_time=$(date +%s)
    
    # Resample ROI from fsaverage to native space
    echo "[$dirSub] Resampling ROI from fsaverage to native space..."
    wb_command -label-resample "$roi_label" \
        "$hcp_sphere" \
        "$surf_sphere_reg" ADAP_BARY_AREA \
        "$deepret_output_dir/$dirSub.ROI.$hemisphere.native.label.gii" \
        -area-surfs "$surf_midthickness_32k" "$surf_midthickness_native"
    
    if [ $? -ne 0 ]; then
        echo "[$dirSub] ERROR: Failed to resample ROI"
        return 1
    fi
    
    # Resample predicted map from fsaverage to native space
    echo "[$dirSub] Resampling predicted map from fsaverage to native space..."
    if [ -n "$seed" ]; then
        local output_native="$deepret_output_dir/seed${seed}/$dirSub.predicted_${prediction}_${model_name}.$hemisphere.native.func.gii"
    else
        local output_native="$deepret_output_dir/$dirSub.predicted_${prediction}_${model_name}.$hemisphere.native.func.gii"
    fi
    
    wb_command -metric-resample "$input_prediction_file" \
        "$hcp_sphere" \
        "$surf_sphere_reg" ADAP_BARY_AREA \
        "$output_native" \
        -area-surfs "$surf_midthickness_32k" "$surf_midthickness_native" \
        -current-roi "$roi_label"
    
    if [ $? -ne 0 ]; then
        echo "[$dirSub] ERROR: Failed to resample predicted map"
        return 1
    fi
    
    # Transform polar angle for left hemisphere
    if [ "$prediction" == "polarAngle" ] && [ "$hemisphere" == "lh" ]; then
        echo "[$dirSub] Transforming polar angle map..."
        
        # Create file with expected filename for transform script
        local transform_input="$deepret_output_dir/$dirSub.predicted_polarAngle_${model_name}.lh.native.func.gii"
        cp "$output_native" "$transform_input"
        
        # Find transform script (try multiple possible locations within current project)
        local transform_script=""
        if [ -f "/workspace/utils/transform_polarangle_lh.py" ]; then
            transform_script="/workspace/utils/transform_polarangle_lh.py"
        elif [ -f "$project_root/utils/transform_polarangle_lh.py" ]; then
            transform_script="$project_root/utils/transform_polarangle_lh.py"
        elif [ -f "/workspace/run_from_freesurfer/transform_polarangle_lh.py" ]; then
            transform_script="/workspace/run_from_freesurfer/transform_polarangle_lh.py"
        elif [ -f "$script_dir/transform_polarangle_lh.py" ]; then
            transform_script="$script_dir/transform_polarangle_lh.py"
        fi
        
        if [ -n "$transform_script" ]; then
            # Ensure path ends with /
            local transform_path="$deepret_output_dir"
            if [ "${transform_path: -1}" != "/" ]; then
                transform_path="$transform_path/"
            fi
            
            echo "[$dirSub] Using transform script: $transform_script"
            python "$transform_script" --path "$transform_path" --model "$model_name"
            
            if [ $? -eq 0 ]; then
                # Copy transformed file back to original output filename
                cp "$transform_input" "$output_native"
                echo "[$dirSub] Polar angle transformation completed"
            else
                echo "[$dirSub] WARNING: Polar angle transformation failed"
            fi
        else
            echo "[$dirSub] WARNING: transform_polarangle_lh.py not found, skipping polar angle transformation"
            echo "[$dirSub]   Searched: /workspace/utils/transform_polarangle_lh.py"
            echo "[$dirSub]   Searched: $project_root/utils/transform_polarangle_lh.py"
            echo "[$dirSub]   Searched: /workspace/run_from_freesurfer/transform_polarangle_lh.py"
            echo "[$dirSub]   Searched: $script_dir/transform_polarangle_lh.py"
        fi
    fi
    
    # Apply mask to predicted map
    echo "[$dirSub] Applying mask to predicted map..."
    wb_command -metric-mask "$output_native" \
        "$deepret_output_dir/$dirSub.ROI.$hemisphere.native.label.gii" \
        "$output_native"
    
    if [ $? -ne 0 ]; then
        echo "[$dirSub] WARNING: Failed to apply mask"
    fi
    
    end_time=$(date +%s)
    execution_time=$((end_time-start_time))
    execution_time_minutes=$((execution_time / 60))
    echo "=== Subject $dirSub completed in $execution_time_minutes minutes ==="
    
    return 0
}

# Process single subject or multiple subjects
if [ -n "$subject_id" ]; then
    # Single subject processing
    if [ ! -d "$freesurfer_dir/$subject_id" ]; then
        echo "ERROR: Subject directory '$subject_id' not found in $freesurfer_dir"
        exit 1
    fi
    
    echo "Processing subject: $subject_id"
    process_subject_step2 "$subject_id" "$hemisphere" "$prediction" "$MODEL_NAME" "$freesurfer_dir" "$hcp_surface_dir" "$script_dir" "$output_dir" "$myelination" "$seed"
    
    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    # Multiple subjects processing
    export -f process_subject_step2
    export hemisphere prediction MODEL_NAME freesurfer_dir hcp_surface_dir script_dir output_dir myelination model_type seed
    
    cd "$freesurfer_dir"
    
    # Collect subjects
    subjects=()
    for dirSub in `ls .`; do
        if [ "$dirSub" != "fsaverage" ] && \
           [[ ! "$dirSub" =~ ^\. ]] && \
           [[ ! "$dirSub" =~ ^processed_ ]] && \
           [[ ! "$dirSub" =~ \.txt$ ]] && \
           [[ ! "$dirSub" =~ \.log$ ]] && \
           [ "$dirSub" != "logs" ]; then
            subjects+=("$dirSub")
        fi
    done
    
    echo "Found ${#subjects[@]} subjects to process: ${subjects[*]}"
    
    # Process in parallel
    printf '%s\n' "${subjects[@]}" | xargs -I {} -P $n_jobs bash -c "process_subject_step2 '{}' '$hemisphere' '$prediction' '$MODEL_NAME' '$freesurfer_dir' '$hcp_surface_dir' '$script_dir' '$output_dir' '$myelination' '$seed'"
fi

# Calculate and display total time
total_end_time=$(date +%s)
total_execution_time=$((total_end_time-total_start_time))
total_minutes=$((total_execution_time / 60))
total_seconds=$((total_execution_time % 60))

echo ""
echo "==============================================="
echo "[Step 2: Fsaverage to Native] COMPLETED!"
echo "Total execution time: ${total_minutes}m ${total_seconds}s"
echo "==============================================="

if [ -n "$subject_id" ]; then
    echo "Subject processed: $subject_id"
    echo "Prediction: $prediction | Model: $MODEL_NAME | Hemisphere: $hemisphere"
else
    echo "Subjects processed: ${#subjects[@]}"
    echo "Prediction: $prediction | Model: $MODEL_NAME | Hemisphere: $hemisphere"
    if [ ${#subjects[@]} -gt 0 ]; then
        echo "Average time per subject: $((total_minutes * 60 + total_seconds))s ÷ ${#subjects[@]} = $(( (total_minutes * 60 + total_seconds) / ${#subjects[@]} ))s"
    fi
    echo "Parallel jobs used: $n_jobs"
fi

if [ -n "$output_dir" ]; then
    echo "Output location: $output_dir"
else
    echo "Output location: In-place within FreeSurfer directory"
fi
echo "==============================================="

