#!/usr/bin/env bash

# Auto-detect number of cores (leave 1 core free)
auto_cores=$(($(nproc) - 1))
[ $auto_cores -lt 1 ] && auto_cores=1  # Ensure at least 1 core

# Default values
n_jobs=$auto_cores
subject_id=""
output_dir=""
t1_path=""
t2_path=""
smooth="yes"  # Default to smoothing the myelin map

while getopts s:h:j:i:o:t1:t2:n: flag
do
    case "${flag}" in
        s) dirSubs=${OPTARG};;
        h) hemisphere=${OPTARG};
           case "$hemisphere" in
               lh|rh) ;;
               *) echo "Invalid hemisphere argument: $hemisphere"; exit 1;;
           esac;;
        j) n_jobs=${OPTARG};;
        i) subject_id=${OPTARG};;
        o) output_dir=${OPTARG};;
        t1) t1_path=${OPTARG};;
        t2) t2_path=${OPTARG};;
        n) smooth=${OPTARG};
           case "$smooth" in
               'yes'|'no') ;;
               *) echo "Invalid smooth argument: $smooth (must be 'yes' or 'no')"; exit 1;;
           esac;;
        ?)
            echo "script usage: $(basename "$0") [-s path to subs] [-h hemisphere] [-j number of cores] [-i subject ID] [-o output directory] [-t1 path to T1 image] [-t2 path to T2 image] [-n smooth (yes/no)]" >&2
            exit 1;;
    esac
done

echo "Hemisphere: $hemisphere"

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
process_subject() {
    local dirSub=$1
    local hemisphere=$2
    local dirSubs=$3
    local output_dir=$4
    local t1_path=$5
    local t2_path=$6
    local smooth=$7
    
    echo "=== Processing myelin map generation for subject: $dirSub ==="
    
    # Determine output paths
    if [ -n "$output_dir" ]; then
        local subject_output_dir="$output_dir/$dirSub"
        local surf_output_dir="$subject_output_dir/surf"
        local mri_output_dir="$subject_output_dir/mri"
        mkdir -p "$surf_output_dir"
        mkdir -p "$mri_output_dir"
        echo "[$dirSub] Using custom output directory: $subject_output_dir"
    else
        local surf_output_dir="$dirSubs/$dirSub/surf"
        local mri_output_dir="$dirSubs/$dirSub/mri"
        echo "[$dirSub] Using FreeSurfer directory: $surf_output_dir"
    fi
    
    if [ ! -d "$dirSubs/$dirSub" ]; then
        echo "[$dirSub] ERROR: Subject directory not found"
        exit 1
    fi
    
    # Determine T1 and T2 paths for this subject
    local subject_t1=""
    local subject_t2=""
    
    # If global T1/T2 paths provided, use them
    if [ -n "$t1_path" ] && [ -n "$t2_path" ]; then
        subject_t1="$t1_path"
        subject_t2="$t2_path"
    # Otherwise, try to find T1/T2 in subject directory
    elif [ -f "$dirSubs/$dirSub/mri/T1.mgz" ] && [ -f "$dirSubs/$dirSub/mri/T2.mgz" ]; then
        subject_t1="$dirSubs/$dirSub/mri/T1.mgz"
        subject_t2="$dirSubs/$dirSub/mri/T2.mgz"
    elif [ -f "$dirSubs/$dirSub/mri/orig.mgz" ] && [ -f "$dirSubs/$dirSub/mri/T2.mgz" ]; then
        subject_t1="$dirSubs/$dirSub/mri/orig.mgz"
        subject_t2="$dirSubs/$dirSub/mri/T2.mgz"
    else
        echo "[$dirSub] ERROR: T1 and T2 images not found"
        echo "[$dirSub] Please provide T1 and T2 paths using -t1 and -t2 options, or ensure T1.mgz/T2.mgz exist in $dirSubs/$dirSub/mri/"
        exit 1
    fi
    
    echo "[$dirSub] Using T1: $subject_t1"
    echo "[$dirSub] Using T2: $subject_t2"
    
    start_time=$(date +%s)
    
    # Step 1: Generate myelin map volume from T1/T2 ratio
    local myelin_volume="$mri_output_dir/myelin_map.nii.gz"
    
    if [ ! -f "$myelin_volume" ]; then
        echo "[$dirSub] [Step 1] Computing myelin map volume from T1/T2 ratio..."
        
        # Convert T1 and T2 to NIFTI if needed (for wb_command compatibility)
        local t1_nii="$mri_output_dir/t1_temp.nii.gz"
        local t2_nii="$mri_output_dir/t2_temp.nii.gz"
        
        # Convert MGZ to NIFTI if needed
        if [[ "$subject_t1" == *.mgz ]]; then
            echo "[$dirSub] Converting T1 from MGZ to NIFTI..."
            mri_convert "$subject_t1" "$t1_nii"
        else
            # If already NIFTI, copy or use directly
            if [[ "$subject_t1" == *.nii.gz ]] || [[ "$subject_t1" == *.nii ]]; then
                cp "$subject_t1" "$t1_nii"
            else
                echo "[$dirSub] Converting T1 to NIFTI..."
                mri_convert "$subject_t1" "$t1_nii"
            fi
        fi
        
        if [[ "$subject_t2" == *.mgz ]]; then
            echo "[$dirSub] Converting T2 from MGZ to NIFTI..."
            mri_convert "$subject_t2" "$t2_nii"
        else
            # If already NIFTI, copy or use directly
            if [[ "$subject_t2" == *.nii.gz ]] || [[ "$subject_t2" == *.nii ]]; then
                cp "$subject_t2" "$t2_nii"
            else
                echo "[$dirSub] Converting T2 to NIFTI..."
                mri_convert "$subject_t2" "$t2_nii"
            fi
        fi
        
        # Reslice T2 to T1 space if needed
        echo "[$dirSub] Reslicing T2 to T1 space..."
        local t2_resliced="$mri_output_dir/t2_resliced.nii.gz"
        mri_vol2vol --mov "$t2_nii" --targ "$t1_nii" --regheader --interp nearest --o "$t2_resliced"
        
        # Compute myelin map using wb_command (T1/T2 ratio, clamped to 0-100)
        echo "[$dirSub] Computing T1/T2 ratio for myelin map..."
        wb_command -volume-math "clamp((T1w / T2w), 0, 100)" "$myelin_volume" \
            -var T1w "$t1_nii" -var T2w "$t2_resliced" -fixnan 0
        
        # Clean up temporary files
        rm -f "$t1_nii" "$t2_nii" "$t2_resliced"
        
        if [ ! -f "$myelin_volume" ]; then
            echo "[$dirSub] ERROR: Failed to generate myelin map volume"
            exit 1
        fi
        echo "[$dirSub] Myelin map volume generated: $myelin_volume"
    else
        echo "[$dirSub] Myelin map volume already exists: $myelin_volume"
    fi
    
    # Step 1.5: Generate graymid surface if not available (required for mapping)
    local graymid_surf=""
    local expected_graymid=""
    
    if [ -n "$output_dir" ]; then
        expected_graymid="$surf_output_dir/$hemisphere.graymid"
    else
        expected_graymid="$dirSubs/$dirSub/surf/$hemisphere.graymid"
    fi
    
    # Check if graymid surface exists
    if [ -f "$expected_graymid" ]; then
        graymid_surf="$expected_graymid"
        echo "[$dirSub] Graymid surface found: $graymid_surf"
    elif [ -f "$dirSubs/$dirSub/surf/$hemisphere.graymid" ]; then
        graymid_surf="$dirSubs/$dirSub/surf/$hemisphere.graymid"
        echo "[$dirSub] Graymid surface found in original location: $graymid_surf"
        # Copy to expected location if using output directory
        if [ -n "$output_dir" ]; then
            echo "[$dirSub] Copying graymid surface to output directory..."
            cp "$graymid_surf" "$expected_graymid"
            graymid_surf="$expected_graymid"
        fi
    else
        # Generate graymid surface if not found
        echo "[$dirSub] [Step 1.5] Generating graymid surface..."
        
        # Check if white and pial surfaces exist
        local white_surf=""
        local pial_surf=""
        
        if [ -f "$dirSubs/$dirSub/surf/$hemisphere.white" ]; then
            white_surf="$dirSubs/$dirSub/surf/$hemisphere.white"
        else
            echo "[$dirSub] ERROR: White surface not found: $dirSubs/$dirSub/surf/$hemisphere.white"
            exit 1
        fi
        
        if [ -f "$dirSubs/$dirSub/surf/$hemisphere.pial" ]; then
            pial_surf="$dirSubs/$dirSub/surf/$hemisphere.pial"
        else
            echo "[$dirSub] ERROR: Pial surface not found: $dirSubs/$dirSub/surf/$hemisphere.pial"
            exit 1
        fi
        
        # Generate graymid using mris_expand
        echo "[$dirSub] Expanding white surface to midthickness..."
        mris_expand -thickness "$white_surf" 0.5 "$expected_graymid"
        
        if [ ! -f "$expected_graymid" ]; then
            echo "[$dirSub] ERROR: Failed to generate graymid surface"
            exit 1
        fi
        
        # Compute curvature for the graymid surface
        echo "[$dirSub] Computing curvature for graymid surface..."
        mris_curvature -w "$expected_graymid"
        
        graymid_surf="$expected_graymid"
        echo "[$dirSub] Graymid surface generated: $graymid_surf"
    fi
    
    # Step 2: Map myelin volume to surface
    local myelin_surf="$surf_output_dir/$hemisphere.MyelinMap"
    local smoothed_myelin_surf="$surf_output_dir/$hemisphere.SmoothedMyelinMap"
    
    # Convert myelin volume to MGZ format for mri_vol2surf
    local myelin_mgz="$mri_output_dir/myelin_map.mgz"
    if [ ! -f "$myelin_mgz" ]; then
        echo "[$dirSub] Converting myelin map to MGZ format..."
        mri_convert "$myelin_volume" "$myelin_mgz"
    fi
    
    # Map volume to surface
    if [ ! -f "$myelin_surf" ]; then
        echo "[$dirSub] [Step 2] Mapping myelin volume to $hemisphere surface..."
        
        # Determine subjects directory and subject name for mri_vol2surf
        local vol2surf_sd=""
        local vol2surf_subject=""
        
        if [ -n "$output_dir" ]; then
            vol2surf_sd="$output_dir"
            vol2surf_subject="$dirSub"
        else
            vol2surf_sd="$dirSubs"
            vol2surf_subject="$dirSub"
        fi
        
        # Ensure graymid surface exists in the expected location for mri_vol2surf
        if [ "$graymid_surf" != "$expected_graymid" ] && [ -f "$graymid_surf" ]; then
            echo "[$dirSub] Ensuring graymid surface is in expected location..."
            cp "$graymid_surf" "$expected_graymid"
        fi
        
        # Use mri_vol2surf to map volume to surface
        mri_vol2surf --mov "$myelin_mgz" \
            --regheader \
            --projfrac 0.5 \
            --hemi "$hemisphere" \
            --surf graymid \
            --sd "$vol2surf_sd" \
            --s "$vol2surf_subject" \
            --o "$myelin_surf"
        
        if [ ! -f "$myelin_surf" ]; then
            echo "[$dirSub] ERROR: Failed to map myelin volume to surface"
            exit 1
        fi
        echo "[$dirSub] Myelin map mapped to surface: $myelin_surf"
    else
        echo "[$dirSub] Myelin map surface file already exists: $myelin_surf"
    fi
    
    # Step 3: Smooth the myelin map if requested
    if [ "$smooth" == "yes" ] && [ ! -f "$smoothed_myelin_surf" ]; then
        echo "[$dirSub] [Step 3] Smoothing myelin map..."
        
        local subject_dir_for_smooth=""
        if [ -n "$output_dir" ]; then
            subject_dir_for_smooth="$subject_output_dir"
        else
            subject_dir_for_smooth="$dirSubs/$dirSub"
        fi
        
        # Use mris_smooth to smooth the surface data
        mris_smooth -nw -n 10 "$myelin_surf" "$smoothed_myelin_surf"
        
        if [ ! -f "$smoothed_myelin_surf" ]; then
            echo "[$dirSub] WARNING: Smoothing failed, but unsmoothed map is available"
        else
            echo "[$dirSub] Smoothed myelin map created: $smoothed_myelin_surf"
        fi
    elif [ "$smooth" == "yes" ] && [ -f "$smoothed_myelin_surf" ]; then
        echo "[$dirSub] Smoothed myelin map already exists: $smoothed_myelin_surf"
    else
        echo "[$dirSub] Smoothing skipped (smooth=$smooth)"
    fi
    
    end_time=$(date +%s)
    execution_time=$((end_time-start_time))
    execution_time_minutes=$((execution_time / 60))
    echo "=== Subject $dirSub completed in $execution_time_minutes minutes ==="
}

# Process single subject or multiple subjects
if [ -n "$subject_id" ]; then
    # Single subject processing
    if [ ! -d "$dirSubs/$subject_id" ]; then
        echo "ERROR: Subject directory '$subject_id' not found in $dirSubs"
        exit 1
    fi
    
    # Validate T1/T2 paths for single subject
    if [ -z "$t1_path" ] || [ -z "$t2_path" ]; then
        echo "ERROR: T1 and T2 paths must be provided for single subject processing using -t1 and -t2 options"
        exit 1
    fi
    
    if [ ! -f "$t1_path" ]; then
        echo "ERROR: T1 file not found: $t1_path"
        exit 1
    fi
    
    if [ ! -f "$t2_path" ]; then
        echo "ERROR: T2 file not found: $t2_path"
        exit 1
    fi
    
    echo "Processing subject: $subject_id"
    process_subject "$subject_id" "$hemisphere" "$dirSubs" "$output_dir" "$t1_path" "$t2_path" "$smooth"
    
else
    # Multiple subjects processing
    export -f process_subject
    export hemisphere dirSubs output_dir t1_path t2_path smooth
    
    cd $dirSubs
    
    # Collect subjects
    subjects=()
    for dirSub in `ls .`; do
        if [ "$dirSub" != "fsaverage" ] && [[ "$dirSub" != .* ]] && [ "$dirSub" != processed_* ] && [[ "$dirSub" != *.txt ]] && [[ "$dirSub" != *.log ]] && [[ "$dirSub" != "logs" ]]; then
            subjects+=("$dirSub")
        fi
    done

    echo "Found ${#subjects[@]} subjects to process: ${subjects[*]}"
    
    if [ -z "$t1_path" ] || [ -z "$t2_path" ]; then
        echo "WARNING: Global T1/T2 paths not provided. Will attempt to find T1/T2 in each subject's mri directory."
    fi

    # Process in parallel
    printf '%s\n' "${subjects[@]}" | xargs -I {} -P $n_jobs bash -c "process_subject '{}' '$hemisphere' '$dirSubs' '$output_dir' '$t1_path' '$t2_path' '$smooth'"
fi

# Calculate and display total time
total_end_time=$(date +%s)
total_execution_time=$((total_end_time-total_start_time))
total_minutes=$((total_execution_time / 60))
total_seconds=$((total_execution_time % 60))

echo ""
echo "==============================================="
echo "[Myelin Map Generation] COMPLETED!"
echo "Total execution time: ${total_minutes}m ${total_seconds}s"

if [ -n "$subject_id" ]; then
    echo "Subject processed: $subject_id"
else
    echo "Subjects processed: ${#subjects[@]}"
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



