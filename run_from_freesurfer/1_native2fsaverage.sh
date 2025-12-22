#!/usr/bin/env bash

# Auto-detect number of cores (leave 1 core free)
auto_cores=$(($(nproc) - 1))
[ $auto_cores -lt 1 ] && auto_cores=1  # Ensure at least 1 core

# Default values
n_jobs=$auto_cores
subject_id=""
output_dir=""
fast="no"

while getopts s:t:h:g:j:i:o: flag
do
    case "${flag}" in
        s) dirSubs=${OPTARG};;
        t) dirHCP=${OPTARG};;
        h) hemisphere=${OPTARG};
           case "$hemisphere" in
               lh|rh) ;;
               *) echo "Invalid hemisphere argument: $hemisphere"; exit 1;;
           esac;;
        g) fast=${OPTARG};
            case "$fast" in
            'yes'|'no') ;;
            *) echo "Invalid fast argument: $fast"; exit 1;;
            esac;;
        j) n_jobs=${OPTARG};;
        i) subject_id=${OPTARG};;
        o) output_dir=${OPTARG};;
        ?)
            echo "script usage: $(basename "$0") [-s path to subs] [-t path to HCP surfaces] [-h hemisphere] [-g fast generation of midthickness surface] [-j number of cores for parallelization] [-i subject ID for single subject processing] [-o output directory]" >&2
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

# Check HCP surface directory
if [ -z "$dirHCP" ]; then
    echo "ERROR: HCP surface directory (-t) is required"
    echo "Please provide path to HCP surface templates containing fs_LR-deformed_to-fsaverage.*.sphere.32k_fs_LR.surf.gii files"
    exit 1
fi

if [ ! -f "$dirHCP/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii" ] || \
   [ ! -f "$dirHCP/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii" ]; then
    echo "ERROR: HCP surface template files not found in $dirHCP"
    echo "Required files:"
    echo "  - fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii"
    echo "  - fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii"
    exit 1
fi

# Start total timing
total_start_time=$(date +%s)

# Get script directory for midthickness_surf.py
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define the processing function
process_subject() {
    local dirSub=$1
    local hemisphere=$2
    local fast=$3
    local dirSubs=$4
    local dirHCP=$5
    local output_dir=$6
    
    echo "=== Processing Step 1 for subject: $dirSub ==="
    
    # Determine output paths
    if [ -n "$output_dir" ]; then
        local subject_output_dir="$output_dir/$dirSub"
        local surf_output_dir="$subject_output_dir/surf"
        mkdir -p "$surf_output_dir"
        echo "[$dirSub] Using custom output directory: $subject_output_dir"
    else
        local surf_output_dir="$dirSubs/$dirSub/surf"
        echo "[$dirSub] Using FreeSurfer directory: $surf_output_dir"
    fi
    
    if [ -d "$dirSubs/$dirSub/surf" ]; then
        start_time=$(date +%s)
        
        if [ "$fast" == "yes" ]; then
            echo "[$dirSub] Fast mode enabled."
            echo "[$dirSub] [Step 1.1] Generating mid-thickness surface and curvature data if not available..."
            if [ ! -f "$surf_output_dir/$hemisphere.graymid" ]; then
                if [ ! -f "$dirSubs/$dirSub/surf/$hemisphere.white" ]; then
                    echo "[$dirSub] ERROR: No white surface found"
                    exit 1
                else
                    echo "[$dirSub] Converting surfaces..."
                    mris_convert "$dirSubs/$dirSub/surf/$hemisphere.white" "$surf_output_dir/$hemisphere.white.gii"
                    mris_convert "$dirSubs/$dirSub/surf/$hemisphere.pial" "$surf_output_dir/$hemisphere.pial.gii"
                    echo "[$dirSub] Generating midthickness surface..."
                    python "$SCRIPT_DIR/midthickness_surf.py" --path "$surf_output_dir/" --hemisphere $hemisphere
                    mris_convert "$surf_output_dir/$hemisphere.graymid.gii" "$surf_output_dir/$hemisphere.graymid"
                    echo "[$dirSub] Computing curvature..."
                    mris_curvature -w "$surf_output_dir/$hemisphere.graymid"
                fi
                echo "[$dirSub] Mid-thickness surface has been generated"
            else 
                echo "[$dirSub] Mid-thickness surface and curvature data already available"
            fi
        else
            echo "[$dirSub] [Step 1.1] Generating mid-thickness surface and curvature data if not available..."
            if [ ! -f "$surf_output_dir/$hemisphere.graymid" ]; then
                if [ ! -f "$dirSubs/$dirSub/surf/$hemisphere.white" ]; then
                    echo "[$dirSub] ERROR: No white surface found"
                    exit 1
                else
                    echo "[$dirSub] Expanding white surface to midthickness..."
                    mris_expand -thickness "$dirSubs/$dirSub/surf/$hemisphere.white" 0.5 "$surf_output_dir/$hemisphere.graymid"
                    echo "[$dirSub] Computing curvature..."
                    mris_curvature -w "$surf_output_dir/$hemisphere.graymid"
                fi
                echo "[$dirSub] Mid-thickness surface has been generated"
            else 
                echo "[$dirSub] Mid-thickness surface and curvature data already available"
            fi
        fi    
        
        echo "[$dirSub] [Step 1.2] Preparing native surfaces for resampling..."
        if [ ! -f "$surf_output_dir/$dirSub.curvature-midthickness.$hemisphere.32k_fs_LR.func.gii" ]; then
            echo "[$dirSub] Running freesurfer-resample-prep..."
            if [ "$hemisphere" == "lh" ]; then
                wb_shortcuts -freesurfer-resample-prep "$dirSubs/$dirSub/surf/$hemisphere.white" "$dirSubs/$dirSub/surf/$hemisphere.pial" \
                "$dirSubs/$dirSub/surf/$hemisphere.sphere.reg" "$dirHCP/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii" \
                "$surf_output_dir/$hemisphere.midthickness.surf.gii" "$surf_output_dir/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii" \
                "$surf_output_dir/$hemisphere.sphere.reg.surf.gii"
                
                echo "[$dirSub] Converting curvature data..."
                mris_convert -c "$surf_output_dir/$hemisphere.graymid.H" "$surf_output_dir/$hemisphere.graymid" "$surf_output_dir/$hemisphere.graymid.H.gii"
                
                echo "[$dirSub] Resampling native data to fsaverage space..."
                wb_command -metric-resample "$surf_output_dir/$hemisphere.graymid.H.gii" \
                "$surf_output_dir/$hemisphere.sphere.reg.surf.gii" "$dirHCP/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii" \
                ADAP_BARY_AREA "$surf_output_dir/$dirSub.curvature-midthickness.$hemisphere.32k_fs_LR.func.gii" \
                -area-surfs "$surf_output_dir/$hemisphere.midthickness.surf.gii" "$surf_output_dir/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii"
            else
                wb_shortcuts -freesurfer-resample-prep "$dirSubs/$dirSub/surf/$hemisphere.white" "$dirSubs/$dirSub/surf/$hemisphere.pial" \
                "$dirSubs/$dirSub/surf/$hemisphere.sphere.reg" "$dirHCP/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii" \
                "$surf_output_dir/$hemisphere.midthickness.surf.gii" "$surf_output_dir/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii" \
                "$surf_output_dir/$hemisphere.sphere.reg.surf.gii"
                
                echo "[$dirSub] Converting curvature data..."
                mris_convert -c "$surf_output_dir/$hemisphere.graymid.H" "$surf_output_dir/$hemisphere.graymid" "$surf_output_dir/$hemisphere.graymid.H.gii"
                
                echo "[$dirSub] Resampling native data to fsaverage space..."
                wb_command -metric-resample "$surf_output_dir/$hemisphere.graymid.H.gii" \
                "$surf_output_dir/$hemisphere.sphere.reg.surf.gii" "$dirHCP/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii" \
                ADAP_BARY_AREA "$surf_output_dir/$dirSub.curvature-midthickness.$hemisphere.32k_fs_LR.func.gii" \
                -area-surfs "$surf_output_dir/$hemisphere.midthickness.surf.gii" "$surf_output_dir/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii"
            fi
            echo "[$dirSub] Data resampling complete"
        else
            echo "[$dirSub] Resampled curvature data already available"
        fi
        
        echo "[$dirSub] [Step 1.3] Processing myelin map for fslr template..."
        # Check for myelin map files
        local myelin_map_file=""
        local graymid_surf_file=""
        
        if [ -f "$dirSubs/$dirSub/surf/$hemisphere.SmoothedMyelinMap" ]; then
            myelin_map_file="$dirSubs/$dirSub/surf/$hemisphere.SmoothedMyelinMap"
            graymid_surf_file="$dirSubs/$dirSub/surf/$hemisphere.graymid"
        elif [ -f "$dirSubs/$dirSub/surf/$hemisphere.MyelinMap" ]; then
            myelin_map_file="$dirSubs/$dirSub/surf/$hemisphere.MyelinMap"
            graymid_surf_file="$dirSubs/$dirSub/surf/$hemisphere.graymid"
        elif [ -f "$surf_output_dir/$hemisphere.SmoothedMyelinMap" ]; then
            myelin_map_file="$surf_output_dir/$hemisphere.SmoothedMyelinMap"
            graymid_surf_file="$surf_output_dir/$hemisphere.graymid"
        elif [ -f "$surf_output_dir/$hemisphere.MyelinMap" ]; then
            myelin_map_file="$surf_output_dir/$hemisphere.MyelinMap"
            graymid_surf_file="$surf_output_dir/$hemisphere.graymid"
        fi
        
        if [ -n "$myelin_map_file" ]; then
            if [ ! -f "$surf_output_dir/$dirSub.myelin-midthickness.$hemisphere.32k_fs_LR.func.gii" ]; then
                echo "[$dirSub] Myelin map found: $myelin_map_file"
                
                # Check if required surface files exist
                if [ ! -f "$graymid_surf_file" ]; then
                    echo "[$dirSub] ERROR: Graymid surface not found for myelin map conversion: $graymid_surf_file"
                    exit 1
                fi
                
                # Check if sphere.reg.surf.gii exists (should be created in Step 1.2)
                if [ ! -f "$surf_output_dir/$hemisphere.sphere.reg.surf.gii" ]; then
                    echo "[$dirSub] WARNING: sphere.reg.surf.gii not found. Running freesurfer-resample-prep first..."
                    if [ "$hemisphere" == "lh" ]; then
                        wb_shortcuts -freesurfer-resample-prep "$dirSubs/$dirSub/surf/$hemisphere.white" "$dirSubs/$dirSub/surf/$hemisphere.pial" \
                        "$dirSubs/$dirSub/surf/$hemisphere.sphere.reg" "$dirHCP/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii" \
                        "$surf_output_dir/$hemisphere.midthickness.surf.gii" "$surf_output_dir/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii" \
                        "$surf_output_dir/$hemisphere.sphere.reg.surf.gii"
                    else
                        wb_shortcuts -freesurfer-resample-prep "$dirSubs/$dirSub/surf/$hemisphere.white" "$dirSubs/$dirSub/surf/$hemisphere.pial" \
                        "$dirSubs/$dirSub/surf/$hemisphere.sphere.reg" "$dirHCP/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii" \
                        "$surf_output_dir/$hemisphere.midthickness.surf.gii" "$surf_output_dir/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii" \
                        "$surf_output_dir/$hemisphere.sphere.reg.surf.gii"
                    fi
                fi
                
                # Ensure midthickness surface in GIFTI format exists
                local midthickness_surf_gii="$surf_output_dir/$hemisphere.midthickness.surf.gii"
                if [ ! -f "$midthickness_surf_gii" ]; then
                    echo "[$dirSub] Converting midthickness surface to GIFTI format..."
                    mris_convert "$graymid_surf_file" "$midthickness_surf_gii"
                fi
                
                echo "[$dirSub] Converting myelin map data to GIFTI format..."
                # Convert myelin map to GIFTI using graymid surface as reference
                mris_convert -c "$myelin_map_file" "$graymid_surf_file" "$surf_output_dir/$hemisphere.myelin.gii"
                
                echo "[$dirSub] Resampling myelin map to fslr template..."
                if [ "$hemisphere" == "lh" ]; then
                    wb_command -metric-resample "$surf_output_dir/$hemisphere.myelin.gii" \
                    "$surf_output_dir/$hemisphere.sphere.reg.surf.gii" "$dirHCP/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii" \
                    ADAP_BARY_AREA "$surf_output_dir/$dirSub.myelin-midthickness.$hemisphere.32k_fs_LR.func.gii" \
                    -area-surfs "$surf_output_dir/$hemisphere.midthickness.surf.gii" "$surf_output_dir/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii"
                else
                    wb_command -metric-resample "$surf_output_dir/$hemisphere.myelin.gii" \
                    "$surf_output_dir/$hemisphere.sphere.reg.surf.gii" "$dirHCP/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii" \
                    ADAP_BARY_AREA "$surf_output_dir/$dirSub.myelin-midthickness.$hemisphere.32k_fs_LR.func.gii" \
                    -area-surfs "$surf_output_dir/$hemisphere.midthickness.surf.gii" "$surf_output_dir/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii"
                fi
                echo "[$dirSub] Myelin map resampling complete"
            else
                echo "[$dirSub] Resampled myelin map data already available"
            fi
        else
            echo "[$dirSub] WARNING: Myelin map file not found. Skipping myelin map processing."
            echo "[$dirSub] Expected files: $hemisphere.SmoothedMyelinMap or $hemisphere.MyelinMap in surf directory"
        fi
        
        end_time=$(date +%s)
        execution_time=$((end_time-start_time))
        execution_time_minutes=$((execution_time / 60))
        echo "=== Subject $dirSub completed in $execution_time_minutes minutes ==="         
    else
        echo "[$dirSub] ERROR: No surface directory found"
        exit 1
    fi
}

# Process single subject or multiple subjects
if [ -n "$subject_id" ]; then
    # Single subject processing
    if [ ! -d "$dirSubs/$subject_id" ]; then
        echo "ERROR: Subject directory '$subject_id' not found in $dirSubs"
        exit 1
    fi
    
    echo "Processing subject: $subject_id"
    process_subject "$subject_id" "$hemisphere" "$fast" "$dirSubs" "$dirHCP" "$output_dir"
    
else
    # Multiple subjects processing
    export -f process_subject
    export hemisphere fast dirSubs dirHCP output_dir SCRIPT_DIR
    
    cd $dirSubs
    
    # Collect subjects
    subjects=()
    for dirSub in `ls .`; do
        if [ "$dirSub" != "fsaverage" ] && [[ "$dirSub" != .* ]] && [ "$dirSub" != processed_* ] && [[ "$dirSub" != *.txt ]] && [[ "$dirSub" != *.log ]] && [[ "$dirSub" != "logs" ]]; then
            subjects+=("$dirSub")
        fi
    done

    echo "Found ${#subjects[@]} subjects to process: ${subjects[*]}"

    # Process in parallel
    printf '%s\n' "${subjects[@]}" | xargs -I {} -P $n_jobs bash -c "process_subject '{}' '$hemisphere' '$fast' '$dirSubs' '$dirHCP' '$output_dir'"
fi

# Calculate and display total time
total_end_time=$(date +%s)
total_execution_time=$((total_end_time-total_start_time))
total_minutes=$((total_execution_time / 60))
total_seconds=$((total_execution_time % 60))

echo ""
echo "==============================================="
echo "[Step 1] COMPLETED!"
echo "Total execution time: ${total_minutes}m ${total_seconds}s"

if [ -n "$subject_id" ]; then
    echo "Subject processed: $subject_id"
else
    echo "Subjects processed: ${#subjects[@]}"
    echo "Average time per subject: $((total_minutes * 60 + total_seconds))s ÷ ${#subjects[@]} = $(( (total_minutes * 60 + total_seconds) / ${#subjects[@]} ))s"
    echo "Parallel jobs used: $n_jobs"
fi

if [ -n "$output_dir" ]; then
    echo "Output location: $output_dir"
else
    echo "Output location: In-place within FreeSurfer directory"
fi
echo "==============================================="



