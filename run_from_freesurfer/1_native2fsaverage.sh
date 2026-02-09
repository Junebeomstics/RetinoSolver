#!/usr/bin/env bash

# Script: 1_native2fsaverage.sh
# Purpose: Convert native FreeSurfer surfaces to fsaverage space
#          Generates midthickness surfaces and resamples curvature/myelin data to 32k_fs_LR template
#
# USAGE EXAMPLES:
#
# 1. Process single subject (recommended):
#    ./1_native2fsaverage.sh \
#        -s /path/to/freesurfer/subjects \
#        -t /path/to/HCP/surfaces \
#        -h lh \
#        -i sub-191033
#
# 2. Process single subject with custom output directory:
#    ./1_native2fsaverage.sh \
#        -s /path/to/freesurfer/subjects \
#        -t /path/to/HCP/surfaces \
#        -h lh \
#        -i sub-191033 \
#        -o /path/to/output
#
# 3. Process single subject with fast mode:
#    ./1_native2fsaverage.sh \
#        -s /path/to/freesurfer/subjects \
#        -t /path/to/HCP/surfaces \
#        -h lh \
#        -i sub-191033 \
#        -g yes
#
# 4. Process single subject using wb_command for curvature calculation:
#    ./1_native2fsaverage.sh \
#        -s /path/to/freesurfer/subjects \
#        -t /path/to/HCP/surfaces \
#        -h lh \
#        -i sub-191033 \
#        -w yes
#
# 5. Process all subjects in directory (parallel processing):
#    ./1_native2fsaverage.sh \
#        -s /path/to/freesurfer/subjects \
#        -t /path/to/HCP/surfaces \
#        -h lh \
#        -j 4
#
# 6. All options combined:
#    ./1_native2fsaverage.sh \
#        -s /path/to/freesurfer/subjects \
#        -t /path/to/HCP/surfaces \
#        -h lh \
#        -i sub-191033 \
#        -o /path/to/output \
#        -g yes \
#        -w yes \
#        -j 4
#
# REQUIRED OPTIONS:
#   -s PATH    Path to FreeSurfer subjects directory
#   -t PATH    Path to HCP surface templates (must contain fs_LR-deformed_to-fsaverage.*.sphere.32k_fs_LR.surf.gii)
#   -h lh|rh   Hemisphere to process (lh or rh)
#
# OPTIONAL OPTIONS:
#   -i ID      Subject ID for single subject processing (if not specified, processes all subjects)
#   -o PATH    Output directory (if not specified, saves in-place within FreeSurfer directory)
#   -g yes|no  Fast mode for midthickness surface generation (default: no)
#              When 'yes', uses Python script instead of mris_expand
#   -w yes|no  Use wb_command for curvature calculation (default: no)
#              When 'yes', uses wb_command -surface-curvature instead of mris_curvature
#              Output files will have '_wb' suffix to avoid overwriting
#   -j N       Number of parallel jobs (default: auto-detect, leaves 1 core free)
#
# NOTES:
# - If -i option is specified, only that subject will be processed
# - If -i option is omitted, all subjects in -s directory will be processed in parallel
# - Output files are saved with different names when using -w yes to avoid overwriting:
#   * Standard: {subject}.curvature-midthickness.{hemisphere}.32k_fs_LR.func.gii
#   * With -w yes: {subject}.curvature-midthickness.{hemisphere}.32k_fs_LR_wb.func.gii
# - Required HCP surface files:
#   * fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii
#   * fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii

# Auto-detect number of cores (leave 1 core free)
auto_cores=$(($(nproc) - 1))
[ $auto_cores -lt 1 ] && auto_cores=1  # Ensure at least 1 core

# Default values
n_jobs=$auto_cores
subject_id=""
output_dir=""
fast="no"
calculate_curv_using_wb="no"

while getopts s:t:h:g:j:i:o:w: flag
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
        w) calculate_curv_using_wb=${OPTARG};
            case "$calculate_curv_using_wb" in
            'yes'|'no') ;;
            *) echo "Invalid calculate_curv_using_wb argument: $calculate_curv_using_wb"; exit 1;;
            esac;;
        ?)
            echo "script usage: $(basename "$0") [-s path to subs] [-t path to HCP surfaces] [-h hemisphere] [-g fast generation of midthickness surface] [-j number of cores for parallelization] [-i subject ID for single subject processing] [-o output directory] [-w calculate curvature using wb_command (yes/no, default: no)]" >&2
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
    local calculate_curv_using_wb=$7
    
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
        
        echo "[$dirSub] [Step 1.1] Generating mid-thickness surface and curvature data if not available..."
        if [ "$fast" == "yes" ]; then
            echo "[$dirSub] Fast mode enabled."
        fi
        
        # Check graymid surface existence separately from curvature
        local need_graymid=0
        local need_curvature=0
        
        if [ ! -f "$surf_output_dir/$hemisphere.graymid" ]; then
            need_graymid=1
        fi
        
        # Check curvature file existence based on calculation method
        if [ "$calculate_curv_using_wb" == "yes" ]; then
            if [ ! -f "$surf_output_dir/$hemisphere.graymid.H.wb.gii" ]; then
                need_curvature=1
            fi
        else
            if [ ! -f "$surf_output_dir/$hemisphere.graymid.H" ]; then
                need_curvature=1
            fi
        fi
        
        if [ $need_graymid -eq 1 ]; then
            # Generate both graymid surface and curvature
            if [ ! -f "$dirSubs/$dirSub/surf/$hemisphere.white" ]; then
                echo "[$dirSub] ERROR: No white surface found"
                exit 1
            else
                # Generate graymid surface using method based on fast flag
                if [ "$fast" == "yes" ]; then
                    echo "[$dirSub] Converting surfaces in freesurfer format into gifti format..."
                    mris_convert "$dirSubs/$dirSub/surf/$hemisphere.white" "$surf_output_dir/$hemisphere.white.gii"
                    mris_convert "$dirSubs/$dirSub/surf/$hemisphere.pial" "$surf_output_dir/$hemisphere.pial.gii"
                    echo "[$dirSub] Generating midthickness surface... (using python script)"
                    python "$SCRIPT_DIR/midthickness_surf.py" --path "$surf_output_dir/" --hemisphere $hemisphere
                    mris_convert "$surf_output_dir/$hemisphere.graymid.gii" "$surf_output_dir/$hemisphere.graymid"
                else
                    echo "[$dirSub] Expanding white surface to midthickness... (using mris_expand)"
                    mris_expand -thickness "$dirSubs/$dirSub/surf/$hemisphere.white" 0.5 "$surf_output_dir/$hemisphere.graymid"
                fi
            fi
        else
            echo "[$dirSub] Mid-thickness surface already available"
        fi
        
        # Compute curvature if needed (either after generating graymid or standalone)
        if [ $need_curvature -eq 1 ]; then
            echo "[$dirSub] Computing curvature..."
            if [ "$calculate_curv_using_wb" == "yes" ]; then
                echo "[$dirSub] Using wb_command for curvature calculation..."
                # Ensure graymid is in GIFTI format for wb_command
                if [ ! -f "$surf_output_dir/$hemisphere.graymid.gii" ]; then
                    mris_convert "$surf_output_dir/$hemisphere.graymid" "$surf_output_dir/$hemisphere.graymid.gii"
                fi
                wb_command -surface-curvature "$surf_output_dir/$hemisphere.graymid.gii" \
                    -mean "$surf_output_dir/$hemisphere.graymid.H.wb.gii"
                # Note: wb_command creates GIFTI format directly, no conversion needed
                # Note: wb_command defines different sign convention for curvature compared to FreeSurfer
            else
                echo "[$dirSub] Using mris_curvature for curvature calculation..."
                mris_curvature -w "$surf_output_dir/$hemisphere.graymid"
            fi
            echo "[$dirSub] Curvature has been computed"
        else
            echo "[$dirSub] Curvature data already available"
        fi
    
        
        echo "[$dirSub] [Step 1.2] Preparing native surfaces for resampling..."
        # Determine curvature file suffix based on calculation method
        if [ "$calculate_curv_using_wb" == "yes" ]; then
            curv_suffix=".wb"
            curv_output_suffix="_wb"
        else
            curv_suffix=""
            curv_output_suffix=""
        fi
        
        if [ ! -f "$surf_output_dir/$dirSub.curvature-midthickness.$hemisphere.32k_fs_LR${curv_output_suffix}.func.gii" ]; then
            echo "[$dirSub] Running freesurfer-resample-prep..."
            
            # Determine HCP hemisphere suffix (L for lh, R for rh)
            if [ "$hemisphere" == "lh" ]; then
                hcp_hemi="L"
            else
                hcp_hemi="R"
            fi
            
            wb_shortcuts -freesurfer-resample-prep "$dirSubs/$dirSub/surf/$hemisphere.white" "$dirSubs/$dirSub/surf/$hemisphere.pial" \
            "$dirSubs/$dirSub/surf/$hemisphere.sphere.reg" "$dirHCP/fs_LR-deformed_to-fsaverage.${hcp_hemi}.sphere.32k_fs_LR.surf.gii" \
            "$surf_output_dir/$hemisphere.midthickness.surf.gii" "$surf_output_dir/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii" \
            "$surf_output_dir/$hemisphere.sphere.reg.surf.gii"
            
            echo "[$dirSub] Preparing curvature data for resampling..."
            if [ "$calculate_curv_using_wb" == "yes" ]; then
                # wb_command creates GIFTI format directly, use it as-is
                local curv_gii_file="$surf_output_dir/$hemisphere.graymid.H.wb.gii"
                if [ ! -f "$curv_gii_file" ]; then
                    echo "[$dirSub] ERROR: Curvature GIFTI file not found: $curv_gii_file"
                    exit 1
                fi
            else
                # Convert FreeSurfer curvature to GIFTI format
                mris_convert -c "$surf_output_dir/$hemisphere.graymid.H" "$surf_output_dir/$hemisphere.graymid" "$surf_output_dir/$hemisphere.graymid.H.gii"
                local curv_gii_file="$surf_output_dir/$hemisphere.graymid.H.gii"
            fi
            
            echo "[$dirSub] Resampling native data to fsaverage space..."
            wb_command -metric-resample "$curv_gii_file" \
            "$surf_output_dir/$hemisphere.sphere.reg.surf.gii" "$dirHCP/fs_LR-deformed_to-fsaverage.${hcp_hemi}.sphere.32k_fs_LR.surf.gii" \
            ADAP_BARY_AREA "$surf_output_dir/$dirSub.curvature-midthickness.$hemisphere.32k_fs_LR${curv_output_suffix}.func.gii" \
            -area-surfs "$surf_output_dir/$hemisphere.midthickness.surf.gii" "$surf_output_dir/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii"
            echo "[$dirSub] Data resampling complete"
        else
            echo "[$dirSub] Resampled curvature data already available"
        fi
        
        # Smoothing curvature data (separate from resampling check)
        if [ ! -f "$surf_output_dir/$dirSub.curvature-smoothed_2mm.$hemisphere.32k_fs_LR${curv_output_suffix}.func.gii" ]; then
            if [ -f "$surf_output_dir/$dirSub.curvature-midthickness.$hemisphere.32k_fs_LR${curv_output_suffix}.func.gii" ]; then
                echo "[$dirSub] Smoothing curvature data..."
                wb_command -metric-smoothing \
                "$surf_output_dir/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii" \
                "$surf_output_dir/$dirSub.curvature-midthickness.$hemisphere.32k_fs_LR${curv_output_suffix}.func.gii" \
                2 \
                "$surf_output_dir/$dirSub.curvature-smoothed_2mm.$hemisphere.32k_fs_LR${curv_output_suffix}.func.gii" \
                -fwhm
                echo "[$dirSub] Curvature smoothing complete"
            else
                echo "[$dirSub] WARNING: Resampled curvature file not found, skipping smoothing"
            fi
        else
            echo "[$dirSub] Smoothed curvature data already available"
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
                    
                    # Determine HCP hemisphere suffix (L for lh, R for rh)
                    if [ "$hemisphere" == "lh" ]; then
                        hcp_hemi="L"
                    else
                        hcp_hemi="R"
                    fi
                    
                    wb_shortcuts -freesurfer-resample-prep "$dirSubs/$dirSub/surf/$hemisphere.white" "$dirSubs/$dirSub/surf/$hemisphere.pial" \
                    "$dirSubs/$dirSub/surf/$hemisphere.sphere.reg" "$dirHCP/fs_LR-deformed_to-fsaverage.${hcp_hemi}.sphere.32k_fs_LR.surf.gii" \
                    "$surf_output_dir/$hemisphere.midthickness.surf.gii" "$surf_output_dir/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii" \
                    "$surf_output_dir/$hemisphere.sphere.reg.surf.gii"
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
                
                # Determine HCP hemisphere suffix (L for lh, R for rh)
                if [ "$hemisphere" == "lh" ]; then
                    hcp_hemi="L"
                else
                    hcp_hemi="R"
                fi
                
                wb_command -metric-resample "$surf_output_dir/$hemisphere.myelin.gii" \
                "$surf_output_dir/$hemisphere.sphere.reg.surf.gii" "$dirHCP/fs_LR-deformed_to-fsaverage.${hcp_hemi}.sphere.32k_fs_LR.surf.gii" \
                ADAP_BARY_AREA "$surf_output_dir/$dirSub.myelin-midthickness.$hemisphere.32k_fs_LR.func.gii" \
                -area-surfs "$surf_output_dir/$hemisphere.midthickness.surf.gii" "$surf_output_dir/$dirSub.$hemisphere.midthickness.32k_fs_LR.surf.gii"
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
    process_subject "$subject_id" "$hemisphere" "$fast" "$dirSubs" "$dirHCP" "$output_dir" "$calculate_curv_using_wb"
    
else
    # Multiple subjects processing
    export -f process_subject
    export hemisphere fast dirSubs dirHCP output_dir SCRIPT_DIR calculate_curv_using_wb
    
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
    printf '%s\n' "${subjects[@]}" | xargs -I {} -P $n_jobs bash -c "process_subject '{}' '$hemisphere' '$fast' '$dirSubs' '$dirHCP' '$output_dir' '$calculate_curv_using_wb'"
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



