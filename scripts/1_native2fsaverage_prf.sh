#!/usr/bin/env bash

# Script: 1_native2fsaverage_prf.sh
# Purpose: Convert native pRF data (eccentricity, polar angle, pRF size) from HCP_prf to 32k_fs_LR template
#
# USAGE EXAMPLES:
#
# 1. Process single subject:
#    ./1_native2fsaverage_prf.sh \
#        -s /path/to/freesurfer/subjects \
#        -t /path/to/HCP/surfaces \
#        -p /path/to/HCP_prf/proj-xxx \
#        -h lh \
#        -i sub-100610
#
# 2. Process all subjects in HCP_prf:
#    ./1_native2fsaverage_prf.sh \
#        -s /path/to/freesurfer/subjects \
#        -t /path/to/HCP/surfaces \
#        -p /path/to/HCP_prf/proj-xxx \
#        -h lh \
#        -j 4
#
# REQUIRED OPTIONS:
#   -s PATH    Path to FreeSurfer subjects directory (for sphere.reg, midthickness)
#   -t PATH    Path to HCP surface templates (fs_LR-deformed_to-fsaverage.*.sphere.32k_fs_LR.surf.gii)
#   -p PATH    Path to HCP_prf (proj-xxx dir or parent; auto-detects proj-* for Brainlife layout)
#   -h lh|rh   Hemisphere to process
#
# OPTIONAL OPTIONS:
#   -i ID      Subject ID for single subject (e.g., sub-100610 or 100610)
#   -o PATH    Output directory (default: in-place within FreeSurfer directory)
#   -j N       Number of parallel jobs for batch (default: auto-detect)
#   -d PATTERN Custom dt-neuro-prf glob pattern (default: dt-neuro-prf.tag-*)

# Auto-detect number of cores
auto_cores=$(($(nproc) - 1))
[ $auto_cores -lt 1 ] && auto_cores=1

# Default values
n_jobs=$auto_cores
subject_id=""
output_dir=""
dt_pattern="dt-neuro-prf.tag-*"

while getopts s:t:p:h:j:i:o:d: flag
do
    case "${flag}" in
        s) dirSubs=${OPTARG};;
        t) dirHCP=${OPTARG};;
        p) dirHCPprf=${OPTARG};;
        h) hemisphere=${OPTARG};
           case "$hemisphere" in
               lh|rh) ;;
               *) echo "Invalid hemisphere: $hemisphere"; exit 1;;
           esac;;
        j) n_jobs=${OPTARG};;
        i) subject_id=${OPTARG};;
        o) output_dir=${OPTARG};;
        d) dt_pattern=${OPTARG};;
        ?)
            echo "Usage: $(basename "$0") -s freesurfer_dir -t hcp_surfaces -p hcp_prf_dir -h lh|rh [-i subject_id] [-o output_dir] [-j n_jobs] [-d dt_pattern]" >&2
            exit 1;;
    esac
done

echo "Hemisphere: $hemisphere"

# Validation
if [ -z "$dirSubs" ]; then
    echo "ERROR: FreeSurfer subjects directory (-s) is required"
    exit 1
fi

if [ -z "$dirHCP" ]; then
    echo "ERROR: HCP surface directory (-t) is required"
    exit 1
fi

if [ -z "$dirHCPprf" ]; then
    echo "ERROR: HCP_prf directory (-p) is required"
    exit 1
fi

# Auto-detect proj-* subdirectory if -p points to HCP_prf parent (Brainlife/datalad layout)
# e.g. HCP_prf/proj-5dceb267c4ae281d2c297b92/sub-100610
if ! ls -d "$dirHCPprf"/sub-* 1>/dev/null 2>&1; then
    proj_dir=$(find "$dirHCPprf" -maxdepth 1 -type d -name "proj-*" 2>/dev/null | head -1)
    if [ -n "$proj_dir" ] && ls -d "$proj_dir"/sub-* 1>/dev/null 2>&1; then
        dirHCPprf="$proj_dir"
        echo "Using HCP_prf base: $dirHCPprf"
    fi
fi

if [ ! -f "$dirHCP/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii" ] || \
   [ ! -f "$dirHCP/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii" ]; then
    echo "ERROR: HCP sphere templates not found in $dirHCP"
    exit 1
fi

# HCP hemisphere suffix
if [ "$hemisphere" == "lh" ]; then
    hcp_hemi="L"
else
    hcp_hemi="R"
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
total_start_time=$(date +%s)

# Find prf_surfaces directory for a subject
find_prf_surfaces() {
    local sub_id=$1
    local sub_dir=""
    # Normalize: ensure sub- prefix for HCP_prf
    if [[ "$sub_id" == sub-* ]]; then
        sub_dir="$dirHCPprf/$sub_id"
    else
        sub_dir="$dirHCPprf/sub-$sub_id"
    fi
    if [ ! -d "$sub_dir" ]; then
        echo ""
        return 1
    fi
    local prf_dir=$(find "$sub_dir" -maxdepth 3 -type d -name "prf_surfaces" 2>/dev/null | head -1)
    if [ -n "$prf_dir" ]; then
        echo "$prf_dir"
        return 0
    fi
    # Try dt-neuro-prf pattern
    local dt_dirs=($(find "$sub_dir" -maxdepth 1 -type d -name "$dt_pattern" 2>/dev/null))
    for dt in "${dt_dirs[@]}"; do
        if [ -d "$dt/prf_surfaces" ]; then
            echo "$dt/prf_surfaces"
            return 0
        fi
    done
    echo ""
    return 1
}

# Find FreeSurfer surf directory (supports HCP dt-neuro-freesurfer structure)
find_fs_surf_dir() {
    local sub_id=$1
    local sub_dir=""
    local fs_base="$dirSubs"
    if [[ "$sub_id" == sub-* ]]; then
        sub_dir="$fs_base/$sub_id"
    else
        sub_dir="$fs_base/sub-$sub_id"
    fi
    # Try HCP_freesurfer fallback if dirSubs has no surf data (Brainlife: HCP_prf and HCP_freesurfer are siblings)
    local has_surf=""
    if [ -d "$sub_dir" ]; then
        has_surf=$(find "$sub_dir" -maxdepth 3 -path "*/output/surf" -type d 2>/dev/null | head -1)
    fi
    if [ -z "$has_surf" ] && [[ "$dirHCPprf" == *"/proj-"* ]]; then
        local prj_parent="$(dirname "$dirHCPprf")"
        local prj_name="$(basename "$dirHCPprf")"
        local fs_fallback="$(dirname "$prj_parent")/HCP_freesurfer/$prj_name"
        if [ -d "$fs_fallback/$sub_id" ] 2>/dev/null || [ -d "$fs_fallback/sub-$sub_id" ] 2>/dev/null; then
            fs_base="$fs_fallback"
            sub_dir="$fs_base/$sub_id"
            [ -d "$sub_dir" ] || sub_dir="$fs_base/sub-$sub_id"
        fi
    fi
    if [ ! -d "$sub_dir" ]; then
        echo ""
        return 1
    fi
    # HCP structure: sub-xxx/dt-neuro-freesurfer.../output/surf
    local surf_path=$(find "$sub_dir" -maxdepth 3 -type d -path "*/output/surf" 2>/dev/null | head -1)
    if [ -n "$surf_path" ] && [ -f "$surf_path/$hemisphere.white" ] 2>/dev/null; then
        echo "$surf_path"
        return 0
    fi
    # Flat structure: dirSubs/sub_id/surf
    if [ -d "$sub_dir/surf" ]; then
        echo "$sub_dir/surf"
        return 0
    fi
    echo ""
    return 1
}

# Find input pRF file (check multiple naming conventions)
find_prf_file() {
    local prf_dir=$1
    local metric=$2
    local hemi=$3
    case "$metric" in
        eccentricity)
            for name in "${hemi}.eccentricity.gii" "${hemi}.ecc.gii"; do
                if [ -f "$prf_dir/$name" ]; then echo "$prf_dir/$name"; return 0; fi
            done;;
        polarAngle)
            for name in "${hemi}.polarAngle.gii" "${hemi}.polar_angle.gii"; do
                if [ -f "$prf_dir/$name" ]; then echo "$prf_dir/$name"; return 0; fi
            done;;
        pRFsize)
            for name in "${hemi}.pRFsize.gii" "${hemi}.rfWidth.gii" "${hemi}.receptiveFieldSize.gii" "${hemi}.sigma.gii"; do
                if [ -f "$prf_dir/$name" ]; then echo "$prf_dir/$name"; return 0; fi
            done;;
    esac
    echo ""
    return 1
}

process_subject() {
    local sub_id=$1
    local prf_surfaces=$(find_prf_surfaces "$sub_id")
    local fs_surf=$(find_fs_surf_dir "$sub_id")

    if [ -z "$prf_surfaces" ]; then
        echo "[$sub_id] SKIP: prf_surfaces not found in HCP_prf"
        return 1
    fi

    if [ -z "$fs_surf" ]; then
        echo "[$sub_id] WARNING: FreeSurfer surf directory not found (needed for sphere.reg)"
        echo "[$sub_id] Attempting to use prf_surfaces only - sphere.reg must exist elsewhere"
    fi

    # Determine output directory
    local surf_output_dir=""
    if [ -n "$output_dir" ]; then
        local subject_output_dir="$output_dir/$sub_id"
        surf_output_dir="$subject_output_dir/surf"
        mkdir -p "$surf_output_dir"
    else
        if [ -n "$fs_surf" ]; then
            surf_output_dir="$fs_surf"
        else
            surf_output_dir="$dirSubs/$sub_id/surf"
            mkdir -p "$surf_output_dir"
        fi
    fi

    echo "=== Processing pRF for subject: $sub_id ==="

    # Surface prep: need sphere.reg and midthickness for resampling
    if [ -z "$fs_surf" ]; then
        echo "[$sub_id] ERROR: FreeSurfer directory required for sphere.reg - cannot resample without it"
        return 1
    fi

    local fs_sub_dir="$fs_surf"
    if [ ! -f "$surf_output_dir/$hemisphere.sphere.reg.surf.gii" ] || \
       [ ! -f "$surf_output_dir/$sub_id.$hemisphere.midthickness.32k_fs_LR.surf.gii" ]; then
        # Prefer pre-generated GIFTI files (HCP/Brainlife dt-neuro-freesurfer often has these)
        local src_sphere="$fs_sub_dir/$hemisphere.sphere.reg.surf.gii"
        local src_mid="$fs_sub_dir/$hemisphere.midthickness.surf.gii"
        local src_32k=$(find "$fs_sub_dir" -maxdepth 1 -name "*${hemisphere}.midthickness.32k_fs_LR.surf.gii" 2>/dev/null | head -1)
        [ -z "$src_32k" ] && src_32k=$(find "$fs_sub_dir" -maxdepth 1 -name "*.${hcp_hemi}.midthickness.32k_fs_LR.surf.gii" 2>/dev/null | head -1)
        if [ -f "$src_sphere" ] && [ -n "$src_32k" ]; then
            echo "[$sub_id] Using pre-generated GIFTI surfaces from source..."
            cp "$src_sphere" "$surf_output_dir/$hemisphere.sphere.reg.surf.gii"
            cp "$src_32k" "$surf_output_dir/$sub_id.$hemisphere.midthickness.32k_fs_LR.surf.gii"
            if [ -f "$src_mid" ]; then
                cp "$src_mid" "$surf_output_dir/$hemisphere.midthickness.surf.gii"
            elif [ -f "$fs_sub_dir/$hemisphere.graymid.H.gii" ]; then
                cp "$fs_sub_dir/$hemisphere.graymid.H.gii" "$surf_output_dir/$hemisphere.midthickness.surf.gii"
            else
                echo "[$sub_id] WARNING: Native midthickness not found; resampling may use 32k as fallback"
            fi
        elif [ -f "$fs_sub_dir/$hemisphere.white" ] && [ -f "$fs_sub_dir/$hemisphere.sphere.reg" ]; then
            echo "[$sub_id] Running freesurfer-resample-prep (requires FreeSurfer mris_convert in PATH)..."
            if ! wb_shortcuts -freesurfer-resample-prep \
                "$fs_sub_dir/$hemisphere.white" \
                "$fs_sub_dir/$hemisphere.pial" \
                "$fs_sub_dir/$hemisphere.sphere.reg" \
                "$dirHCP/fs_LR-deformed_to-fsaverage.${hcp_hemi}.sphere.32k_fs_LR.surf.gii" \
                "$surf_output_dir/$hemisphere.midthickness.surf.gii" \
                "$surf_output_dir/$sub_id.$hemisphere.midthickness.32k_fs_LR.surf.gii" \
                "$surf_output_dir/$hemisphere.sphere.reg.surf.gii"; then
                echo "[$sub_id] ERROR: wb_shortcuts -freesurfer-resample-prep failed. Ensure mris_convert is in PATH and FreeSurfer surfaces are valid."
                return 1
            fi
        else
            echo "[$sub_id] ERROR: FreeSurfer surfaces required. Need either: (1) sphere.reg.surf.gii + *midthickness.32k_fs_LR.surf.gii in source, or (2) white + pial + sphere.reg for freesurfer-resample-prep"
            return 1
        fi
    fi

    local sphere_reg="$surf_output_dir/$hemisphere.sphere.reg.surf.gii"
    if [ ! -f "$sphere_reg" ]; then
        echo "[$sub_id] ERROR: Surface prep failed - $sphere_reg not found. Check that wb_shortcuts/mris_convert ran successfully and FreeSurfer has lh.white, lh.pial, lh.sphere.reg"
        return 1
    fi
    local area_native="$surf_output_dir/$hemisphere.midthickness.surf.gii"
    local area_target="$surf_output_dir/$sub_id.$hemisphere.midthickness.32k_fs_LR.surf.gii"
    local target_sphere="$dirHCP/fs_LR-deformed_to-fsaverage.${hcp_hemi}.sphere.32k_fs_LR.surf.gii"

    # Ensure midthickness exists (may need conversion from graymid)
    if [ ! -f "$area_native" ] && [ -f "$surf_output_dir/$hemisphere.graymid" ]; then
        mris_convert "$surf_output_dir/$hemisphere.graymid" "$area_native"
    fi

    processed_any=0

    # Eccentricity
    local ecc_input=$(find_prf_file "$prf_surfaces" "eccentricity" "$hemisphere")
    local ecc_output="$surf_output_dir/$sub_id.eccentricity.$hemisphere.32k_fs_LR.func.gii"
    if [ -n "$ecc_input" ] && [ ! -f "$ecc_output" ]; then
        echo "[$sub_id] Resampling eccentricity..."
        wb_command -metric-resample "$ecc_input" "$sphere_reg" "$target_sphere" \
            ADAP_BARY_AREA "$ecc_output" \
            -area-surfs "$area_native" "$area_target"
        processed_any=1
    elif [ -f "$ecc_output" ]; then
        echo "[$sub_id] Eccentricity already resampled"
    fi

    # Polar angle (sin/cos method)
    local pa_input=$(find_prf_file "$prf_surfaces" "polarAngle" "$hemisphere")
    local pa_output="$surf_output_dir/$sub_id.polarAngle.$hemisphere.32k_fs_LR.func.gii"
    if [ -n "$pa_input" ] && [ ! -f "$pa_output" ]; then
        echo "[$sub_id] Resampling polar angle (sin/cos method)..."
        python3 "$SCRIPT_DIR/resample_polar_angle_sincos.py" \
            --input "$pa_input" \
            --sphere-reg "$sphere_reg" \
            --target-sphere "$target_sphere" \
            --area-surf-native "$area_native" \
            --area-surf-target "$area_target" \
            --output "$pa_output"
        processed_any=1
    elif [ -f "$pa_output" ]; then
        echo "[$sub_id] Polar angle already resampled"
    fi

    # pRF size
    local size_input=$(find_prf_file "$prf_surfaces" "pRFsize" "$hemisphere")
    local size_output="$surf_output_dir/$sub_id.pRFsize.$hemisphere.32k_fs_LR.func.gii"
    if [ -n "$size_input" ] && [ ! -f "$size_output" ]; then
        echo "[$sub_id] Resampling pRF size..."
        wb_command -metric-resample "$size_input" "$sphere_reg" "$target_sphere" \
            ADAP_BARY_AREA "$size_output" \
            -area-surfs "$area_native" "$area_target"
        processed_any=1
    elif [ -f "$size_output" ]; then
        echo "[$sub_id] pRF size already resampled"
    fi

    if [ $processed_any -eq 0 ] && [ -z "$ecc_input" ] && [ -z "$pa_input" ] && [ -z "$size_input" ]; then
        echo "[$sub_id] WARNING: No pRF files found in $prf_surfaces"
    fi

    echo "=== Subject $sub_id completed ==="
    return 0
}

# Collect subjects
if [ -n "$subject_id" ]; then
    process_subject "$subject_id"
else
    # Batch: find all subjects in HCP_prf
    subjects=()
    for sub_dir in "$dirHCPprf"/sub-*; do
        [ -d "$sub_dir" ] || continue
        sub_name=$(basename "$sub_dir")
        if [ -n "$(find_prf_surfaces "$sub_name")" ]; then
            subjects+=("$sub_name")
        fi
    done
    echo "Found ${#subjects[@]} subjects with pRF data"
    if [ ${#subjects[@]} -eq 0 ]; then
        echo "No subjects to process. Exiting."
        exit 0
    fi
    export -f process_subject find_prf_surfaces find_fs_surf_dir find_prf_file
    export dirHCP dirHCPprf dirSubs hemisphere output_dir SCRIPT_DIR hcp_hemi dt_pattern
    printf '%s\n' "${subjects[@]}" | xargs -I {} -P $n_jobs bash -c "process_subject '{}'"
fi

# Summary
total_end_time=$(date +%s)
total_seconds=$((total_end_time - total_start_time))
total_minutes=$((total_seconds / 60))
echo ""
echo "==============================================="
echo "[pRF Conversion] COMPLETED!"
echo "Total time: ${total_minutes}m $((total_seconds % 60))s"
echo "==============================================="
