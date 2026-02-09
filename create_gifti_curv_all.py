#!/usr/bin/env python3
"""
Create gifti_curv_all.mat file from GIFTI curvature files.
This script collects curvature files from all subjects and creates a MATLAB file
following the same structure as cifti_curv_all.mat.
"""

import os
import os.path as osp
import glob
import numpy as np
import scipy.io
import nibabel as nib
from collections import OrderedDict


def load_existing_structure(mat_path):
    """Load existing cifti_curv_all.mat to understand its structure."""
    print(f"Loading existing structure from: {mat_path}")
    mat_data = scipy.io.loadmat(mat_path, struct_as_record=False)
    
    if 'cifti_curv' in mat_data:
        curv = mat_data['cifti_curv']
        print(f"Type of cifti_curv: {type(curv)}")
        print(f"Shape: {curv.shape if hasattr(curv, 'shape') else 'N/A'}")
        
        # Try to inspect the structure
        if isinstance(curv, np.ndarray):
            if curv.size > 0:
                first_elem = curv.flat[0]
                print(f"First element type: {type(first_elem)}")
                if hasattr(first_elem, 'dtype') and first_elem.dtype.names:
                    print(f"Field names: {first_elem.dtype.names}")
                    print(f"Number of fields: {len(first_elem.dtype.names)}")
                    # Check a sample field
                    if len(first_elem.dtype.names) > 0:
                        sample_field = first_elem[first_elem.dtype.names[0]]
                        print(f"Sample field shape: {sample_field.shape if hasattr(sample_field, 'shape') else 'N/A'}")
                        print(f"Sample field type: {type(sample_field)}")
        
        return curv
    else:
        print("Warning: 'cifti_curv' key not found in MAT file")
        return None


def collect_curvature_files(subject_list_path, base_path):
    """
    Collect all curvature files from subjects.
    
    Returns:
        dict: Dictionary mapping subject_id to curvature file path
    """
    # Read subjects
    with open(subject_list_path, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]
    
    curvature_files = {}
    missing_files = []
    
    print(f"\nCollecting curvature files for {len(subjects)} subjects...")
    
    for idx, subject in enumerate(subjects):
        if idx % 20 == 0:
            print(f"  Processing subject {idx+1}/{len(subjects)}: {subject}")
        
        subject_dir = osp.join(base_path, f'sub-{subject}')
        
        if not osp.exists(subject_dir):
            missing_files.append(subject)
            continue
        
        # Search for dt-neuro-freesurfer directory
        pattern = osp.join(subject_dir, 'dt-neuro-freesurfer.tag-v5.tag-thalamic_nuclei.id-*')
        matching_dirs = glob.glob(pattern)
        
        if not matching_dirs:
            missing_files.append(subject)
            continue
        
        # Find curvature file
        found = False
        for dt_dir in matching_dirs:
            curv_file = osp.join(dt_dir, 'output/surf/output.curvature-midthickness.lh.32k_fs_LR.func.gii')
            if osp.exists(curv_file):
                curvature_files[subject] = curv_file
                found = True
                break
        
        if not found:
            missing_files.append(subject)
    
    print(f"\nFound curvature files: {len(curvature_files)}/{len(subjects)}")
    if missing_files:
        print(f"Missing files: {len(missing_files)}")
        print(f"  First 10 missing: {missing_files[:10]}")
    
    return curvature_files, subjects


def load_curvature_data(curvature_file, number_hemi_nodes=32492):
    """
    Load curvature data from GIFTI file.
    
    Args:
        curvature_file: Path to GIFTI curvature file
        number_hemi_nodes: Number of hemisphere nodes (default 32492)
    
    Returns:
        numpy array: Curvature data of shape (number_hemi_nodes,)
    """
    try:
        gii = nib.load(curvature_file)
        data = np.array(gii.agg_data()).flatten()
        
        # Ensure correct size
        if len(data) != number_hemi_nodes:
            print(f"Warning: Expected {number_hemi_nodes} vertices, got {len(data)}")
            # Pad or truncate if necessary
            if len(data) < number_hemi_nodes:
                padded = np.full(number_hemi_nodes, np.nan, dtype=np.float32)
                padded[:len(data)] = data
                data = padded
            else:
                data = data[:number_hemi_nodes]
        
        # Replace NaN with 0 (following the pattern from read_freesurfer.py)
        data = np.nan_to_num(data, nan=0.0)
        
        return data.astype(np.float32)
    except Exception as e:
        print(f"Error loading {curvature_file}: {e}")
        # Return zeros if file cannot be loaded
        return np.zeros(number_hemi_nodes, dtype=np.float32)


def create_matlab_structure(curvature_files, subjects, number_hemi_nodes=32492):
    """
    Create MATLAB structure following the pattern from cifti_curv_all.mat.
    
    Based on the usage pattern:
    - curv['pos'][0][0] for position data
    - curv['x{subject_id}_curvature'][0][0] for subject curvature data
    
    The structure appears to be a MATLAB struct array where:
    - First field is 'pos' (position data)
    - Subsequent fields are 'x{subject_id}_curvature' for each subject
    """
    print("\nCreating MATLAB structure...")
    
    # We need to create a structure similar to the existing one
    # Based on convert_cifti_to_mat.py pattern, it seems to use a list approach
    
    # For now, let's create a dictionary structure that MATLAB can understand
    # We'll need to check if 'pos' exists in the original file
    # If not, we'll create a placeholder or load it from another source
    
    # Load position data - need (64984, 3) for combined left + right hemispheres
    number_cortical_nodes = 64984  # Left + right hemispheres combined
    pos_file_L = 'Retinotopy/data/raw/converted/mid_pos_L.mat'
    pos_file_R = 'Retinotopy/data/raw/converted/mid_pos_R.mat'
    
    if osp.exists(pos_file_L) and osp.exists(pos_file_R):
        print("  Loading position data from mid_pos_L.mat and mid_pos_R.mat...")
        pos_data_L = scipy.io.loadmat(pos_file_L)['mid_pos_L']
        pos_data_R = scipy.io.loadmat(pos_file_R)['mid_pos_R']
        # Combine left and right: (64984, 3)
        pos_data = np.concatenate([pos_data_L, pos_data_R], axis=0)
        print(f"  Position data shape: {pos_data.shape}")
    elif osp.exists(pos_file_L):
        print("  Loading position data from mid_pos_L.mat (right hemisphere not found)...")
        pos_data_L = scipy.io.loadmat(pos_file_L)['mid_pos_L']
        # Create placeholder for right hemisphere
        pos_data_R = np.zeros((number_hemi_nodes, 3), dtype=np.float32)
        pos_data = np.concatenate([pos_data_L, pos_data_R], axis=0)
        print(f"  Position data shape: {pos_data.shape}")
    else:
        print("  Warning: mid_pos_L.mat not found. Creating placeholder for pos.")
        # Create a placeholder (64984, 3) for position
        pos_data = np.zeros((number_cortical_nodes, 3), dtype=np.float32)
    
    # Create the structure dictionary
    # MATLAB struct arrays are tricky - we'll use the approach from convert_cifti_to_mat.py
    # which creates a list and wraps it
    
    # Create field names and data arrays
    # 'dimord' must be first field (as in original file)
    field_names = ['dimord']
    # dimord value is ['pos'] based on original file
    dimord_value = np.array(['pos'], dtype='<U3')
    data_arrays = [dimord_value]
    
    # Add 'pos' field (will be added after subject fields to match original order)
    # But we'll add it at the end to match the original structure
    
    # Add subject curvature data
    # Note: Original file has (64984, 1) shape = left (32492) + right (32492) hemispheres
    print(f"  Adding curvature data for {len(curvature_files)} subjects...")
    for subject in subjects:
        if subject in curvature_files:
            curv_data_lh = load_curvature_data(curvature_files[subject], number_hemi_nodes)
            field_name = f'x{subject}_curvature'
            field_names.append(field_name)
            # Create combined array: left hemisphere data + NaN for right hemisphere
            # Shape should be (64984, 1) to match original format
            curv_data_rh = np.full(number_hemi_nodes, np.nan, dtype=np.float32)
            combined_data = np.concatenate([curv_data_lh, curv_data_rh]).reshape(-1, 1)
            data_arrays.append(combined_data)
        else:
            # Create NaN array for missing subjects (64984, 1)
            field_name = f'x{subject}_curvature'
            field_names.append(field_name)
            nan_data = np.full((number_cortical_nodes, 1), np.nan, dtype=np.float32)
            data_arrays.append(nan_data)
    
    # Add 'pos' field at the end (to match original file structure)
    field_names.append('pos')
    data_arrays.append(pos_data)
    
    print(f"  Created structure with {len(field_names)} fields")
    print(f"    First field: {field_names[0]}")
    print(f"    Sample subject field: {field_names[1] if len(field_names) > 1 else 'N/A'}")
    
    # Create MATLAB-compatible structure
    # Following the pattern from convert_cifti_to_mat.py
    # We need to create a structure array that MATLAB can read
    
    # Create a structured array
    # The tricky part is that MATLAB structs are accessed as curv['field'][0][0]
    # This suggests a nested structure
    
    # Try creating it as a 1x1 struct array with named fields
    dtype_list = [(name, 'O') for name in field_names]
    struct_array = np.zeros((1, 1), dtype=dtype_list)
    
    # Fill in the data
    # Based on original file structure: 
    # - curv['x100610_curvature'] returns (1, 1) array (struct array indexing)
    # - curv['x100610_curvature'][0][0] returns (64984, 1) array (actual data)
    # The struct array itself is (1, 1), and each field contains the data directly
    # So we store data directly in the struct array field
    for i, (field_name, data) in enumerate(zip(field_names, data_arrays)):
        # Store data directly - the struct array indexing handles the (1, 1) wrapper
        struct_array[0, 0][field_name] = data
    
    return struct_array


def main():
    """Main function."""
    print("=" * 80)
    print("Creating gifti_curv_all.mat from GIFTI curvature files")
    print("=" * 80)
    
    # Configuration
    subject_list_path = 'Retinotopy/data/list_subj'
    base_path = '/mnt/storage/junb/proj-5dceb267c4ae281d2c297b92'
    output_path = 'Retinotopy/data/raw/converted/gifti_curv_all.mat'
    existing_mat_path = 'Retinotopy/data/raw/converted/cifti_curv_all.mat'
    number_hemi_nodes = 32492
    
    # Step 1: Try to load existing structure for reference
    if osp.exists(existing_mat_path):
        print("\nStep 1: Inspecting existing structure...")
        existing_curv = load_existing_structure(existing_mat_path)
    else:
        print("\nStep 1: Existing cifti_curv_all.mat not found, proceeding without reference")
        existing_curv = None
    
    # Step 2: Collect curvature files
    print("\nStep 2: Collecting curvature files...")
    curvature_files, subjects = collect_curvature_files(subject_list_path, base_path)
    
    if len(curvature_files) == 0:
        print("Error: No curvature files found!")
        return
    
    # Step 3: Create MATLAB structure
    print("\nStep 3: Creating MATLAB structure...")
    struct_array = create_matlab_structure(curvature_files, subjects, number_hemi_nodes)
    
    # Step 4: Save to MATLAB file
    print(f"\nStep 4: Saving to {output_path}...")
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    
    # Wrap in dictionary with key 'cifti_curv' (keeping same key name for compatibility)
    final_dict = {'cifti_curv': struct_array}
    
    scipy.io.savemat(output_path, final_dict, oned_as='column', format='5')
    print(f"  Saved successfully!")
    
    # Step 5: Verify the saved file
    print(f"\nStep 5: Verifying saved file...")
    try:
        verify_data = scipy.io.loadmat(output_path, struct_as_record=False)
        if 'cifti_curv' in verify_data:
            curv = verify_data['cifti_curv']
            print(f"  Verification: File loaded successfully")
            print(f"  Structure shape: {curv.shape}")
            if curv.size > 0:
                first_elem = curv.flat[0]
                if hasattr(first_elem, 'dtype') and first_elem.dtype.names:
                    print(f"  Number of fields: {len(first_elem.dtype.names)}")
                    print(f"  First 5 fields: {first_elem.dtype.names[:5]}")
        else:
            print("  Warning: 'cifti_curv' key not found in saved file")
    except Exception as e:
        print(f"  Warning: Could not verify file: {e}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
