#!/usr/bin/env python3
"""
Generate different subject splits for different seeds (0, 1, 2).
This script creates processed data files with seed suffixes and saves
subject lists for each seed's train/dev/test splits.
"""

import os
import os.path as osp
import numpy as np
import scipy.io
from numpy.random import seed

def get_subject_list(data_path):
    """Load subject list from MATLAB file or list_subj file"""
    # Try to get subjects from MATLAB file (like get_test_subjects.py)
    mat_file = osp.join(data_path, 'cifti_curv_all.mat')
    if osp.exists(mat_file):
        mat_data = scipy.io.loadmat(mat_file)
        curv_data = mat_data['cifti_curv']
        
        subject_ids = []
        for key in curv_data.dtype.names:
            if key.startswith('x') and '_curvature' in key:
                subject_id = key.replace('x', '').replace('_curvature', '')
                subject_ids.append(subject_id)
        
        # Sort to ensure consistent ordering before shuffling
        subject_ids = sorted(subject_ids)
        return subject_ids
    
    # Fallback: try to read from list_subj file
    list_subj_path = osp.join(data_path, '..', 'list_subj')
    if osp.exists(list_subj_path):
        with open(list_subj_path) as fp:
            subjects = fp.read().split("\n")
        subjects = [s.strip() for s in subjects if s.strip()]
        return subjects
    
    raise FileNotFoundError(f"Could not find subject list. Checked {mat_file} and {list_subj_path}")

def generate_seed_splits(data_path, output_dir, seeds=[0, 1, 2], n_train=161, n_dev=10):
    """
    Generate subject splits for different seeds.
    
    Args:
        data_path: Path to raw data directory (containing cifti_curv_all.mat or list_subj)
        output_dir: Directory to save subject list files
        seeds: List of seeds to use
        n_train: Number of training subjects
        n_dev: Number of development subjects
    """
    print("=" * 80)
    print("Generating Subject Splits for Different Seeds")
    print("=" * 80)
    
    # Get all subjects
    print(f"\n1. Loading subject list from: {data_path}")
    all_subjects = get_subject_list(data_path)
    print(f"   Found {len(all_subjects)} total subjects")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate splits for each seed
    for seed_val in seeds:
        print(f"\n2. Generating split for seed {seed_val}...")
        
        # Shuffle subjects with this seed
        seed(seed_val)
        shuffled_subjects = all_subjects.copy()
        np.random.shuffle(shuffled_subjects)
        
        # Split subjects
        train_subjects = shuffled_subjects[0:n_train]
        dev_subjects = shuffled_subjects[n_train:n_train+n_dev]
        test_subjects = shuffled_subjects[n_train+n_dev:]
        
        print(f"   Train: {len(train_subjects)} subjects (indices 0-{n_train-1})")
        print(f"   Dev: {len(dev_subjects)} subjects (indices {n_train}-{n_train+n_dev-1})")
        print(f"   Test: {len(test_subjects)} subjects (indices {n_train+n_dev}-{len(shuffled_subjects)-1})")
        
        # Save subject lists
        seed_output_dir = osp.join(output_dir, f'seed{seed_val}')
        os.makedirs(seed_output_dir, exist_ok=True)
        
        # Save train subjects
        train_file = osp.join(seed_output_dir, 'train_subjects.txt')
        with open(train_file, 'w') as f:
            for subj in train_subjects:
                f.write(f"{subj}\n")
        print(f"   Saved train subjects to: {train_file}")
        
        # Save dev subjects
        dev_file = osp.join(seed_output_dir, 'dev_subjects.txt')
        with open(dev_file, 'w') as f:
            for subj in dev_subjects:
                f.write(f"{subj}\n")
        print(f"   Saved dev subjects to: {dev_file}")
        
        # Save test subjects
        test_file = osp.join(seed_output_dir, 'test_subjects.txt')
        with open(test_file, 'w') as f:
            for subj in test_subjects:
                f.write(f"{subj}\n")
        print(f"   Saved test subjects to: {test_file}")
        
        # Save combined file with all splits
        combined_file = osp.join(seed_output_dir, 'all_subjects.txt')
        with open(combined_file, 'w') as f:
            f.write("# Train subjects\n")
            for subj in train_subjects:
                f.write(f"{subj}\n")
            f.write("\n# Dev subjects\n")
            for subj in dev_subjects:
                f.write(f"{subj}\n")
            f.write("\n# Test subjects\n")
            for subj in test_subjects:
                f.write(f"{subj}\n")
        print(f"   Saved combined list to: {combined_file}")
    
    print("\n" + "=" * 80)
    print("Subject split generation completed!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Generated splits for seeds: {seeds}")

if __name__ == '__main__':
    import os
    import sys
    
    # Set paths
    script_dir = osp.dirname(osp.abspath(__file__))
    data_path = osp.join(script_dir, 'Retinotopy', 'data', 'raw', 'converted')
    output_dir = osp.join(script_dir, 'Retinotopy', 'data', 'subject_splits')
    
    # Generate splits
    generate_seed_splits(data_path, output_dir, seeds=[0, 1, 2], n_train=161, n_dev=10)
