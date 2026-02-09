#!/usr/bin/env python3
"""
Batch inference script for running inference on multiple subjects with multiple model checkpoints.

Usage:
    python run_batch_inference_from_checkpoints.py \
        --subject_list Retinotopy/data/subject_splits/seed0/test_subjects.txt \
        --checkpoint_dirs Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed0 \
                         Models/output_wandb/eccentricity_Left_transolver_optionA_noMyelin_seed0 \
        [--output_base_dir inference_output_batch] \
        [--data_dir Retinotopy/data/raw/converted] \
        [--parallel] \
        [--num_workers 4]

Example:
    # Run sequentially
    python run_batch_inference_from_checkpoints.py \
        --subject_list Retinotopy/data/subject_splits/seed0/test_subjects.txt \
        --checkpoint_dirs Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed0
    
    # Run in parallel with 4 workers
    python run_batch_inference_from_checkpoints.py \
        --subject_list Retinotopy/data/subject_splits/seed0/test_subjects.txt \
        --checkpoint_dirs Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed0 \
        --parallel --num_workers 4
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp
from functools import partial


def find_best_model_checkpoint(checkpoint_dir: str) -> str:
    """
    Find the best model checkpoint (.pt file) in the given directory.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        
    Returns:
        Path to the best model checkpoint file
        
    Raises:
        FileNotFoundError: If no best model checkpoint is found
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Find all best_model checkpoint files
    best_model_files = list(checkpoint_path.glob("*best_model*.pt"))
    
    if not best_model_files:
        raise FileNotFoundError(f"No best model checkpoint found in {checkpoint_dir}")
    
    # If multiple best model files exist, use the one with the highest epoch number
    if len(best_model_files) > 1:
        # Extract epoch numbers and sort
        def get_epoch_num(filepath):
            import re
            match = re.search(r'epoch(\d+)', filepath.name)
            return int(match.group(1)) if match else 0
        
        best_model_files.sort(key=get_epoch_num, reverse=True)
        print(f"Warning: Multiple best model checkpoints found in {checkpoint_dir}")
        print(f"Using the latest: {best_model_files[0].name}")
    
    return str(best_model_files[0])


def load_subject_list(subject_list_file: str) -> List[str]:
    """
    Load subject IDs from a text file.
    
    Args:
        subject_list_file: Path to the text file containing subject IDs
        
    Returns:
        List of subject IDs as strings
    """
    subject_list_path = Path(subject_list_file)
    
    if not subject_list_path.exists():
        raise FileNotFoundError(f"Subject list file does not exist: {subject_list_file}")
    
    with open(subject_list_path, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(subjects)} subjects from {subject_list_file}")
    return subjects


def run_single_inference(
    subject_id: str,
    checkpoint_path: str,
    data_dir: str,
    output_base_dir: str,
    script_path: str
) -> Tuple[str, str, int]:
    """
    Run inference for a single subject with a single checkpoint.
    
    Args:
        subject_id: Subject ID
        checkpoint_path: Path to the model checkpoint
        data_dir: Path to the data directory
        output_base_dir: Base directory for output
        script_path: Path to the inference script
        
    Returns:
        Tuple of (subject_id, checkpoint_name, return_code)
    """
    # Extract model name from checkpoint path for organizing output
    checkpoint_name = Path(checkpoint_path).parent.name
    output_dir = os.path.join(output_base_dir, checkpoint_name)
    
    # Construct command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        script_path,
        "--subject_id", subject_id,
        "--checkpoint_path", checkpoint_path,
        "--data_dir", data_dir,
        "--output_dir", output_dir
    ]
    
    print(f"\n{'='*80}")
    print(f"Running inference:")
    print(f"  Subject: {subject_id}")
    print(f"  Checkpoint: {checkpoint_name}")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return_code = result.returncode
        status = "SUCCESS" if return_code == 0 else "FAILED"
        print(f"[{status}] Subject {subject_id} with {checkpoint_name}")
        return (subject_id, checkpoint_name, return_code)
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] Subject {subject_id} with {checkpoint_name}: {e}")
        return (subject_id, checkpoint_name, e.returncode)
    except Exception as e:
        print(f"[ERROR] Subject {subject_id} with {checkpoint_name}: {e}")
        return (subject_id, checkpoint_name, -1)


def run_batch_inference(
    subjects: List[str],
    checkpoint_paths: List[str],
    data_dir: str,
    output_base_dir: str,
    script_path: str,
    parallel: bool = False,
    num_workers: int = 4
) -> None:
    """
    Run batch inference for all subjects and checkpoints.
    
    Args:
        subjects: List of subject IDs
        checkpoint_paths: List of checkpoint file paths
        data_dir: Path to the data directory
        output_base_dir: Base directory for output
        script_path: Path to the inference script
        parallel: Whether to run in parallel
        num_workers: Number of parallel workers
    """
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Generate all combinations of subjects and checkpoints
    tasks = [
        (subject_id, checkpoint_path, data_dir, output_base_dir, script_path)
        for subject_id in subjects
        for checkpoint_path in checkpoint_paths
    ]
    
    total_tasks = len(tasks)
    print(f"\n{'='*80}")
    print(f"Batch Inference Configuration:")
    print(f"  Total subjects: {len(subjects)}")
    print(f"  Total checkpoints: {len(checkpoint_paths)}")
    print(f"  Total inference tasks: {total_tasks}")
    print(f"  Parallel mode: {parallel}")
    if parallel:
        print(f"  Number of workers: {num_workers}")
    print(f"  Output directory: {output_base_dir}")
    print(f"{'='*80}\n")
    
    # Run inference tasks
    results = []
    if parallel:
        print(f"Running inference in parallel with {num_workers} workers...\n")
        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(run_single_inference, tasks)
    else:
        print("Running inference sequentially...\n")
        for task in tasks:
            result = run_single_inference(*task)
            results.append(result)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Batch Inference Summary:")
    print(f"{'='*80}")
    
    success_count = sum(1 for _, _, code in results if code == 0)
    failed_count = total_tasks - success_count
    
    print(f"Total tasks: {total_tasks}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    
    if failed_count > 0:
        print("\nFailed tasks:")
        for subject_id, checkpoint_name, code in results:
            if code != 0:
                print(f"  - Subject {subject_id}, Checkpoint {checkpoint_name} (code: {code})")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference script for multiple subjects and checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--subject_list",
        type=str,
        required=True,
        help="Path to text file containing subject IDs (one per line)"
    )
    
    parser.add_argument(
        "--checkpoint_dirs",
        type=str,
        nargs='+',
        required=True,
        help="List of checkpoint directories (will use best_model*.pt from each)"
    )
    
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="inference_output_batch",
        help="Base directory for all inference outputs (default: inference_output_batch)"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Retinotopy/data/raw/converted",
        help="Path to the data directory (default: Retinotopy/data/raw/converted)"
    )
    
    parser.add_argument(
        "--inference_script",
        type=str,
        default="run_inference_from_fslr_curv_using_checkpoint.py",
        help="Path to the inference script (default: run_inference_from_fslr_curv_using_checkpoint.py)"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run inference tasks in parallel"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4, only used with --parallel)"
    )
    
    args = parser.parse_args()
    
    # Validate inference script exists
    script_path = Path(args.inference_script)
    if not script_path.exists():
        print(f"Error: Inference script not found: {args.inference_script}")
        sys.exit(1)
    
    # Load subjects
    try:
        subjects = load_subject_list(args.subject_list)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if not subjects:
        print("Error: No subjects found in the subject list file")
        sys.exit(1)
    
    # Find best model checkpoints
    checkpoint_paths = []
    for checkpoint_dir in args.checkpoint_dirs:
        try:
            checkpoint_path = find_best_model_checkpoint(checkpoint_dir)
            checkpoint_paths.append(checkpoint_path)
            print(f"Found checkpoint: {checkpoint_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    if not checkpoint_paths:
        print("Error: No valid checkpoints found")
        sys.exit(1)
    
    # Run batch inference
    run_batch_inference(
        subjects=subjects,
        checkpoint_paths=checkpoint_paths,
        data_dir=args.data_dir,
        output_base_dir=args.output_base_dir,
        script_path=str(script_path.absolute()),
        parallel=args.parallel,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
