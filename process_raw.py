#!/usr/bin/env python3
"""
Process raw data for all combinations of predictions, hemispheres, seeds, and myelination settings.
This script pre-processes data files so they are ready for training.
"""

import os.path as osp
import argparse
import torch_geometric.transforms as T
from Retinotopy.dataset.HCP_3sets_ROI import Retinotopy

# Set data path
path = osp.join(osp.dirname(osp.realpath(__file__)), 'Retinotopy', 'data')

pre_transform = T.Compose([T.FaceToEdge()])

def process_data(predictions=None, hemispheres=None, seeds=None, myelination_options=None, 
                 sets=None, n_examples=181, verbose=True):
    """
    Process data for all specified combinations.
    
    Args:
        predictions: List of predictions to process (default: ['eccentricity', 'polarAngle', 'pRFsize'])
        hemispheres: List of hemispheres to process (default: ['Left', 'Right'])
        seeds: List of seeds to process (default: [0, 1, 2]). Use [None] for default seed=1
        myelination_options: List of myelination settings (default: [True, False])
        sets: List of sets to process (default: ['Train', 'Development', 'Test'])
        n_examples: Number of examples (default: 181)
        verbose: Print progress messages (default: True)
    """
    # Set defaults
    if predictions is None:
        predictions = ['eccentricity', 'polarAngle', 'pRFsize']
    if hemispheres is None:
        hemispheres = ['Left', 'Right']
    if seeds is None:
        seeds = [0, 1, 2]
    if myelination_options is None:
        myelination_options = [True, False]
    if sets is None:
        sets = ['Train', 'Development', 'Test']
    
    total_combinations = len(predictions) * len(hemispheres) * len(seeds) * len(myelination_options) * len(sets)
    current = 0
    
    print("=" * 80)
    print("Processing Raw Data")
    print("=" * 80)
    print(f"Predictions: {predictions}")
    print(f"Hemispheres: {hemispheres}")
    print(f"Seeds: {seeds}")
    print(f"Myelination options: {myelination_options}")
    print(f"Sets: {sets}")
    print(f"Total combinations: {total_combinations}")
    print("=" * 80)
    print()
    
    for prediction in predictions:
        for hemisphere in hemispheres:
            for seed_val in seeds:
                for myelination in myelination_options:
                    for dataset_set in sets:
                        current += 1
                        
                        # Build description
                        seed_str = f"seed{seed_val}" if seed_val is not None else "default_seed"
                        myelination_str = "with_myelin" if myelination else "no_myelin"
                        
                        if verbose:
                            print(f"[{current}/{total_combinations}] Processing: {prediction} | {hemisphere} | {seed_str} | {myelination_str} | {dataset_set}")
                        
                        try:
                            # Create dataset instance - this will trigger processing if files don't exist
                            dataset = Retinotopy(
                                path, 
                                dataset_set,
                                transform=T.Cartesian(), 
                                pre_transform=pre_transform, 
                                n_examples=n_examples, 
                                prediction=prediction, 
                                myelination=myelination, 
                                hemisphere=hemisphere,
                                seed=seed_val
                            )
                            
                            if verbose:
                                print(f"  ✓ Successfully processed/loaded {dataset_set} set")
                                
                        except Exception as e:
                            print(f"  ✗ Error processing {prediction} | {hemisphere} | {seed_str} | {myelination_str} | {dataset_set}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
    
    print()
    print("=" * 80)
    print("All processing completed!")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process raw data for deepRetinotopy')
    parser.add_argument('--predictions', nargs='+', 
                       choices=['eccentricity', 'polarAngle', 'pRFsize'],
                       default=['eccentricity', 'polarAngle', 'pRFsize'],
                       help='Predictions to process (default: all)')
    parser.add_argument('--hemispheres', nargs='+', 
                       choices=['Left', 'Right'],
                       default=['Left', 'Right'],
                       help='Hemispheres to process (default: both)')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[0, 1, 2],
                       help='Seeds to process (default: 0 1 2). Use --seeds None for default seed=1')
    parser.add_argument('--myelination', nargs='+', type=str,
                       choices=['True', 'False', 'true', 'false', 'both'],
                       default=['both'],
                       help='Myelination options: True, False, or both (default: both)')
    parser.add_argument('--sets', nargs='+',
                       choices=['Train', 'Development', 'Test'],
                       default=['Train', 'Development', 'Test'],
                       help='Dataset sets to process (default: all)')
    parser.add_argument('--n_examples', type=int, default=181,
                       help='Number of examples (default: 181)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Handle seeds: convert None string to actual None
    seeds = []
    for s in args.seeds:
        if isinstance(s, str) and s.lower() == 'none':
            seeds.append(None)
        else:
            seeds.append(s)
    
    # Handle myelination options
    myelination_options = []
    if 'both' in args.myelination or len(args.myelination) == 0:
        myelination_options = [True, False]
    else:
        for m in args.myelination:
            if m.lower() in ['true', '1']:
                myelination_options.append(True)
            elif m.lower() in ['false', '0']:
                myelination_options.append(False)
    
    # Run processing
    process_data(
        predictions=args.predictions,
        hemispheres=args.hemispheres,
        seeds=seeds,
        myelination_options=myelination_options,
        sets=args.sets,
        n_examples=args.n_examples,
        verbose=not args.quiet
    )