#!/usr/bin/env python3
"""
End-to-end inference script that extracts task and hemisphere from checkpoint path
and runs inference on training/dev/test data (subject_index or subject_id) from Retinotopy/data/raw/converted. 

Docker Usage (Recommended):
    docker exec -it <CONTAINER_ID> python run_inference_from_checkpoint.py \
        --subject_id <SUBJECT_ID> \
        --checkpoint_path Models/output_wandb/<MODEL_DIR>/<CHECKPOINT_FILE>.pt
    
    Example:
    docker exec -it b5842b28ca21 python run_inference_from_checkpoint.py \
        --subject_id 157336 \
        --checkpoint_path Models/output_wandb/eccentricity_Left_baseline_noMyelin_seed1/ecc_Left_baseline_noMyelin_seed1_best_model_epoch66.pt
    
    Note: 
    - Use subject_id without 'sub-' prefix (e.g., 157336 instead of sub-157336)
    - Checkpoint path is relative to workspace directory (/workspace)
    - Output will be saved to /workspace/inference_output by default

Direct Usage:
    python run_inference_from_checkpoint.py \
        --subject_index SUBJECT_INDEX \
        --checkpoint_path /path/to/model.pt \
        --data_dir /path/to/Retinotopy/data/raw/converted \
        [--output_dir /path/to/output]
    
    Or with subject ID:
    python run_inference_from_checkpoint.py \
        --subject_id SUBJECT_ID \
        --checkpoint_path /path/to/model.pt \
        --data_dir /path/to/Retinotopy/data/raw/converted \
        [--output_dir /path/to/output]
"""

import os
import os.path as osp
import sys
import argparse
import re
import torch
import torch_geometric.transforms as T
import numpy as np
from pathlib import Path
import warnings
import scipy.io
warnings.filterwarnings("ignore")

# Add project root to path
project_root = osp.dirname(osp.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add Models directory to path
models_dir = osp.join(project_root, 'Models')
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from Retinotopy.read.read_HCPdata import read_HCP
from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from Retinotopy.functions.labels import labels
from models import (
    deepRetinotopy_Baseline,
    deepRetinotopy_OptionA,
    deepRetinotopy_OptionB,
    deepRetinotopy_OptionC
)


def parse_model_path(checkpoint_path):
    """
    Extract prediction type, hemisphere, model type, and seed from checkpoint path.
    
    Expected patterns:
    - {prediction_short}_{hemisphere}_{model_type}{myelination_suffix}{seed_suffix}_best_model_epoch{epoch}.pt
    - {prediction_short}_{hemisphere}_{model_type}{myelination_suffix}_best_model_epoch{epoch}.pt
    - {prediction_short}_{hemisphere}_{model_type}_best_model_epoch{epoch}.pt
    
    Examples:
    - ecc_Left_baseline_noMyelin_seed0_best_model_epoch66.pt
    - PA_Right_transolver_optionA_noMyelin_best_model_epoch50.pt
    - size_Left_transolver_optionC_noMyelin_seed1_best_model_epoch62.pt
    
    Returns:
        dict with keys: prediction, hemisphere, model_type, myelination, seed
    """
    filename = osp.basename(checkpoint_path)
    
    # Prediction type mapping
    prediction_map = {
        'ecc': 'eccentricity',
        'PA': 'polarAngle',
        'size': 'pRFsize',
        'pRFsize': 'pRFsize'
    }
    
    # Model type mapping
    model_map = {
        'baseline': 'baseline',
        'transolver_optionA': 'transolver_optionA',
        'transolver_optionB': 'transolver_optionB',
        'transolver_optionC': 'transolver_optionC',
        'optionA': 'transolver_optionA',
        'optionB': 'transolver_optionB',
        'optionC': 'transolver_optionC'
    }
    
    # Extract prediction short form (more flexible pattern)
    pred_match = re.match(r'^(ecc|PA|size|pRFsize)_', filename)
    if not pred_match:
        # Try alternative pattern: might have different prefix
        pred_match = re.search(r'_(ecc|PA|size|pRFsize)_', filename)
        if not pred_match:
            raise ValueError(f"Could not extract prediction type from filename: {filename}")
    pred_short = pred_match.group(1)
    prediction = prediction_map.get(pred_short)
    if not prediction:
        raise ValueError(f"Unknown prediction short form: {pred_short}")
    
    # Extract hemisphere (more flexible pattern)
    hemi_match = re.search(r'_(Left|Right|LH|RH|lh|rh)_', filename)
    if not hemi_match:
        raise ValueError(f"Could not extract hemisphere from filename: {filename}")
    hemisphere_str = hemi_match.group(1)
    if hemisphere_str.upper() in ['LEFT', 'LH']:
        hemisphere = 'Left'
    elif hemisphere_str.upper() in ['RIGHT', 'RH']:
        hemisphere = 'Right'
    else:
        raise ValueError(f"Unknown hemisphere: {hemisphere_str}")
    
    # Extract model type (more flexible pattern)
    model_match = re.search(r'_(baseline|transolver_optionA|transolver_optionB|transolver_optionC|optionA|optionB|optionC)(_|$)', filename)
    if not model_match:
        raise ValueError(f"Could not extract model type from filename: {filename}")
    model_str = model_match.group(1)
    model_type = model_map.get(model_str)
    if not model_type:
        raise ValueError(f"Unknown model type: {model_str}")
    
    # Check myelination
    # If explicitly contains 'noMyelin', then myelination=False
    # Otherwise, check if it contains 'myelin' (case insensitive)
    if 'noMyelin' in filename or 'no_myelin' in filename.lower():
        myelination = False
    elif 'myelin' in filename.lower():
        myelination = True
    else:
        # Default: assume no myelination if not specified
        myelination = False
    
    # Extract seed information
    # Pattern: _seed{number}_ or _seed{number} before _best_model
    seed_match = re.search(r'_seed(\d+)(_|$)', filename)
    if seed_match:
        seed = int(seed_match.group(1))
    else:
        # Default to seed0 if not found
        seed = 0
    
    return {
        'prediction': prediction,
        'hemisphere': hemisphere,
        'model_type': model_type,
        'myelination': myelination,
        'seed': seed
    }


def create_model(model_type, num_features=2, args=None):
    """Create model based on model_type"""
    if model_type == 'baseline':
        return deepRetinotopy_Baseline(num_features)
    elif model_type == 'transolver_optionA':
        return deepRetinotopy_OptionA(num_features)
    elif model_type == 'transolver_optionB':
        return deepRetinotopy_OptionB(num_features)
    elif model_type == 'transolver_optionC':
        if args is not None:
            return deepRetinotopy_OptionC(
                num_features=num_features,
                space_dim=3,
                n_layers=args.n_layers if hasattr(args, 'n_layers') else 8,
                n_hidden=args.n_hidden if hasattr(args, 'n_hidden') else 128,
                dropout=args.dropout if hasattr(args, 'dropout') else 0.0,
                n_head=args.n_heads if hasattr(args, 'n_heads') else 8,
                act='gelu',
                mlp_ratio=args.mlp_ratio if hasattr(args, 'mlp_ratio') else 1,
                slice_num=args.slice_num if hasattr(args, 'slice_num') else 64,
                ref=args.ref if hasattr(args, 'ref') else 8,
                unified_pos=bool(args.unified_pos if hasattr(args, 'unified_pos') else 0)
            )
        else:
            return deepRetinotopy_OptionC(num_features)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and return state dict"""
    if not osp.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        else:
            # Assume it's the state dict itself
            return checkpoint
    else:
        return checkpoint


def get_subject_index(subject_id, data_dir):
    """Get subject index from subject ID by reading list_subj file"""
    list_subj_path = osp.join(data_dir, '..', '..', 'list_subj')
    if not osp.exists(list_subj_path):
        # Try alternative location
        list_subj_path = osp.join(data_dir, 'list_subj')
    
    if not osp.exists(list_subj_path):
        raise FileNotFoundError(
            f"Could not find list_subj file. Tried: {list_subj_path}\n"
            f"Please ensure list_subj file exists in Retinotopy/data/ directory."
        )
    
    with open(list_subj_path, 'r') as fp:
        subjects = fp.read().split("\n")
    subjects = [s.strip() for s in subjects if s.strip()]
    
    # Find subject index
    try:
        index = subjects.index(subject_id)
        return index
    except ValueError:
        raise ValueError(
            f"Subject ID '{subject_id}' not found in list_subj.\n"
            f"Available subjects: {subjects[:10]}..." if len(subjects) > 10 else f"Available subjects: {subjects}"
        )


def load_data_from_converted(data_dir, subject_index, hemisphere, myelination, prediction):
    """Load data directly from Retinotopy/data/raw/converted folder"""
    # Get ROI masks
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(label_primary_visual_areas)
    
    # Load faces
    faces_R = labels(
        scipy.io.loadmat(osp.join(data_dir, 'tri_faces_R.mat'))['tri_faces_R'] - 1,
        index_R_mask
    )
    faces_L = labels(
        scipy.io.loadmat(osp.join(data_dir, 'tri_faces_L.mat'))['tri_faces_L'] - 1,
        index_L_mask
    )
    
    # Read HCP data
    data = read_HCP(
        data_dir,
        Hemisphere=hemisphere,
        index=subject_index,
        surface='mid',
        visual_mask_L=final_mask_L,
        visual_mask_R=final_mask_R,
        faces_L=faces_L,
        faces_R=faces_R,
        myelination=myelination,
        prediction=prediction,
        shuffle_seed=1  # Use default seed for consistency
    )
    
    # Apply transforms
    pre_transform = T.Compose([T.FaceToEdge()])
    transform = T.Cartesian()
    
    data = pre_transform(data)
    data = transform(data)
    
    return data


def run_inference(model, data, device):
    """Run inference on single data object"""
    model.eval()
    with torch.no_grad():
        pred = model(data.to(device)).detach().cpu()
    return pred


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end inference: extract info from checkpoint and run inference on training data'
    )
    parser.add_argument('--subject_index', type=int, default=None,
                        help='Subject index (0-based) to process. Mutually exclusive with --subject_id.')
    parser.add_argument('--subject_id', type=str, default=None,
                        help='Subject ID to process. Will look up index from list_subj file. Mutually exclusive with --subject_index.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to pre-trained model checkpoint (.pt file)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to Retinotopy/data/raw/converted directory. If None, uses default location.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for predictions. If None, saves in current directory.')
    
    # Optional model hyperparameters (for transolver_optionC)
    parser.add_argument('--n_layers', type=int, default=8,
                        help='Number of Transolver blocks (default: 8)')
    parser.add_argument('--n_hidden', type=int, default=128,
                        help='Hidden dimension (default: 128)')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--slice_num', type=int, default=64,
                        help='Number of slice tokens in Physics Attention (default: 64)')
    parser.add_argument('--mlp_ratio', type=int, default=1,
                        help='MLP ratio in Transolver blocks (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (default: 0.0)')
    parser.add_argument('--ref', type=int, default=8,
                        help='Reference grid size for unified_pos (default: 8)')
    parser.add_argument('--unified_pos', type=int, default=0,
                        help='Use unified position encoding (0=False, 1=True, default: 0)')
    
    args = parser.parse_args()
    
    # Validate subject input
    if args.subject_index is None and args.subject_id is None:
        parser.error("Either --subject_index or --subject_id must be provided.")
    if args.subject_index is not None and args.subject_id is not None:
        parser.error("--subject_index and --subject_id are mutually exclusive.")
    
    # Determine data directory
    if args.data_dir is None:
        # Use default location: Retinotopy/data/raw/converted
        args.data_dir = osp.join(project_root, 'Retinotopy', 'data', 'raw', 'converted')
    
    if not osp.exists(args.data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {args.data_dir}\n"
            f"Please ensure Retinotopy/data/raw/converted directory exists or specify --data_dir."
        )
    
    # Parse model path to extract information
    print("="*60)
    print("Extracting information from checkpoint path...")
    print("="*60)
    try:
        model_info = parse_model_path(args.checkpoint_path)
        print(f"Prediction type: {model_info['prediction']}")
        print(f"Hemisphere: {model_info['hemisphere']}")
        print(f"Model type: {model_info['model_type']}")
        print(f"Myelination: {model_info['myelination']}")
        print(f"Seed: {model_info['seed']}")
    except Exception as e:
        print(f"Error parsing checkpoint path: {e}")
        print(f"Checkpoint path: {args.checkpoint_path}")
        return
    
    # Determine subject index
    if args.subject_id is not None:
        print(f"\nLooking up subject index for ID: {args.subject_id}")
        try:
            subject_index = get_subject_index(args.subject_id, args.data_dir)
            print(f"Found subject index: {subject_index}")
        except Exception as e:
            print(f"Error looking up subject index: {e}")
            return
    else:
        subject_index = args.subject_index
        print(f"\nUsing subject index: {subject_index}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Determine number of features
    num_features = 2 if model_info['myelination'] else 1
    
    # Create model
    print(f"\nCreating model: {model_info['model_type']}")
    model = create_model(model_info['model_type'], num_features=num_features, args=args).to(device)
    
    # Load checkpoint
    try:
        state_dict = load_checkpoint(args.checkpoint_path, device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get ROI masks for output
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(label_primary_visual_areas)
    
    # Determine which mask to use
    if model_info['hemisphere'] in ['Left', 'LH', 'left', 'lh']:
        visual_mask = final_mask_L
        hemi_str = 'lh'
    else:
        visual_mask = final_mask_R
        hemi_str = 'rh'
    
    # Load data from converted folder
    print(f"\nLoading data from: {args.data_dir}")
    print(f"  Subject index: {subject_index}")
    print(f"  Prediction: {model_info['prediction']}")
    print(f"  Hemisphere: {model_info['hemisphere']}")
    print(f"  Myelination: {model_info['myelination']}")
    
    try:
        data = load_data_from_converted(
            data_dir=args.data_dir,
            subject_index=subject_index,
            hemisphere=model_info['hemisphere'],
            myelination=model_info['myelination'],
            prediction=model_info['prediction']
        )
        
        print(f"Data loaded successfully. Number of vertices: {data.x.shape[0]}")
        
        # Run inference
        print("\nRunning inference...")
        prediction = run_inference(model, data, device)
        
        # Get prediction values
        pred_values = prediction.view(-1).numpy()
        
        print(f"Inference completed. Predicted {len(pred_values)} vertices.")
        print(f"Prediction statistics:")
        print(f"  Min: {pred_values.min():.4f}")
        print(f"  Max: {pred_values.max():.4f}")
        print(f"  Mean: {pred_values.mean():.4f}")
        print(f"  Std: {pred_values.std():.4f}")
        
        # Create output directory with seed subfolder
        if args.output_dir:
            base_output_dir = Path(args.output_dir)
        else:
            base_output_dir = Path.cwd() / 'fslr_inference_output'
        
        # Create seed-specific subdirectory
        seed_dir = base_output_dir / f"seed{model_info['seed']}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        output_dir = seed_dir
        
        # Create output filename
        prediction_short = {
            'eccentricity': 'ecc',
            'polarAngle': 'PA',
            'pRFsize': 'size'
        }[model_info['prediction']]
        
        myelination_suffix = '_myelin' if model_info['myelination'] else '_noMyelin'
        model_suffix = f'_{model_info["model_type"]}' if model_info['model_type'] != 'baseline' else ''
        
        # Use subject_id if available, otherwise use index
        subject_identifier = args.subject_id if args.subject_id else f"subj{subject_index}"
        
        output_filename = f'{subject_identifier}_{prediction_short}_{model_info["hemisphere"]}{model_suffix}{myelination_suffix}_prediction.pt'
        output_path = output_dir / output_filename
        
        # Save prediction as .pt file
        save_dict = {
            'subject_id': subject_identifier,
            'subject_index': subject_index,
            'prediction_type': model_info['prediction'],
            'hemisphere': model_info['hemisphere'],
            'model_type': model_info['model_type'],
            'myelination': model_info['myelination'],
            'seed': model_info['seed'],
            'checkpoint_path': args.checkpoint_path,
            'predicted_values': pred_values,
            'visual_mask': visual_mask,
            'num_vertices': len(pred_values)
        }
        
        torch.save(save_dict, output_path)
        print(f"\nPrediction saved to: {output_path}")
        print(f"  File contains: predicted_values, visual_mask, and metadata")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("Inference completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
