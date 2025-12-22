#!/usr/bin/env python
"""
Inference script for FreeSurfer subjects using pre-trained deepRetinotopy models.

This script processes FreeSurfer folders and generates retinotopic predictions
using pre-trained models.
"""

import os
import os.path as osp
import sys
import argparse
import torch
import torch_geometric.transforms as T
import numpy as np
import nibabel as nib
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add Models directory to path
models_dir = osp.dirname(osp.abspath(__file__))
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from Retinotopy.dataset.FreesurferDataset import FreesurferDataset
from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from torch_geometric.loader import DataLoader
from models import (
    deepRetinotopy_Baseline,
    deepRetinotopy_OptionA,
    deepRetinotopy_OptionB,
    deepRetinotopy_OptionC
)


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
                n_layers=args.n_layers,
                n_hidden=args.n_hidden,
                dropout=args.dropout,
                n_head=args.n_heads,
                act='gelu',
                mlp_ratio=args.mlp_ratio,
                slice_num=args.slice_num,
                ref=args.ref,
                unified_pos=bool(args.unified_pos)
            )
        else:
            return deepRetinotopy_OptionC(num_features)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def inference(model, data_loader, device):
    """Run inference on data"""
    model.eval()
    y_hat = []
    with torch.no_grad():
        for data in data_loader:
            pred = model(data.to(device)).detach().cpu()
            y_hat.append(pred)
    return y_hat


def main():
    parser = argparse.ArgumentParser(
        description='Inference with pre-trained deepRetinotopy models on FreeSurfer data'
    )
    parser.add_argument('--freesurfer_dir', type=str, required=True,
                        help='Path to FreeSurfer subjects directory')
    parser.add_argument('--subject_id', type=str, default=None,
                        help='Single subject ID to process. If None, processes all subjects.')
    parser.add_argument('--model_type', type=str, default='baseline',
                        choices=['baseline', 'transolver_optionA', 'transolver_optionB', 'transolver_optionC'],
                        help='Model architecture type')
    parser.add_argument('--prediction', type=str, default='eccentricity',
                        choices=['eccentricity', 'polarAngle', 'pRFsize'],
                        help='Prediction target')
    parser.add_argument('--hemisphere', type=str, default='Left',
                        choices=['Left', 'Right'],
                        help='Hemisphere to use for prediction')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to pre-trained model checkpoint (.pt file)')
    parser.add_argument('--myelination', type=str, default='True',
                        choices=['True', 'False', 'true', 'false', '1', '0'],
                        help='Whether myelination was used during training (default: True)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for predictions. If None, saves in FreeSurfer directory structure.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    
    # Model architecture hyperparameters (for Transolver models)
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
    
    # Convert myelination string to boolean
    args.myelination = args.myelination.lower() in ['true', '1']
    
    # Determine number of features
    num_features = 2 if args.myelination else 1
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {args.model_type}")
    model = create_model(args.model_type, num_features=num_features, args=args).to(device)
    
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    print("Model loaded successfully")
    
    # Get list of subjects
    if args.subject_id:
        list_subs = [args.subject_id]
    else:
        # List all subject directories
        list_subs = []
        for item in os.listdir(args.freesurfer_dir):
            item_path = osp.join(args.freesurfer_dir, item)
            if osp.isdir(item_path) and item != 'fsaverage' and not item.startswith('.'):
                # Check if subject has required surf directory
                surf_path = osp.join(item_path, 'surf')
                if osp.exists(surf_path):
                    list_subs.append(item)
    
    if len(list_subs) == 0:
        raise ValueError("No valid subjects found. Please check --freesurfer_dir and --subject_id.")
    
    print(f"Found {len(list_subs)} subject(s) to process: {list_subs}")
    
    # Get ROI masks
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(label_primary_visual_areas)
    
    # Determine which mask to use
    if args.hemisphere in ['Left', 'LH', 'left', 'lh']:
        visual_mask = final_mask_L
        hemi_str = 'lh'
    else:
        visual_mask = final_mask_R
        hemi_str = 'rh'
    
    # Process each subject
    for subject_id in list_subs:
        print(f"\n{'='*60}")
        print(f"Processing subject: {subject_id}")
        print(f"{'='*60}")
        
        try:
            # Create dataset for this subject
            pre_transform = T.Compose([T.FaceToEdge()])
            transform = T.Cartesian()
            
            dataset = FreesurferDataset(
                root=args.freesurfer_dir,
                transform=transform,
                pre_transform=pre_transform,
                list_subs=[subject_id],
                myelination=args.myelination,
                prediction=args.prediction,
                hemisphere=args.hemisphere
            )
            
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            
            # Run inference
            print("Running inference...")
            predictions = inference(model, data_loader, device)
            
            # Get prediction values
            pred_values = predictions[0].view(-1).numpy()
            
            # Create output array for full hemisphere (32492 vertices)
            num_hemi_nodes = 32492
            output_array = np.full(num_hemi_nodes, -1.0, dtype=np.float32)
            output_array[visual_mask == 1] = pred_values
            
            # Determine output directory
            if args.output_dir:
                output_dir = osp.join(args.output_dir, subject_id, 'deepRetinotopy')
            else:
                output_dir = osp.join(args.freesurfer_dir, subject_id, 'deepRetinotopy')
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Load template GIFTI file for structure
            template_file = osp.join(args.freesurfer_dir, subject_id, 'surf',
                                   f'{subject_id}.curvature-midthickness.{hemi_str}.32k_fs_LR.func.gii')
            
            if not osp.exists(template_file):
                raise FileNotFoundError(f"Template file not found: {template_file}")
            
            template = nib.load(template_file)
            
            # Create output filename
            myelination_suffix = '_myelin' if args.myelination else ''
            model_suffix = f'_{args.model_type}' if args.model_type != 'baseline' else ''
            output_filename = f'{subject_id}.predicted_{args.prediction}_{hemi_str}{myelination_suffix}{model_suffix}.func.gii'
            output_path = osp.join(output_dir, output_filename)
            
            # Save prediction
            template.agg_data()[:] = output_array
            nib.save(template, output_path)
            
            print(f"Prediction saved to: {output_path}")
            print(f"Prediction statistics:")
            print(f"  Min: {pred_values.min():.4f}")
            print(f"  Max: {pred_values.max():.4f}")
            print(f"  Mean: {pred_values.mean():.4f}")
            print(f"  Std: {pred_values.std():.4f}")
            
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Inference completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()



