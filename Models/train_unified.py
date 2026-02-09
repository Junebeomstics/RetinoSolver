import os
import os.path as osp
import sys
import time
import argparse
import copy
import random
import torch
import torch_geometric.transforms as T
import numpy as np
import scipy.stats
from contextlib import contextmanager

# Astropy import (optional, for circular correlation)
try:
    from astropy.stats import circcorrcoef
    from astropy import units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("Warning: Astropy not available. Circular correlation will use scipy alternative.")

# Wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: Wandb not available. Logging will be skipped.")

# =============================
# Argument Parser Setup
# =============================

# Helper function to parse boolean arguments correctly
def str_to_bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(
    description="Unified training script for deepRetinotopy models with Transolver variants."
)
parser.add_argument('--model_type', type=str, default='baseline',
                    choices=['baseline', 'transolver_optionA', 'transolver_optionB', 'transolver_optionC'],
                    help='Model architecture type')
parser.add_argument('--prediction', type=str, default='eccentricity',
                    choices=['eccentricity', 'polarAngle', 'pRFsize'],
                    help='Prediction target')
parser.add_argument('--hemisphere', type=str, default='Left', choices=['Left', 'Right'],
                    help='Hemisphere to use for prediction')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size for loader')
parser.add_argument('--n_epochs', type=int, default=200,
                    help='Total number of epochs to train')
parser.add_argument('--lr_init', type=float, default=0.001,
                    help='Initial learning rate')
parser.add_argument('--lr_decay_epoch', type=int, default=250,
                    help='Epoch at which learning rate decays (for step scheduler)')
parser.add_argument('--lr_decay', type=float, default=0.0001,
                    help='Learning rate after decay (for step scheduler)')
parser.add_argument('--scheduler', type=str, default='cosine',
                    choices=['step', 'cosine', 'onecycle'],
                    help='Learning rate scheduler type: step, cosine, or onecycle')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay for optimizer')
parser.add_argument('--optimizer', type=str, default='AdamW',
                    choices=['Adam', 'AdamW'],
                    help='Optimizer type: Adam or AdamW')
parser.add_argument('--max_grad_norm', type=float, default=None,
                    help='Maximum gradient norm for clipping (default: None, no clipping)')
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
parser.add_argument('--interm_save_every', type=int, default=25,
                    help='Interval (epoch) for saving intermediate predictions')
parser.add_argument('--output_dir', type=str, default='./output',
                    help='Base directory to save outputs')
parser.add_argument('--n_examples', type=int, default=181,
                    help='Number of examples per split')
parser.add_argument('--myelination', type=str_to_bool, default=True,
                    help='Use myelination as feature (default: True). Accepts: True/False, yes/no, 1/0')
parser.add_argument('--use_wandb', action='store_true',
                    help='Enable Wandb logging')
parser.add_argument('--wandb_project', type=str, default='deepRetinotopy',
                    help='Wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None,
                    help='Wandb entity/team name (optional)')
parser.add_argument('--run_test', type=str, default='True',
                    choices=['True', 'False', 'true', 'false', '1', '0'],
                    help='Run test set evaluation after training (default: True)')
parser.add_argument('--early_stopping_patience', type=int, default=30,
                    help='Number of epochs to wait before early stopping if no improvement (default: 30)')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to local checkpoint file (.pt) to load (if provided, only test will be run)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility and data splitting (default: 42). For data splitting, if you want to use default seed=1, set --seed 1')
parser.add_argument('--save_predicted_map', type=str_to_bool, default=False,
                    help='Save predicted maps as separate numpy files (default: False). Accepts: True/False, yes/no, 1/0')
parser.add_argument('--r2_scaling', type=str_to_bool, default=True,
                    help='Use R2 scaling in loss calculation during training (default: True). Accepts: True/False, yes/no, 1/0')
parser.add_argument('--use_freesurfer_curv', type=str_to_bool, default=False,
                    help='Use FreeSurfer GIFTI curvature data (gifti_curv_all.mat) instead of CIFTI curvature (default: False). Results will be loaded from processed_fs folder. Accepts: True/False, yes/no, 1/0')

args = parser.parse_args()

# Set random seed for reproducibility
def set_seed(seed):
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(args.seed)
print(f"Random seed set to: {args.seed}")

# Use the same seed for data splitting
# Note: For backward compatibility, if you want to use default data seed=1, 
# explicitly set --seed 1. Otherwise, the specified seed will be used for both 
# model training randomness and data splitting.
data_split_seed = args.seed
print(f"Data splitting seed: {data_split_seed} (same as training seed)")

# Convert run_test string to boolean
args.run_test = args.run_test.lower() in ['true', '1']

# Test-only mode if local checkpoint path is provided
test_only_mode = args.checkpoint_path is not None

# =============================
# Path, Import, Dataset Preparation
# =============================

# Add project root directory to sys.path as absolute path
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add Models directory to sys.path (for importing models module)
models_dir = osp.dirname(osp.abspath(__file__))
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from Retinotopy.functions.plusFovea import add_fovea, add_fovea_R
from Retinotopy.functions.error_metrics import smallest_angle
from torch_geometric.loader import DataLoader
import scipy.io

# Import dataset class based on use_freesurfer_curv flag
if args.use_freesurfer_curv:
    from Retinotopy.dataset.HCP_3sets_ROI_fs import RetinotopyFS as Retinotopy
    print("Using FreeSurfer GIFTI curvature data (processed_fs folder)")
else:
    from Retinotopy.dataset.HCP_3sets_ROI import Retinotopy
    print("Using CIFTI curvature data (processed folder)")

# Import models from separate modules
from models import (
    deepRetinotopy_Baseline,
    deepRetinotopy_OptionA,
    deepRetinotopy_OptionB,
    deepRetinotopy_OptionC
)

# Create Wandb run (optional)
wandb_run = None

# Define myelination suffix for naming (used in wandb run name, folder names, and file names)
myelination_suffix = "" if args.myelination else "_noMyelin"

# Add seed suffix to project name and file names if seed is not default (42)
# This helps distinguish experiments with different seeds
data_seed_suffix = f"_seed{data_split_seed}" if data_split_seed != 42 else ""
wandb_project_name = args.wandb_project + data_seed_suffix if data_split_seed != 42 else args.wandb_project

# Initialize Wandb if enabled
if args.use_wandb and WANDB_AVAILABLE and not test_only_mode:
    try:
        # Create run name from model config (include myelination status and data seed)
        run_name = f"{args.prediction}_{args.hemisphere}_{args.model_type}{myelination_suffix}{data_seed_suffix}"
        
        # Initialize wandb
        wandb.init(
            project=wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            reinit=True
        )
        wandb_run = wandb.run
        print(f"Wandb initialized: {wandb_run.name} (ID: {wandb_run.id})")
    except Exception as e:
        print(f"Warning: Failed to initialize Wandb: {e}. Continuing without logging.")
        wandb_run = None
elif args.use_wandb and not WANDB_AVAILABLE:
    print("Warning: Wandb requested but not available. Skipping Wandb logging.")

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Retinotopy', 'data')

pre_transform = T.Compose([T.FaceToEdge()])
train_dataset = Retinotopy(
    path, 'Train',
    transform=T.Cartesian(),
    pre_transform=pre_transform,
    n_examples=args.n_examples,
    prediction=args.prediction,
    myelination=args.myelination,
    hemisphere=args.hemisphere,
    seed=data_split_seed
)
dev_dataset = Retinotopy(
    path, 'Development',
    transform=T.Cartesian(),
    pre_transform=pre_transform,
    n_examples=args.n_examples,
    prediction=args.prediction,
    myelination=args.myelination,
    hemisphere=args.hemisphere,
    seed=data_split_seed
)

# Setup DataLoader with reproducible shuffling
def seed_worker(worker_id):
    """Fix worker seed for DataLoader reproducibility"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Create generator with data_split_seed for reproducible shuffling
g = torch.Generator()
g.manual_seed(data_split_seed)

train_loader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g
)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

# Test dataset (only if run_test is True)
test_dataset = None
test_loader = None
test_subject_names = None
if args.run_test:
    test_dataset = Retinotopy(
        path, 'Test',
        transform=T.Cartesian(),
        pre_transform=pre_transform,
        n_examples=args.n_examples,
        prediction=args.prediction,
        myelination=args.myelination,
        hemisphere=args.hemisphere,
        seed=data_split_seed
    )
    # Use batch_size=args.batch_size for test set (note: may want batch_size=1 for per-subject processing)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load test subject names from file
    subject_splits_dir = osp.join(path, 'subject_splits', f'seed{data_split_seed}')
    test_subjects_file = osp.join(subject_splits_dir, 'test_subjects.txt')
    if osp.exists(test_subjects_file):
        with open(test_subjects_file, 'r') as f:
            test_subject_names = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(test_subject_names)} test subject names from {test_subjects_file}")
    else:
        print(f"Warning: Test subject names file not found: {test_subjects_file}")
        print("Subject names will not be included in saved results.")
        test_subject_names = None

# Wandb config logging (already done in wandb.init, but add additional configs)
if wandb_run is not None:
    wandb.config.update({
        "architecture": args.model_type,
        "loss_fn": "SmoothL1Loss",
        "r2_scaling": args.r2_scaling,
        "dataset": "HCP_3sets_ROI",
        "train_size": len(train_dataset),
        "dev_size": len(dev_dataset),
    })
    if args.run_test:
        wandb.config.update({"test_size": len(test_dataset)})

# =============================
# Model Size Calculation
# =============================
def count_parameters(model):
    """Count total and trainable parameters in the model"""
    total_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
        if parameter.requires_grad:
            trainable_params += params
    return total_params, trainable_params

def calculate_model_size_mb(model):
    """Calculate model size in MB (assuming float32, 4 bytes per parameter)"""
    total_params, _ = count_parameters(model)
    # Each float32 parameter is 4 bytes
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

# =============================
# Model Factory
# =============================
def create_model(model_type, num_features=2, args=None):
    """Create model based on model_type"""
    if model_type == 'baseline':
        return deepRetinotopy_Baseline(num_features)
    elif model_type == 'transolver_optionA':
        return deepRetinotopy_OptionA(num_features)
    elif model_type == 'transolver_optionB':
        return deepRetinotopy_OptionB(num_features)
    elif model_type == 'transolver_optionC':
        # Pass architecture hyperparameters for Transolver models
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


# =============================
# Training Setup
# =============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(args.model_type, num_features=2 if args.myelination else 1, args=args).to(device)

# Calculate and log model size
total_params, trainable_params = count_parameters(model)
model_size_mb = calculate_model_size_mb(model)
print(f"Model size:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size: {model_size_mb:.2f} MB")

# Log model size to Wandb
if wandb_run is not None:
    wandb.config.update({
        "model/total_parameters": total_params,
        "model/trainable_parameters": trainable_params,
        "model/size_mb": model_size_mb
    })

# Setup optimizer
if args.optimizer == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)

# Setup learning rate scheduler
if args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
elif args.scheduler == 'onecycle':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr_init, 
        epochs=args.n_epochs,
        steps_per_epoch=len(train_loader)
    )
else:  # step scheduler
    scheduler = None  # Step decay is handled manually in train function

# Create output directory structure: {output_dir}/{wandb_run_name}/ or {output_dir}/{model_type}_{prediction}_{hemisphere}[_noMyelin][_seed{seed}]/
if args.prediction == 'eccentricity':
    prediction_short = 'ecc'
elif args.prediction == 'polarAngle':
    prediction_short = 'PA'
elif args.prediction == 'pRFsize':
    prediction_short = 'size'
else:
    prediction_short = args.prediction
# myelination_suffix is already defined above
if wandb_run is not None:
    output_subdir = wandb_run.name  # Use meaningful run name instead of random ID (already includes myelination info)
else:
    output_subdir = f"{args.prediction}_{args.hemisphere}_{args.model_type}{myelination_suffix}{data_seed_suffix}"
output_path = osp.join(osp.dirname(osp.realpath(__file__)), args.output_dir, output_subdir)
if not osp.exists(output_path):
    os.makedirs(output_path)

# Setup logging to file for training output
class Tee:
    """Class to write to both file and stdout"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stdout.write(text)
        self.stdout.flush()
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def close(self):
        self.file.close()

# Create log file path
log_file_path = osp.join(output_path, 'training_log.txt')
# Save original stdout
original_stdout = sys.stdout
# Create Tee object to write to both file and stdout (only during training, not test-only mode)
log_tee = None
if not test_only_mode:
    log_tee = Tee(log_file_path)
    # Redirect stdout to Tee
    sys.stdout = log_tee

# Load checkpoint if in test-only mode
if test_only_mode:
    print("\n" + "="*50)
    if args.checkpoint_path is not None:
        print("Test-only mode: Loading checkpoint from local file")
        print("="*50)
        
        try:
            # Check if checkpoint file exists
            if not osp.exists(args.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
            
            print(f"Loading checkpoint from: {args.checkpoint_path}")
            
            # Load checkpoint into model
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # If checkpoint contains 'model_state_dict' key (full checkpoint)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("Loaded model from checkpoint (full checkpoint format)")
                    # Try to extract epoch info if available
                    if 'epoch' in checkpoint:
                        print(f"Checkpoint epoch: {checkpoint['epoch']}")
                    if 'best_epoch' in checkpoint:
                        print(f"Best epoch: {checkpoint['best_epoch']}")
                # If checkpoint is just state_dict
                elif any(key.startswith(('conv', 'lin', 'transformer', 'encoder', 'decoder')) for key in checkpoint.keys()):
                    model.load_state_dict(checkpoint)
                    print("Loaded model from checkpoint (state_dict format)")
                else:
                    # Try to load as state_dict anyway
                    model.load_state_dict(checkpoint)
                    print("Loaded model from checkpoint")
            else:
                # Assume it's a state_dict
                model.load_state_dict(checkpoint)
                print("Loaded model from checkpoint (state_dict format)")
            
            print(f"Model successfully loaded from checkpoint.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from local file: {e}")
    
    # Skip training and go directly to testing
    print("\nSkipping training, proceeding to test evaluation...")

print(f"Output directory: {output_path}")
print(f"Model type: {args.model_type}")
print(f"Prediction: {args.prediction}")
print(f"Hemisphere: {args.hemisphere}")

if not test_only_mode:
    print(f"R2 scaling in loss: {args.r2_scaling}")
    print(f"Optimizer: {args.optimizer} (lr={args.lr_init}, weight_decay={args.weight_decay})")
    print(f"Scheduler: {args.scheduler}")
    if args.scheduler == 'step':
        print(f"  Step decay at epoch {args.lr_decay_epoch}: {args.lr_init} -> {args.lr_decay}")
    elif args.scheduler == 'cosine':
        print(f"  CosineAnnealingLR with T_max={args.n_epochs}")
    elif args.scheduler == 'onecycle':
        print(f"  OneCycleLR with max_lr={args.lr_init}, steps_per_epoch={len(train_loader)}")
    if args.max_grad_norm is not None:
        print(f"Gradient clipping: max_norm={args.max_grad_norm}")
    if args.model_type in ['transolver_optionA', 'transolver_optionB', 'transolver_optionC']:
        print(f"Model architecture:")
        print(f"  n_layers={args.n_layers}, n_hidden={args.n_hidden}, n_heads={args.n_heads}")
        print(f"  slice_num={args.slice_num}, mlp_ratio={args.mlp_ratio}, dropout={args.dropout}")
        print(f"  ref={args.ref}, unified_pos={bool(args.unified_pos)}")


# =============================
# Training Functions
# =============================
def get_current_lr(optimizer):
    # Returns current learning rate (handles param group lists)
    return optimizer.param_groups[0]['lr']

def train(epoch):
    model.train()

    # Step scheduler: manual learning rate decay
    if args.scheduler == 'step' and epoch == args.lr_decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr_decay
        if wandb_run is not None:
            wandb.log({"lr_current": args.lr_decay}, step=epoch)

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        R2 = data.R2.view(-1)
        threshold = R2.view(-1) > 2.2

        loss = torch.nn.SmoothL1Loss()
        # Apply R2 scaling if enabled
        if args.r2_scaling:
            output_loss = loss(R2 * model(data), R2 * data.y.view(-1))
        else:
            output_loss = loss(model(data), data.y.view(-1))
        output_loss.backward()

        # Gradient clipping (same as Transolver)
        if args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        MAE = torch.mean(abs(
            data.y.view(-1)[threshold == 1] - model(data)[threshold == 1])).item()

        optimizer.step()
        
        # OneCycleLR scheduler steps per batch
        if args.scheduler == 'onecycle':
            scheduler.step()
            if wandb_run is not None:
                wandb.log({"lr_current": scheduler.get_last_lr()[0]}, step=epoch)
    
    # CosineAnnealingLR scheduler steps per epoch
    if args.scheduler == 'cosine':
        scheduler.step()
        if wandb_run is not None:
            wandb.log({"lr_current": scheduler.get_last_lr()[0]}, step=epoch)
    
    # Wandb logging
    if wandb_run is not None:
        wandb.log({
            "train/loss": output_loss.detach().cpu().item(),
            "train/mae": MAE
        }, step=epoch)
    
    return output_loss.detach(), MAE


def validate():
    model.eval()
    MeanAbsError = 0
    MeanAbsError_thr = 0
    MeanAbsError_no_thr = 0
    y = []
    y_hat = []
    R2_plot = []

    for data in dev_loader:
        pred = model(data.to(device)).detach()
        y_hat.append(pred)
        y.append(data.to(device).y.view(-1))

        R2 = data.R2.view(-1)
        R2_plot.append(R2)
        threshold = R2.view(-1) > 2.2
        threshold2 = R2.view(-1) > 17

        # MAE without thresholding
        if args.prediction == 'polarAngle':
            # Use smallest_angle for polar angle (circular data)
            pred_np = pred.cpu().numpy().flatten()
            y_np = data.to(device).y.view(-1).cpu().numpy().flatten()
            # Convert degrees to radians for smallest_angle
            pred_rad = np.deg2rad(pred_np)
            y_rad = np.deg2rad(y_np)
            angle_diff = smallest_angle(y_rad, pred_rad)  # Returns degrees
            MAE_no_thr = np.mean(angle_diff)
        else:
            MAE_no_thr = torch.mean(abs(data.to(device).y.view(-1) - pred.view(-1))).item()
        MeanAbsError_no_thr += MAE_no_thr

        # MAE with R2 > 2.2 threshold
        if args.prediction == 'polarAngle':
            pred_thr_np = pred[threshold == 1].cpu().numpy().flatten()
            y_thr_np = data.to(device).y.view(-1)[threshold == 1].cpu().numpy().flatten()
            if len(pred_thr_np) > 0:
                pred_thr_rad = np.deg2rad(pred_thr_np)
                y_thr_rad = np.deg2rad(y_thr_np)
                angle_diff_thr = smallest_angle(y_thr_rad, pred_thr_rad)
                MAE = np.mean(angle_diff_thr)
            else:
                MAE = float('nan')
        else:
            MAE = torch.mean(abs(data.to(device).y.view(-1)[threshold == 1] - pred[
                threshold == 1])).item()
        MeanAbsError += MAE

        # MAE with R2 > 17 threshold
        if args.prediction == 'polarAngle':
            pred_thr2_np = pred[threshold2 == 1].cpu().numpy().flatten()
            y_thr2_np = data.to(device).y.view(-1)[threshold2 == 1].cpu().numpy().flatten()
            if len(pred_thr2_np) > 0:
                pred_thr2_rad = np.deg2rad(pred_thr2_np)
                y_thr2_rad = np.deg2rad(y_thr2_np)
                angle_diff_thr2 = smallest_angle(y_thr2_rad, pred_thr2_rad)
                MAE_thr = np.mean(angle_diff_thr2)
            else:
                MAE_thr = float('nan')
        else:
            MAE_thr = torch.mean(abs(
                data.to(device).y.view(-1)[threshold2 == 1] - pred[
                    threshold2 == 1])).item()
        MeanAbsError_thr += MAE_thr

    test_MAE = MeanAbsError / len(dev_loader)
    test_MAE_thr = MeanAbsError_thr / len(dev_loader)
    test_MAE_no_thr = MeanAbsError_no_thr / len(dev_loader)
    output = {'Predicted_values': y_hat, 'Measured_values': y, 'R2': R2_plot,
              'MAE': test_MAE, 'MAE_thr': test_MAE_thr, 'MAE_no_thr': test_MAE_no_thr}
    
    return output

# =============================
# Load Eccentricity 1-8 Mask
# =============================
def load_eccentricity_1to8_mask(hemisphere):
    """
    Load eccentricity 1-8 mask (eccentricity between 1 and 8 degrees).
    
    Args:
        hemisphere: 'Left' or 'Right'
    
    Returns:
        ecc_1to8_mask: Boolean mask for eccentricity 1-8 range indexed by ROI vertices (len(roi_indices),)
    """
    # Path to mask file
    mask_file = osp.join(
        osp.dirname(osp.realpath(__file__)), 
        '..', 
        'Manuscript', 
        'plots', 
        'output',
        f'MaskEccentricity_above1below8ecc_{"LH" if hemisphere == "Left" else "RH"}.npz'
    )
    
    if not osp.exists(mask_file):
        raise FileNotFoundError(f"Eccentricity 1-8 mask file not found: {mask_file}")
    
    # Load mask (full hemisphere mask: 32492,)
    ecc_1to8_full_mask = np.load(mask_file)['list']
    
    # Load ROI mask to get the full surface indices
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(label_primary_visual_areas)
    
    # Select hemisphere mask
    if hemisphere == 'Left':
        roi_mask = final_mask_L
        roi_indices = index_L_mask
    else:
        roi_mask = final_mask_R
        roi_indices = index_R_mask
    
    # Get intersection with ROI mask (only vertices that are both in ROI and ecc 1-8)
    ecc_1to8_roi_mask = ecc_1to8_full_mask & (roi_mask > 0)
    
    # Create a boolean mask for ROI indices that are in ecc 1-8 range
    # roi_indices contains the full surface indices of ROI vertices
    # We need to create a mask of length len(roi_indices) indicating which ROI vertices are in ecc 1-8
    ecc_1to8_roi_bool_mask = np.array([ecc_1to8_roi_mask[idx] for idx in roi_indices], dtype=bool)
    
    return ecc_1to8_roi_bool_mask

# =============================
# Load V1-V3 Mask
# =============================
def load_V1V2V3_mask(hemisphere):
    """
    Load V1, V2, V3 mask using add_fovea function (same approach as notebook).
    
    Args:
        hemisphere: 'Left' or 'Right'
    
    Returns:
        v1v2v3_mask: Boolean mask for V1-V3 areas indexed by ROI vertices (len(roi_indices),)
    """
    # Define primary visual areas (same as notebook)
    primary_visual_areas = ['V1d', 'V1v', 'fovea_V1', 'V2d', 'V2v', 'fovea_V2', 'V3d', 'V3v', 'fovea_V3']
    
    # Load ROI mask to get the full surface indices
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(label_primary_visual_areas)
    
    # Get V1, V2, V3 masks using add_fovea function
    if hemisphere == 'Left':
        V1, V2, V3 = add_fovea(primary_visual_areas)
        roi_mask = final_mask_L
        roi_indices = index_L_mask
    else:
        V1, V2, V3 = add_fovea_R(primary_visual_areas)
        roi_mask = final_mask_R
        roi_indices = index_R_mask
    
    # Combine V1, V2, V3 into a single mask (same as notebook)
    V1V2V3_mask = ((V1 > 0) | (V2 > 0) | (V3 > 0)).astype(bool)
    
    # Get intersection with ROI mask (only vertices that are both in ROI and V1-V3)
    v1v2v3_roi_mask = V1V2V3_mask & (roi_mask > 0)
    
    # Create a boolean mask for ROI indices that are in V1-V3
    # roi_indices contains the full surface indices of ROI vertices
    # We need to create a mask of length len(roi_indices) indicating which ROI vertices are in V1-V3
    v1v2v3_roi_bool_mask = np.array([v1v2v3_roi_mask[idx] for idx in roi_indices], dtype=bool)
    
    return v1v2v3_roi_bool_mask

# =============================
# Save Predicted Maps
# =============================
def save_predicted_maps(predicted_values, measured_values, R2_values, subject_names, output_path, 
                        prediction_type, hemisphere, model_type, myelination_suffix, data_seed_suffix, epoch=None):
    """
    Save predicted maps as numpy arrays for each subject.
    
    Args:
        predicted_values: List of tensors, one per subject (from test_loader batches)
        measured_values: List of tensors, one per subject (from test_loader batches)
        R2_values: List of tensors, one per subject with R2 values (from test_loader batches)
        subject_names: List of subject names/IDs (optional)
        output_path: Directory to save predicted maps
        prediction_type: 'eccentricity', 'polarAngle', or 'pRFsize'
        hemisphere: 'Left' or 'Right'
        model_type: Model type string
        myelination_suffix: Suffix for myelination (e.g., '_noMyelin' or '')
        data_seed_suffix: Suffix for data seed (e.g., '_seed0' or '')
        epoch: Epoch number (optional, for best model vs final model)
    """
    # Create directory for predicted maps if it doesn't exist
    predicted_maps_dir = osp.join(output_path, 'predicted_maps')
    os.makedirs(predicted_maps_dir, exist_ok=True)
    
    # Determine file prefix
    prediction_short = 'ecc' if prediction_type == 'eccentricity' else ('PA' if prediction_type == 'polarAngle' else 'size')
    epoch_str = f'_epoch{epoch}' if epoch is not None else ''
    base_filename = f'{prediction_short}_{hemisphere}_{model_type}{myelination_suffix}{data_seed_suffix}{epoch_str}'
    
    print(f"\nSaving predicted maps to: {predicted_maps_dir}")
    
    for subject_idx in range(len(predicted_values)):
        # Get predicted and measured values for this subject
        pred = predicted_values[subject_idx].cpu().numpy().flatten() if isinstance(predicted_values[subject_idx], torch.Tensor) else predicted_values[subject_idx].flatten()
        meas = measured_values[subject_idx].cpu().numpy().flatten() if isinstance(measured_values[subject_idx], torch.Tensor) else measured_values[subject_idx].flatten()
        R2 = R2_values[subject_idx].cpu().numpy().flatten() if isinstance(R2_values[subject_idx], torch.Tensor) else R2_values[subject_idx].flatten()
        
        # Get subject name if available
        subject_name = subject_names[subject_idx] if subject_names and subject_idx < len(subject_names) else f"subject_{subject_idx}"
        
        # Create filename
        filename = f'{base_filename}_{subject_name}.npz'
        filepath = osp.join(predicted_maps_dir, filename)
        
        # Save as compressed numpy array
        np.savez_compressed(
            filepath,
            predicted=pred,
            measured=meas,
            R2=R2,
            subject_name=subject_name,
            prediction_type=prediction_type,
            hemisphere=hemisphere,
            model_type=model_type
        )
    
    print(f"Saved {len(predicted_values)} predicted maps to {predicted_maps_dir}")

# =============================
# Compute Correlations Per Subject
# =============================
def compute_subject_correlations(predicted_values, measured_values, R2_values, prediction_type, R2_threshold=2.2, v1v2v3_mask=None):
    """
    Compute correlations for each subject in the test set.
    
    Args:
        predicted_values: List of tensors, one per subject (from test_loader batches)
        measured_values: List of tensors, one per subject (from test_loader batches)
        R2_values: List of tensors, one per subject with R2 values (from test_loader batches)
        prediction_type: 'eccentricity', 'polarAngle', or 'pRFsize'
        R2_threshold: R2 threshold for filtering vertices (default: 2.2)
        v1v2v3_mask: Optional boolean mask for V1-V3 areas (applied in addition to R2 threshold)
    
    Returns:
        Dictionary with correlation statistics
    """
    subject_correlations = []
    
    for subject_idx in range(len(predicted_values)):
        # Get predicted and measured values for this subject
        pred = predicted_values[subject_idx].cpu().numpy().flatten() if isinstance(predicted_values[subject_idx], torch.Tensor) else predicted_values[subject_idx].flatten()
        meas = measured_values[subject_idx].cpu().numpy().flatten() if isinstance(measured_values[subject_idx], torch.Tensor) else measured_values[subject_idx].flatten()
        R2 = R2_values[subject_idx].cpu().numpy().flatten() if isinstance(R2_values[subject_idx], torch.Tensor) else R2_values[subject_idx].flatten()
        
        # Filter by R2 threshold (same as other metrics)
        R2_mask = R2 > R2_threshold
        
        # Apply V1-V3 mask if provided
        if v1v2v3_mask is not None:
            # Ensure mask length matches or is longer (we'll use first len(pred) elements)
            if len(v1v2v3_mask) >= len(pred):
                area_mask = v1v2v3_mask[:len(pred)]
            else:
                print(f"Warning: V1-V3 mask length ({len(v1v2v3_mask)}) is shorter than data length ({len(pred)}). Skipping V1-V3 filter for this subject.")
                area_mask = np.ones(len(pred), dtype=bool)
        else:
            area_mask = np.ones(len(pred), dtype=bool)
        
        # Remove NaN and Inf values
        valid_mask = np.isfinite(pred) & np.isfinite(meas) & R2_mask & area_mask
        pred_valid = pred[valid_mask]
        meas_valid = meas[valid_mask]
        
        if len(pred_valid) == 0:
            print(f"Warning: No valid values for subject {subject_idx}")
            subject_correlations.append(np.nan)
            continue
        
        # Compute correlation based on prediction type
        if prediction_type == 'polarAngle':
            # Use circular correlation for polar angle
            if ASTROPY_AVAILABLE:
                try:
                    # Convert to radians for circular correlation
                    pred_rad = np.deg2rad(pred_valid)
                    meas_rad = np.deg2rad(meas_valid)
                    # circcorrcoef expects arrays with units
                    pred_rad_with_units = pred_rad * u.rad
                    meas_rad_with_units = meas_rad * u.rad
                    corr_coef = circcorrcoef(pred_rad_with_units, meas_rad_with_units)
                    subject_correlations.append(corr_coef)
                except Exception as e:
                    print(f"Error computing circular correlation for subject {subject_idx}: {e}")
                    subject_correlations.append(np.nan)
            else:
                # Fallback: Use Pearson correlation (not ideal for circular data)
                try:
                    corr_coef, _ = scipy.stats.pearsonr(pred_valid, meas_valid)
                    subject_correlations.append(corr_coef)
                except Exception as e:
                    print(f"Error computing Pearson correlation for subject {subject_idx}: {e}")
                    subject_correlations.append(np.nan)
        else:
            # For eccentricity and pRFsize, use Pearson correlation
            try:
                corr_coef, _ = scipy.stats.pearsonr(pred_valid, meas_valid)
                subject_correlations.append(corr_coef)
            except Exception as e:
                print(f"Error computing Pearson correlation for subject {subject_idx}: {e}")
                subject_correlations.append(np.nan)
    
    subject_correlations = np.array(subject_correlations)
    
    return {
        'subject_correlations': subject_correlations,
        'mean': np.nanmean(subject_correlations),
        'std': np.nanstd(subject_correlations),
        'median': np.nanmedian(subject_correlations),
        'min': np.nanmin(subject_correlations),
        'max': np.nanmax(subject_correlations)
    }

def test(model_to_evaluate, model_name="model", subject_names=None):
    """Evaluate model on test set
    
    Args:
        model_to_evaluate: Model to evaluate
        model_name: Name of the model (for logging)
        subject_names: List of subject names/IDs (optional)
    
    Returns:
        Dictionary with test results including subject names if provided
    """
    model_to_evaluate.eval()
    subject_MAE_list = []
    subject_MAE_thr_list = []
    subject_MAE_no_thr_list = []
    subject_MAE_thr_V1V2V3_list = []
    subject_MAE_ecc1to8_list = []
    subject_MAE_thr_ecc1to8_list = []
    y = []
    y_hat = []
    R2_plot = []
    
    # Load V1-V3 mask for the current hemisphere
    try:
        v1v2v3_mask = load_V1V2V3_mask(args.hemisphere)
        print(f"Loaded V1-V3 mask for {args.hemisphere} hemisphere: {np.sum(v1v2v3_mask)} vertices")
    except Exception as e:
        print(f"Warning: Could not load V1-V3 mask: {e}. V1-V3 metrics will not be computed.")
        v1v2v3_mask = None
    
    # Load eccentricity 1-8 mask (only for eccentricity prediction)
    ecc_1to8_mask = None
    if args.prediction == 'eccentricity':
        try:
            ecc_1to8_mask = load_eccentricity_1to8_mask(args.hemisphere)
            print(f"Loaded eccentricity 1-8 mask for {args.hemisphere} hemisphere: {np.sum(ecc_1to8_mask)} vertices")
        except Exception as e:
            print(f"Warning: Could not load eccentricity 1-8 mask: {e}. Eccentricity 1-8 metrics will not be computed.")
            ecc_1to8_mask = None
    
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for data in test_loader:
            batch = data.to(device)
            pred = model_to_evaluate(batch).detach().view(-1)

            y_true = batch.y.view(-1)
            y_hat.append(pred)
            y.append(y_true)

            R2 = batch.R2.view(-1).to(device)
            R2_plot.append(R2)

            threshold = R2 > 2.2
            threshold2 = R2 > 17

            # MAE without thresholding
            if args.prediction == 'polarAngle':
                # Use smallest_angle for polar angle (circular data)
                pred_np = pred.cpu().numpy().flatten()
                y_true_np = y_true.cpu().numpy().flatten()
                # Convert degrees to radians for smallest_angle
                pred_rad = np.deg2rad(pred_np)
                y_true_rad = np.deg2rad(y_true_np)
                angle_diff = smallest_angle(y_true_rad, pred_rad)  # Returns degrees
                MAE_no_thr = np.mean(angle_diff)
            else:
                MAE_no_thr = torch.mean(torch.abs(y_true - pred)).item()
            subject_MAE_no_thr_list.append(MAE_no_thr)

            # MAE with R2 > 2.2 threshold
            pred_thr = pred[threshold]
            y_thr = y_true[threshold]
            if args.prediction == 'polarAngle':
                if pred_thr.numel() > 0:
                    pred_thr_np = pred_thr.cpu().numpy().flatten()
                    y_thr_np = y_thr.cpu().numpy().flatten()
                    pred_thr_rad = np.deg2rad(pred_thr_np)
                    y_thr_rad = np.deg2rad(y_thr_np)
                    angle_diff_thr = smallest_angle(y_thr_rad, pred_thr_rad)
                    MAE = np.mean(angle_diff_thr)
                else:
                    MAE = float('nan')
            else:
                MAE = torch.mean(torch.abs(y_thr - pred_thr)).item() if pred_thr.numel() > 0 else float('nan')
            subject_MAE_list.append(MAE)

            # MAE with R2 > 17 threshold
            pred_thr2 = pred[threshold2]
            y_thr2 = y_true[threshold2]
            if args.prediction == 'polarAngle':
                if pred_thr2.numel() > 0:
                    pred_thr2_np = pred_thr2.cpu().numpy().flatten()
                    y_thr2_np = y_thr2.cpu().numpy().flatten()
                    pred_thr2_rad = np.deg2rad(pred_thr2_np)
                    y_thr2_rad = np.deg2rad(y_thr2_np)
                    angle_diff_thr2 = smallest_angle(y_thr2_rad, pred_thr2_rad)
                    MAE_thr = np.mean(angle_diff_thr2)
                else:
                    MAE_thr = float('nan')
            else:
                MAE_thr = torch.mean(torch.abs(y_thr2 - pred_thr2)).item() if pred_thr2.numel() > 0 else float('nan')
            subject_MAE_thr_list.append(MAE_thr)
            
            # Compute MAE_thr for V1-V3 only
            if v1v2v3_mask is not None and len(v1v2v3_mask) >= len(pred):
                # Convert mask to tensor and apply to current batch
                # Use only the first len(pred) elements of the mask
                v1v2v3_mask_batch = v1v2v3_mask[:len(pred)]
                v1v2v3_mask_tensor = torch.tensor(v1v2v3_mask_batch, device=device, dtype=torch.bool)
                # Apply V1-V3 mask and R2 > 17 threshold
                v1v2v3_threshold = threshold2 & v1v2v3_mask_tensor
                pred_v1v2v3 = pred[v1v2v3_threshold]
                y_v1v2v3 = y_true[v1v2v3_threshold]
                if args.prediction == 'polarAngle':
                    if pred_v1v2v3.numel() > 0:
                        pred_v1v2v3_np = pred_v1v2v3.cpu().numpy().flatten()
                        y_v1v2v3_np = y_v1v2v3.cpu().numpy().flatten()
                        pred_v1v2v3_rad = np.deg2rad(pred_v1v2v3_np)
                        y_v1v2v3_rad = np.deg2rad(y_v1v2v3_np)
                        angle_diff_v1v2v3 = smallest_angle(y_v1v2v3_rad, pred_v1v2v3_rad)
                        MAE_thr_V1V2V3 = np.mean(angle_diff_v1v2v3)
                    else:
                        MAE_thr_V1V2V3 = float('nan')
                else:
                    MAE_thr_V1V2V3 = torch.mean(torch.abs(y_v1v2v3 - pred_v1v2v3)).item() if pred_v1v2v3.numel() > 0 else float('nan')
                subject_MAE_thr_V1V2V3_list.append(MAE_thr_V1V2V3)
            else:
                subject_MAE_thr_V1V2V3_list.append(float('nan'))
            
            # Compute MAE for eccentricity 1-8 range only (only for eccentricity prediction)
            if args.prediction == 'eccentricity' and ecc_1to8_mask is not None and len(ecc_1to8_mask) >= len(pred):
                # Convert mask to tensor and apply to current batch
                # Use only the first len(pred) elements of the mask
                ecc_1to8_mask_batch = ecc_1to8_mask[:len(pred)]
                ecc_1to8_mask_tensor = torch.tensor(ecc_1to8_mask_batch, device=device, dtype=torch.bool)
                
                # MAE with R2 > 2.2 threshold + ecc 1-8 mask
                ecc_1to8_threshold = threshold & ecc_1to8_mask_tensor
                pred_ecc1to8 = pred[ecc_1to8_threshold]
                y_ecc1to8 = y_true[ecc_1to8_threshold]
                MAE_ecc1to8 = torch.mean(torch.abs(y_ecc1to8 - pred_ecc1to8)).item() if pred_ecc1to8.numel() > 0 else float('nan')
                subject_MAE_ecc1to8_list.append(MAE_ecc1to8)
                
                # MAE with R2 > 17 threshold + ecc 1-8 mask
                ecc_1to8_threshold2 = threshold2 & ecc_1to8_mask_tensor
                pred_ecc1to8_thr = pred[ecc_1to8_threshold2]
                y_ecc1to8_thr = y_true[ecc_1to8_threshold2]
                MAE_thr_ecc1to8 = torch.mean(torch.abs(y_ecc1to8_thr - pred_ecc1to8_thr)).item() if pred_ecc1to8_thr.numel() > 0 else float('nan')
                subject_MAE_thr_ecc1to8_list.append(MAE_thr_ecc1to8)
            else:
                subject_MAE_ecc1to8_list.append(float('nan'))
                subject_MAE_thr_ecc1to8_list.append(float('nan'))

    # 마지막에 저장할 때만 cpu로 변환
    y_hat_save = [x.detach().cpu() for x in y_hat]
    y_save = [x.detach().cpu() for x in y]
    R2_plot_save = [x.detach().cpu() for x in R2_plot]

    # Convert to numpy arrays and compute statistics
    subject_MAE_array = np.array(subject_MAE_list)
    subject_MAE_thr_array = np.array(subject_MAE_thr_list)
    subject_MAE_no_thr_array = np.array(subject_MAE_no_thr_list)
    subject_MAE_thr_V1V2V3_array = np.array(subject_MAE_thr_V1V2V3_list)
    subject_MAE_ecc1to8_array = np.array(subject_MAE_ecc1to8_list)
    subject_MAE_thr_ecc1to8_array = np.array(subject_MAE_thr_ecc1to8_list)
    
    test_MAE = np.nanmean(subject_MAE_array)
    test_MAE_std = np.nanstd(subject_MAE_array)
    test_MAE_thr = np.nanmean(subject_MAE_thr_array)
    test_MAE_thr_std = np.nanstd(subject_MAE_thr_array)
    test_MAE_no_thr = np.nanmean(subject_MAE_no_thr_array)
    test_MAE_no_thr_std = np.nanstd(subject_MAE_no_thr_array)
    
    # Compute V1-V3 statistics
    test_MAE_thr_V1V2V3 = np.nanmean(subject_MAE_thr_V1V2V3_array)
    test_MAE_thr_V1V2V3_std = np.nanstd(subject_MAE_thr_V1V2V3_array)
    
    # Compute eccentricity 1-8 statistics (only for eccentricity prediction)
    test_MAE_ecc1to8 = np.nanmean(subject_MAE_ecc1to8_array) if args.prediction == 'eccentricity' else np.nan
    test_MAE_ecc1to8_std = np.nanstd(subject_MAE_ecc1to8_array) if args.prediction == 'eccentricity' else np.nan
    test_MAE_thr_ecc1to8 = np.nanmean(subject_MAE_thr_ecc1to8_array) if args.prediction == 'eccentricity' else np.nan
    test_MAE_thr_ecc1to8_std = np.nanstd(subject_MAE_thr_ecc1to8_array) if args.prediction == 'eccentricity' else np.nan
    
    # Compute correlations per subject (filtering by R2 > 2.2, same as other metrics)
    correlation_results = compute_subject_correlations(y_hat_save, y_save, R2_plot_save, args.prediction, R2_threshold=2.2)
    
    # Compute correlations for V1-V3 only
    correlation_results_V1V2V3 = None
    if v1v2v3_mask is not None:
        correlation_results_V1V2V3 = compute_subject_correlations(y_hat_save, y_save, R2_plot_save, args.prediction, R2_threshold=2.2, v1v2v3_mask=v1v2v3_mask)
    
    output = {
        'Predicted_values': y_hat_save,
        'Measured_values': y_save,
        'R2': R2_plot_save,
        'MAE': test_MAE,
        'MAE_std': test_MAE_std,
        'MAE_thr': test_MAE_thr,
        'MAE_thr_std': test_MAE_thr_std,
        'MAE_no_thr': test_MAE_no_thr,
        'MAE_no_thr_std': test_MAE_no_thr_std,
        'Subject_MAE': subject_MAE_array,
        'Subject_MAE_thr': subject_MAE_thr_array,
        'Subject_MAE_no_thr': subject_MAE_no_thr_array,
        'Correlation_mean': correlation_results['mean'],
        'Correlation_std': correlation_results['std'],
        'Correlation_median': correlation_results['median'],
        'Correlation_min': correlation_results['min'],
        'Correlation_max': correlation_results['max'],
        'Subject_correlations': correlation_results['subject_correlations'],
        # V1-V3 only metrics
        'MAE_thr_V1V2V3': test_MAE_thr_V1V2V3,
        'MAE_thr_V1V2V3_std': test_MAE_thr_V1V2V3_std,
        'Subject_MAE_thr_V1V2V3': subject_MAE_thr_V1V2V3_array,
        # Eccentricity 1-8 only metrics (only for eccentricity prediction)
        'MAE_ecc1to8': test_MAE_ecc1to8,
        'MAE_ecc1to8_std': test_MAE_ecc1to8_std,
        'MAE_thr_ecc1to8': test_MAE_thr_ecc1to8,
        'MAE_thr_ecc1to8_std': test_MAE_thr_ecc1to8_std,
        'Subject_MAE_ecc1to8': subject_MAE_ecc1to8_array,
        'Subject_MAE_thr_ecc1to8': subject_MAE_thr_ecc1to8_array,
    }
    
    # Add V1-V3 correlation results if available
    if correlation_results_V1V2V3 is not None:
        output.update({
            'Correlation_mean_V1V2V3': correlation_results_V1V2V3['mean'],
            'Correlation_std_V1V2V3': correlation_results_V1V2V3['std'],
            'Correlation_median_V1V2V3': correlation_results_V1V2V3['median'],
            'Correlation_min_V1V2V3': correlation_results_V1V2V3['min'],
            'Correlation_max_V1V2V3': correlation_results_V1V2V3['max'],
            'Subject_correlations_V1V2V3': correlation_results_V1V2V3['subject_correlations']
        })
    else:
        output.update({
            'Correlation_mean_V1V2V3': np.nan,
            'Correlation_std_V1V2V3': np.nan,
            'Correlation_median_V1V2V3': np.nan,
            'Correlation_min_V1V2V3': np.nan,
            'Correlation_max_V1V2V3': np.nan,
            'Subject_correlations_V1V2V3': np.array([np.nan] * len(subject_MAE_array))
        })
    
    # Add subject names if provided
    if subject_names is not None:
        if len(subject_names) != len(subject_MAE_array):
            print(f"Warning: Number of subject names ({len(subject_names)}) does not match number of subjects ({len(subject_MAE_array)}). Subject names will not be included.")
        else:
            output['Subject_names'] = subject_names
    
    return output

# =============================
# Training Loop
# =============================
# Track best model based on dev/mae_thr
best_mae_thr = float('inf')
best_epoch = 0
best_model_state = None
# Early stopping tracking
epochs_without_improvement = 0
early_stopped = False
final_epoch = args.n_epochs  # Default to max epochs if no early stopping

# For test-only mode with local checkpoint, we don't know the epoch
if test_only_mode and args.checkpoint_path is not None:
    final_epoch = 0  # Will be set to "unknown" in output

if not test_only_mode:
    # Normal training loop
    for epoch in range(1, args.n_epochs + 1):
        loss, MAE = train(epoch)
        test_output = validate()
        current_lr = get_current_lr(optimizer)
        print(
            'Epoch: {:02d}, Train_loss: {:.4f}, Train_MAE: {:.4f}, Val_MAE: {'
            ':.4f}, Val_MAE_thr: {:.4f}, Val_MAE_no_thr: {:.4f}, LR: {:.6f}'.format(
                epoch, loss, MAE, test_output['MAE'], test_output['MAE_thr'], test_output['MAE_no_thr'], current_lr))
        
        # Wandb logging (moved inside loop to log every epoch)
        if wandb_run is not None:
            wandb.log({
                "epoch": epoch,
                "monitor/loss": loss.cpu().item(),
                "monitor/mae": MAE,
                "monitor/val_mae": test_output['MAE'],
                "monitor/val_mae_thr": test_output['MAE_thr'],
                "monitor/val_mae_no_thr": test_output['MAE_no_thr'],
                "monitor/best_mae_thr": best_mae_thr,
                "monitor/best_epoch": best_epoch,
                "monitor/lr": current_lr,
                "dev/mae": test_output['MAE'],
                "dev/mae_thr": test_output['MAE_thr'],
                "dev/mae_no_thr": test_output['MAE_no_thr']
            }, step=epoch)
        
        # Track best model based on dev/mae_thr
        if test_output['MAE_thr'] < best_mae_thr:
            best_mae_thr = test_output['MAE_thr']
            best_epoch = epoch
            # Deep copy model state to avoid reference issues
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print(f"New best model at epoch {epoch} with MAE_thr: {best_mae_thr:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Early stopping check
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"No improvement for {args.early_stopping_patience} epochs (best MAE_thr: {best_mae_thr:.4f} at epoch {best_epoch})")
            early_stopped = True
            final_epoch = epoch
            if wandb_run is not None:
                wandb.log({
                    "early_stopping/triggered": True,
                    "early_stopping/epoch": epoch,
                    "early_stopping/best_epoch": best_epoch,
                    "early_stopping/best_mae_thr": best_mae_thr
                }, step=epoch)
            
            break
    # Set final_epoch if training completed normally (no early stopping)
    if not early_stopped:
        final_epoch = args.n_epochs
    
    if wandb_run is not None:
        wandb.log({
            "training/completed": not early_stopped,
            "training/early_stopped": early_stopped,
            "training/final_epoch": final_epoch
        }, step=final_epoch)

    # Saving model's learned parameters (only for training mode)
    # Save best model (lowest dev/mae_thr)
    if best_model_state is not None:
        best_model_file = osp.join(
            output_path,
            f'{prediction_short}_{args.hemisphere}_{args.model_type}{myelination_suffix}{data_seed_suffix}_best_model_epoch{best_epoch}.pt'
        )
        torch.save(best_model_state, best_model_file)
        print(f"Best model saved to: {best_model_file} (epoch {best_epoch}, MAE_thr: {best_mae_thr:.4f})")
        
        if wandb_run is not None:
            wandb.log({
                "best_model/epoch": best_epoch,
                "best_model/mae_thr": best_mae_thr
            })
            # Optionally save model as wandb artifact
            artifact = wandb.Artifact(f"best_model_{wandb_run.id}", type="model")
            artifact.add_file(best_model_file)
            wandb_run.log_artifact(artifact)  

    # Save final model (most recent epoch)
    final_model_file = osp.join(
        output_path,
        f'{prediction_short}_{args.hemisphere}_{args.model_type}{myelination_suffix}{data_seed_suffix}_final_model.pt'
    )
    torch.save(model.state_dict(), final_model_file)
    print(f"Final model saved to: {final_model_file}")
    if early_stopped:
        print(f"Training stopped early at epoch {final_epoch} (best model was at epoch {best_epoch})")
    else:
        print(f"Training completed all {args.n_epochs} epochs")

    if wandb_run is not None:
        # Optionally save model as wandb artifact
        artifact = wandb.Artifact(f"final_model_{wandb_run.id}", type="model")
        artifact.add_file(final_model_file)
        wandb_run.log_artifact(artifact)
    
    # =============================
    # Test Set Evaluation (before closing log file)
    # =============================
    if args.run_test:
        print("\n" + "="*50)
        print("Starting test set evaluation...")
        print("="*50)
        
        if not test_only_mode:
            # Evaluate best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                best_test_output = test(model, "best", subject_names=test_subject_names)
                print(f"\nBest Model (epoch {best_epoch}) Test Results:")
                print(f"  Test MAE (R2>2.2): {best_test_output['MAE']:.4f} ± {best_test_output['MAE_std']:.4f}")
                print(f"  Test MAE_thr (R2>17): {best_test_output['MAE_thr']:.4f} ± {best_test_output['MAE_thr_std']:.4f}")
                print(f"  Test MAE_no_thr (no threshold): {best_test_output['MAE_no_thr']:.4f} ± {best_test_output['MAE_no_thr_std']:.4f}")
                print(f"  Correlation (mean ± std): {best_test_output['Correlation_mean']:.4f} ± {best_test_output['Correlation_std']:.4f}")
                print(f"  Correlation (median): {best_test_output['Correlation_median']:.4f}")
                print(f"  Correlation range: [{best_test_output['Correlation_min']:.4f}, {best_test_output['Correlation_max']:.4f}]")
                # V1-V3 only results
                if not np.isnan(best_test_output['MAE_thr_V1V2V3']):
                    print(f"\n  V1-V3 Only Results:")
                    print(f"  Test MAE_thr (V1-V3): {best_test_output['MAE_thr_V1V2V3']:.4f} ± {best_test_output['MAE_thr_V1V2V3_std']:.4f}")
                    print(f"  Correlation (V1-V3, mean ± std): {best_test_output['Correlation_mean_V1V2V3']:.4f} ± {best_test_output['Correlation_std_V1V2V3']:.4f}")
                    print(f"  Correlation (V1-V3, median): {best_test_output['Correlation_median_V1V2V3']:.4f}")
                    print(f"  Correlation (V1-V3) range: [{best_test_output['Correlation_min_V1V2V3']:.4f}, {best_test_output['Correlation_max_V1V2V3']:.4f}]")
                # Eccentricity 1-8 only results (only for eccentricity prediction)
                if args.prediction == 'eccentricity' and not np.isnan(best_test_output['MAE_ecc1to8']):
                    print(f"\n  Eccentricity 1-8 Only Results:")
                    print(f"  Test MAE_ecc1to8 (R2>2.2, ecc 1-8): {best_test_output['MAE_ecc1to8']:.4f} ± {best_test_output['MAE_ecc1to8_std']:.4f}")
                    print(f"  Test MAE_thr_ecc1to8 (R2>17, ecc 1-8): {best_test_output['MAE_thr_ecc1to8']:.4f} ± {best_test_output['MAE_thr_ecc1to8_std']:.4f}")
                print(f"  Individual subject MAE:")
                for i, mae in enumerate(best_test_output['Subject_MAE']):
                    subject_name = best_test_output.get('Subject_names', [None])[i] if 'Subject_names' in best_test_output else None
                    subject_label = f"Subject {subject_name}" if subject_name else f"Subject {i}"
                    print(f"    {subject_label}: MAE={mae:.4f}, MAE_thr={best_test_output['Subject_MAE_thr'][i]:.4f}, Corr={best_test_output['Subject_correlations'][i]:.4f}")
            
            # Save best model test results
            best_test_file = osp.join(
                output_path,
                f'{prediction_short}_{args.hemisphere}_{args.model_type}{myelination_suffix}_best_test_results.pt'
            )
            save_dict = {
                'Epoch': best_epoch,
                'Predicted_values': best_test_output['Predicted_values'],
                'Measured_values': best_test_output['Measured_values'],
                'R2': best_test_output['R2'],
                'Test_MAE': best_test_output['MAE'],
                'Test_MAE_std': best_test_output['MAE_std'],
                'Test_MAE_thr': best_test_output['MAE_thr'],
                'Test_MAE_thr_std': best_test_output['MAE_thr_std'],
                'Test_MAE_no_thr': best_test_output['MAE_no_thr'],
                'Test_MAE_no_thr_std': best_test_output['MAE_no_thr_std'],
                'Subject_MAE': best_test_output['Subject_MAE'],
                'Subject_MAE_thr': best_test_output['Subject_MAE_thr'],
                'Subject_MAE_no_thr': best_test_output['Subject_MAE_no_thr'],
                'Correlation_mean': best_test_output['Correlation_mean'],
                'Correlation_std': best_test_output['Correlation_std'],
                'Correlation_median': best_test_output['Correlation_median'],
                'Correlation_min': best_test_output['Correlation_min'],
                'Correlation_max': best_test_output['Correlation_max'],
                'Subject_correlations': best_test_output['Subject_correlations'],
                # V1-V3 only metrics
                'Test_MAE_thr_V1V2V3': best_test_output['MAE_thr_V1V2V3'],
                'Test_MAE_thr_V1V2V3_std': best_test_output['MAE_thr_V1V2V3_std'],
                'Subject_MAE_thr_V1V2V3': best_test_output['Subject_MAE_thr_V1V2V3'],
                'Correlation_mean_V1V2V3': best_test_output['Correlation_mean_V1V2V3'],
                'Correlation_std_V1V2V3': best_test_output['Correlation_std_V1V2V3'],
                'Correlation_median_V1V2V3': best_test_output['Correlation_median_V1V2V3'],
                'Correlation_min_V1V2V3': best_test_output['Correlation_min_V1V2V3'],
                'Correlation_max_V1V2V3': best_test_output['Correlation_max_V1V2V3'],
                'Subject_correlations_V1V2V3': best_test_output['Subject_correlations_V1V2V3'],
                # Eccentricity 1-8 only metrics (only for eccentricity prediction)
                'Test_MAE_ecc1to8': best_test_output['MAE_ecc1to8'],
                'Test_MAE_ecc1to8_std': best_test_output['MAE_ecc1to8_std'],
                'Test_MAE_thr_ecc1to8': best_test_output['MAE_thr_ecc1to8'],
                'Test_MAE_thr_ecc1to8_std': best_test_output['MAE_thr_ecc1to8_std'],
                'Subject_MAE_ecc1to8': best_test_output['Subject_MAE_ecc1to8'],
                'Subject_MAE_thr_ecc1to8': best_test_output['Subject_MAE_thr_ecc1to8']
            }
            # Add subject names if available
            if 'Subject_names' in best_test_output:
                save_dict['Subject_names'] = best_test_output['Subject_names']
            torch.save(save_dict, best_test_file)
            
            # Save predicted maps if requested
            if args.save_predicted_map:
                save_predicted_maps(
                    best_test_output['Predicted_values'],
                    best_test_output['Measured_values'],
                    best_test_output['R2'],
                    best_test_output.get('Subject_names', None),
                    output_path,
                    args.prediction,
                    args.hemisphere,
                    args.model_type,
                    myelination_suffix,
                    data_seed_suffix,
                    epoch=best_epoch
                )
            
            if wandb_run is not None:
                wandb.log({
                    "test/best_model/mae": best_test_output['MAE'],
                    "test/best_model/mae_std": best_test_output['MAE_std'],
                    "test/best_model/mae_thr": best_test_output['MAE_thr'],
                    "test/best_model/mae_thr_std": best_test_output['MAE_thr_std'],
                    "test/best_model/mae_no_thr": best_test_output['MAE_no_thr'],
                    "test/best_model/mae_no_thr_std": best_test_output['MAE_no_thr_std'],
                    "test/best_model/correlation_mean": best_test_output['Correlation_mean'],
                    "test/best_model/correlation_std": best_test_output['Correlation_std'],
                    "test/best_model/correlation_median": best_test_output['Correlation_median'],
                    "test/best_model/mae_thr_V1V2V3": best_test_output['MAE_thr_V1V2V3'],
                    "test/best_model/mae_thr_V1V2V3_std": best_test_output['MAE_thr_V1V2V3_std'],
                    "test/best_model/correlation_mean_V1V2V3": best_test_output['Correlation_mean_V1V2V3'],
                    "test/best_model/correlation_std_V1V2V3": best_test_output['Correlation_std_V1V2V3'],
                    "test/best_model/correlation_median_V1V2V3": best_test_output['Correlation_median_V1V2V3'],
                    "test/best_model/mae_ecc1to8": best_test_output['MAE_ecc1to8'],
                    "test/best_model/mae_ecc1to8_std": best_test_output['MAE_ecc1to8_std'],
                    "test/best_model/mae_thr_ecc1to8": best_test_output['MAE_thr_ecc1to8'],
                    "test/best_model/mae_thr_ecc1to8_std": best_test_output['MAE_thr_ecc1to8_std'],
                    "test/best_model/epoch": best_epoch
                })
                # Optionally save test results as wandb artifact
                artifact = wandb.Artifact(f"best_test_results_{wandb_run.id}", type="results")
                artifact.add_file(best_test_file)
                wandb_run.log_artifact(artifact)

        # Only load from file if not in test-only mode
        final_model_file = osp.join(
            output_path,
            f'{prediction_short}_{args.hemisphere}_{args.model_type}{myelination_suffix}{data_seed_suffix}_final_model.pt'
        )
        if osp.exists(final_model_file):
            model.load_state_dict(torch.load(final_model_file, map_location=device))
        else:
            print(f"Warning: Final model file not found: {final_model_file}")
            print("Using current model state for final model evaluation.")
        
        final_test_output = test(model, "final", subject_names=test_subject_names)
        epoch_str = f"epoch {final_epoch}" if final_epoch > 0 else "unknown epoch"
        print(f"\nFinal Model ({epoch_str}) Test Results:")
        print(f"  Test MAE (R2>2.2): {final_test_output['MAE']:.4f} ± {final_test_output['MAE_std']:.4f}")
        print(f"  Test MAE_thr (R2>17): {final_test_output['MAE_thr']:.4f} ± {final_test_output['MAE_thr_std']:.4f}")
        print(f"  Test MAE_no_thr (no threshold): {final_test_output['MAE_no_thr']:.4f} ± {final_test_output['MAE_no_thr_std']:.4f}")
        print(f"  Correlation (mean ± std): {final_test_output['Correlation_mean']:.4f} ± {final_test_output['Correlation_std']:.4f}")
        print(f"  Correlation (median): {final_test_output['Correlation_median']:.4f}")
        print(f"  Correlation range: [{final_test_output['Correlation_min']:.4f}, {final_test_output['Correlation_max']:.4f}]")
        # V1-V3 only results
        if not np.isnan(final_test_output['MAE_thr_V1V2V3']):
            print(f"\n  V1-V3 Only Results:")
            print(f"  Test MAE_thr (V1-V3): {final_test_output['MAE_thr_V1V2V3']:.4f} ± {final_test_output['MAE_thr_V1V2V3_std']:.4f}")
            print(f"  Correlation (V1-V3, mean ± std): {final_test_output['Correlation_mean_V1V2V3']:.4f} ± {final_test_output['Correlation_std_V1V2V3']:.4f}")
            print(f"  Correlation (V1-V3, median): {final_test_output['Correlation_median_V1V2V3']:.4f}")
            print(f"  Correlation (V1-V3) range: [{final_test_output['Correlation_min_V1V2V3']:.4f}, {final_test_output['Correlation_max_V1V2V3']:.4f}]")
        # Eccentricity 1-8 only results (only for eccentricity prediction)
        if args.prediction == 'eccentricity' and not np.isnan(final_test_output['MAE_ecc1to8']):
            print(f"\n  Eccentricity 1-8 Only Results:")
            print(f"  Test MAE_ecc1to8 (R2>2.2, ecc 1-8): {final_test_output['MAE_ecc1to8']:.4f} ± {final_test_output['MAE_ecc1to8_std']:.4f}")
            print(f"  Test MAE_thr_ecc1to8 (R2>17, ecc 1-8): {final_test_output['MAE_thr_ecc1to8']:.4f} ± {final_test_output['MAE_thr_ecc1to8_std']:.4f}")
        print(f"  Individual subject MAE:")
        for i, mae in enumerate(final_test_output['Subject_MAE']):
            subject_name = final_test_output.get('Subject_names', [None])[i] if 'Subject_names' in final_test_output else None
            subject_label = f"Subject {subject_name}" if subject_name else f"Subject {i}"
            print(f"    {subject_label}: MAE={mae:.4f}, MAE_thr={final_test_output['Subject_MAE_thr'][i]:.4f}, Corr={final_test_output['Subject_correlations'][i]:.4f}")

        # Save final model test results
        final_test_file = osp.join(
            output_path,
            f'{prediction_short}_{args.hemisphere}_{args.model_type}{myelination_suffix}{data_seed_suffix}_final_test_results.pt'
        )
        save_dict = {
            'Epoch': final_epoch,
            'Predicted_values': final_test_output['Predicted_values'],
            'Measured_values': final_test_output['Measured_values'],
            'R2': final_test_output['R2'],
            'Test_MAE': final_test_output['MAE'],
            'Test_MAE_std': final_test_output['MAE_std'],
            'Test_MAE_thr': final_test_output['MAE_thr'],
            'Test_MAE_thr_std': final_test_output['MAE_thr_std'],
            'Test_MAE_no_thr': final_test_output['MAE_no_thr'],
            'Test_MAE_no_thr_std': final_test_output['MAE_no_thr_std'],
            'Subject_MAE': final_test_output['Subject_MAE'],
            'Subject_MAE_thr': final_test_output['Subject_MAE_thr'],
            'Subject_MAE_no_thr': final_test_output['Subject_MAE_no_thr'],
            'Correlation_mean': final_test_output['Correlation_mean'],
            'Correlation_std': final_test_output['Correlation_std'],
            'Correlation_median': final_test_output['Correlation_median'],
            'Correlation_min': final_test_output['Correlation_min'],
            'Correlation_max': final_test_output['Correlation_max'],
            'Subject_correlations': final_test_output['Subject_correlations'],
            # V1-V3 only metrics
            'Test_MAE_thr_V1V2V3': final_test_output['MAE_thr_V1V2V3'],
            'Test_MAE_thr_V1V2V3_std': final_test_output['MAE_thr_V1V2V3_std'],
            'Subject_MAE_thr_V1V2V3': final_test_output['Subject_MAE_thr_V1V2V3'],
            'Correlation_mean_V1V2V3': final_test_output['Correlation_mean_V1V2V3'],
            'Correlation_std_V1V2V3': final_test_output['Correlation_std_V1V2V3'],
            'Correlation_median_V1V2V3': final_test_output['Correlation_median_V1V2V3'],
            'Correlation_min_V1V2V3': final_test_output['Correlation_min_V1V2V3'],
            'Correlation_max_V1V2V3': final_test_output['Correlation_max_V1V2V3'],
            'Subject_correlations_V1V2V3': final_test_output['Subject_correlations_V1V2V3'],
            # Eccentricity 1-8 only metrics (only for eccentricity prediction)
            'Test_MAE_ecc1to8': final_test_output['MAE_ecc1to8'],
            'Test_MAE_ecc1to8_std': final_test_output['MAE_ecc1to8_std'],
            'Test_MAE_thr_ecc1to8': final_test_output['MAE_thr_ecc1to8'],
            'Test_MAE_thr_ecc1to8_std': final_test_output['MAE_thr_ecc1to8_std'],
            'Subject_MAE_ecc1to8': final_test_output['Subject_MAE_ecc1to8'],
            'Subject_MAE_thr_ecc1to8': final_test_output['Subject_MAE_thr_ecc1to8']
        }
        # Add subject names if available
        if 'Subject_names' in final_test_output:
            save_dict['Subject_names'] = final_test_output['Subject_names']
        torch.save(save_dict, final_test_file)
        
        # Save predicted maps if requested
        if args.save_predicted_map:
            save_predicted_maps(
                final_test_output['Predicted_values'],
                final_test_output['Measured_values'],
                final_test_output['R2'],
                final_test_output.get('Subject_names', None),
                output_path,
                args.prediction,
                args.hemisphere,
                args.model_type,
                myelination_suffix,
                data_seed_suffix,
                epoch=final_epoch
            )
        
        if wandb_run is not None:
            wandb.log({
                "test/final_model/mae": final_test_output['MAE'],
                "test/final_model/mae_std": final_test_output['MAE_std'],
                "test/final_model/mae_thr": final_test_output['MAE_thr'],
                "test/final_model/mae_thr_std": final_test_output['MAE_thr_std'],
                "test/final_model/mae_no_thr": final_test_output['MAE_no_thr'],
                "test/final_model/mae_no_thr_std": final_test_output['MAE_no_thr_std'],
                "test/final_model/correlation_mean": final_test_output['Correlation_mean'],
                "test/final_model/correlation_std": final_test_output['Correlation_std'],
                "test/final_model/correlation_median": final_test_output['Correlation_median'],
                "test/final_model/mae_thr_V1V2V3": final_test_output['MAE_thr_V1V2V3'],
                "test/final_model/mae_thr_V1V2V3_std": final_test_output['MAE_thr_V1V2V3_std'],
                "test/final_model/correlation_mean_V1V2V3": final_test_output['Correlation_mean_V1V2V3'],
                "test/final_model/correlation_std_V1V2V3": final_test_output['Correlation_std_V1V2V3'],
                "test/final_model/correlation_median_V1V2V3": final_test_output['Correlation_median_V1V2V3'],
                "test/final_model/mae_ecc1to8": final_test_output['MAE_ecc1to8'],
                "test/final_model/mae_ecc1to8_std": final_test_output['MAE_ecc1to8_std'],
                "test/final_model/mae_thr_ecc1to8": final_test_output['MAE_thr_ecc1to8'],
                "test/final_model/mae_thr_ecc1to8_std": final_test_output['MAE_thr_ecc1to8_std'],
                "test/final_model/epoch": final_epoch
            })
            # Optionally save test results as wandb artifact
            artifact = wandb.Artifact(f"final_test_results_{wandb_run.id}", type="results")
            artifact.add_file(final_test_file)
            wandb_run.log_artifact(artifact)
        
        print("\nTest set evaluation completed.")

        # Close Wandb run
        if wandb_run is not None:
            wandb.finish()
        
        print(f"\nTraining completed. Output directory: {output_path}")
    
    # Restore original stdout and close log file after training and test evaluation
    if not test_only_mode:
        sys.stdout = original_stdout
        log_tee.close()
        print(f"Training log saved to: {log_file_path}")
else:
    print("\n" + "="*50)
    print("Skipping training (test-only mode)")
    print("="*50)
    

# =============================
# Test Set Evaluation (for test-only mode)
# =============================
if args.run_test and test_only_mode:
    # Load test subject names if available
    if test_subject_names is None and test_dataset is not None:
        subject_splits_dir = osp.join(path, 'subject_splits', f'seed{data_split_seed}')
        test_subjects_file = osp.join(subject_splits_dir, 'test_subjects.txt')
        if osp.exists(test_subjects_file):
            with open(test_subjects_file, 'r') as f:
                test_subject_names = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(test_subject_names)} test subject names from {test_subjects_file}")
    
    # final model is already loaded in test-only mode
    final_test_output = test(model, "best", subject_names=test_subject_names)
    epoch_str = f"epoch {final_epoch}" if final_epoch > 0 else "unknown epoch"
    print(f"\nFinal Model ({epoch_str}) Test Results:")
    print(f"  Test MAE (R2>2.2): {final_test_output['MAE']:.4f} ± {final_test_output['MAE_std']:.4f}")
    print(f"  Test MAE_thr (R2>17): {final_test_output['MAE_thr']:.4f} ± {final_test_output['MAE_thr_std']:.4f}")
    print(f"  Test MAE_no_thr (no threshold): {final_test_output['MAE_no_thr']:.4f} ± {final_test_output['MAE_no_thr_std']:.4f}")
    print(f"  Correlation (mean ± std): {final_test_output['Correlation_mean']:.4f} ± {final_test_output['Correlation_std']:.4f}")
    print(f"  Correlation (median): {final_test_output['Correlation_median']:.4f}")
    print(f"  Correlation range: [{final_test_output['Correlation_min']:.4f}, {final_test_output['Correlation_max']:.4f}]")
    # V1-V3 only results
    if not np.isnan(final_test_output['MAE_thr_V1V2V3']):
        print(f"\n  V1-V3 Only Results:")
        print(f"  Test MAE_thr (V1-V3): {final_test_output['MAE_thr_V1V2V3']:.4f} ± {final_test_output['MAE_thr_V1V2V3_std']:.4f}")
        print(f"  Correlation (V1-V3, mean ± std): {final_test_output['Correlation_mean_V1V2V3']:.4f} ± {final_test_output['Correlation_std_V1V2V3']:.4f}")
        print(f"  Correlation (V1-V3, median): {final_test_output['Correlation_median_V1V2V3']:.4f}")
        print(f"  Correlation (V1-V3) range: [{final_test_output['Correlation_min_V1V2V3']:.4f}, {final_test_output['Correlation_max_V1V2V3']:.4f}]")
    # Eccentricity 1-8 only results (only for eccentricity prediction)
    if args.prediction == 'eccentricity' and not np.isnan(final_test_output['MAE_ecc1to8']):
        print(f"\n  Eccentricity 1-8 Only Results:")
        print(f"  Test MAE_ecc1to8 (R2>2.2, ecc 1-8): {final_test_output['MAE_ecc1to8']:.4f} ± {final_test_output['MAE_ecc1to8_std']:.4f}")
        print(f"  Test MAE_thr_ecc1to8 (R2>17, ecc 1-8): {final_test_output['MAE_thr_ecc1to8']:.4f} ± {final_test_output['MAE_thr_ecc1to8_std']:.4f}")
    print(f"  Individual subject MAE:")
    for i, mae in enumerate(final_test_output['Subject_MAE']):
        subject_name = final_test_output.get('Subject_names', [None])[i] if 'Subject_names' in final_test_output else None
        subject_label = f"Subject {subject_name}" if subject_name else f"Subject {i}"
        print(f"    {subject_label}: MAE={mae:.4f}, MAE_thr={final_test_output['Subject_MAE_thr'][i]:.4f}, Corr={final_test_output['Subject_correlations'][i]:.4f}")
    
    # Save final model test results
    final_test_file = osp.join(
        output_path,
        f'{prediction_short}_{args.hemisphere}_{args.model_type}{myelination_suffix}{data_seed_suffix}_best_test_results.pt'
    )
    save_dict = {
        'Epoch': final_epoch,
        'Predicted_values': final_test_output['Predicted_values'],
        'Measured_values': final_test_output['Measured_values'],
        'R2': final_test_output['R2'],
        'Test_MAE': final_test_output['MAE'],
        'Test_MAE_std': final_test_output['MAE_std'],
        'Test_MAE_thr': final_test_output['MAE_thr'],
        'Test_MAE_thr_std': final_test_output['MAE_thr_std'],
        'Test_MAE_no_thr': final_test_output['MAE_no_thr'],
        'Test_MAE_no_thr_std': final_test_output['MAE_no_thr_std'],
        'Subject_MAE': final_test_output['Subject_MAE'],
        'Subject_MAE_thr': final_test_output['Subject_MAE_thr'],
        'Subject_MAE_no_thr': final_test_output['Subject_MAE_no_thr'],
        'Correlation_mean': final_test_output['Correlation_mean'],
        'Correlation_std': final_test_output['Correlation_std'],
        'Correlation_median': final_test_output['Correlation_median'],
        'Correlation_min': final_test_output['Correlation_min'],
        'Correlation_max': final_test_output['Correlation_max'],
        'Subject_correlations': final_test_output['Subject_correlations'],
        # V1-V3 only metrics
        'Test_MAE_thr_V1V2V3': final_test_output['MAE_thr_V1V2V3'],
        'Test_MAE_thr_V1V2V3_std': final_test_output['MAE_thr_V1V2V3_std'],
        'Subject_MAE_thr_V1V2V3': final_test_output['Subject_MAE_thr_V1V2V3'],
        'Correlation_mean_V1V2V3': final_test_output['Correlation_mean_V1V2V3'],
        'Correlation_std_V1V2V3': final_test_output['Correlation_std_V1V2V3'],
        'Correlation_median_V1V2V3': final_test_output['Correlation_median_V1V2V3'],
        'Correlation_min_V1V2V3': final_test_output['Correlation_min_V1V2V3'],
        'Correlation_max_V1V2V3': final_test_output['Correlation_max_V1V2V3'],
        'Subject_correlations_V1V2V3': final_test_output['Subject_correlations_V1V2V3'],
        # Eccentricity 1-8 only metrics (only for eccentricity prediction)
        'Test_MAE_ecc1to8': final_test_output['MAE_ecc1to8'],
        'Test_MAE_ecc1to8_std': final_test_output['MAE_ecc1to8_std'],
        'Test_MAE_thr_ecc1to8': final_test_output['MAE_thr_ecc1to8'],
        'Test_MAE_thr_ecc1to8_std': final_test_output['MAE_thr_ecc1to8_std'],
        'Subject_MAE_ecc1to8': final_test_output['Subject_MAE_ecc1to8'],
        'Subject_MAE_thr_ecc1to8': final_test_output['Subject_MAE_thr_ecc1to8']
    }
    # Add subject names if available
    if 'Subject_names' in final_test_output:
        save_dict['Subject_names'] = final_test_output['Subject_names']
    torch.save(save_dict, final_test_file)
    
    # Save predicted maps if requested
    if args.save_predicted_map:
        save_predicted_maps(
            final_test_output['Predicted_values'],
            final_test_output['Measured_values'],
            final_test_output['R2'],
            final_test_output.get('Subject_names', None),
            output_path,
            args.prediction,
            args.hemisphere,
            args.model_type,
            myelination_suffix,
            data_seed_suffix,
            epoch=final_epoch
        )
    
    if wandb_run is not None:
        wandb.log({
            "test/final_model/mae": final_test_output['MAE'],
            "test/final_model/mae_std": final_test_output['MAE_std'],
            "test/final_model/mae_thr": final_test_output['MAE_thr'],
            "test/final_model/mae_thr_std": final_test_output['MAE_thr_std'],
            "test/final_model/mae_no_thr": final_test_output['MAE_no_thr'],
            "test/final_model/mae_no_thr_std": final_test_output['MAE_no_thr_std'],
            "test/final_model/correlation_mean": final_test_output['Correlation_mean'],
            "test/final_model/correlation_std": final_test_output['Correlation_std'],
            "test/final_model/correlation_median": final_test_output['Correlation_median'],
            "test/final_model/mae_thr_V1V2V3": final_test_output['MAE_thr_V1V2V3'],
            "test/final_model/mae_thr_V1V2V3_std": final_test_output['MAE_thr_V1V2V3_std'],
            "test/final_model/correlation_mean_V1V2V3": final_test_output['Correlation_mean_V1V2V3'],
            "test/final_model/correlation_std_V1V2V3": final_test_output['Correlation_std_V1V2V3'],
            "test/final_model/correlation_median_V1V2V3": final_test_output['Correlation_median_V1V2V3'],
            "test/final_model/epoch": final_epoch
        })
        # Optionally save test results as wandb artifact
        artifact = wandb.Artifact(f"final_test_results_{wandb_run.id}", type="results")
        artifact.add_file(final_test_file)
        wandb_run.log_artifact(artifact)
    
    print(f"\nTest-only evaluation completed. Output directory: {output_path}")
