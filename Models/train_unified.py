import os
import os.path as osp
import sys
import time
import argparse
import copy
import torch
import torch_geometric.transforms as T
import numpy as np

# Neptune import (optional)
try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False
    print("Warning: Neptune not available. Logging will be skipped.")

# =============================
# Argument Parser Setup
# =============================
parser = argparse.ArgumentParser(
    description="Unified training script for deepRetinotopy models with Transolver variants."
)
parser.add_argument('--model_type', type=str, default='baseline',
                    choices=['baseline', 'transolver_optionA', 'transolver_optionB', 'transolver_optionC'],
                    help='Model architecture type')
parser.add_argument('--prediction', type=str, default='eccentricity',
                    choices=['eccentricity', 'polarAngle'],
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
parser.add_argument('--myelination', type=bool, default=True,
                    help='Use myelination as feature (default: True)')
parser.add_argument('--use_neptune', action='store_true',
                    help='Enable Neptune logging')
parser.add_argument('--project', type=str, default='kjb961013/Retinosolver',
                    help='Neptune project name (e.g., user/project)')
parser.add_argument('--api_token', type=str, default='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZWNlMzk4Ny1hOTVlLTRjMzgtOWI4ZS1hY2FkYTY4MzNhYzMifQ==',
                    help='Neptune API token or use NEPTUNE_API_TOKEN env variable')
parser.add_argument('--run_test', type=str, default='True',
                    choices=['True', 'False', 'true', 'false', '1', '0'],
                    help='Run test set evaluation after training (default: True)')
parser.add_argument('--early_stopping_patience', type=int, default=30,
                    help='Number of epochs to wait before early stopping if no improvement (default: 30)')
parser.add_argument('--neptune_run_id', type=str, default=None,
                    help='Neptune run ID to load checkpoint from (if provided, only test will be run)')
parser.add_argument('--checkpoint_type', type=str, default='best',
                    choices=['best', 'final'],
                    help='Type of checkpoint to load from Neptune: best or final (default: best)')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to local checkpoint file (.pt) to load (if provided, only test will be run)')

args = parser.parse_args()

# Convert run_test string to boolean
args.run_test = args.run_test.lower() in ['true', '1']

# Test-only mode if Neptune run ID or local checkpoint path is provided
if args.neptune_run_id is not None and args.checkpoint_path is not None:
    raise ValueError("Cannot specify both --neptune_run_id and --checkpoint_path. Please choose one.")
test_only_mode = args.neptune_run_id is not None or args.checkpoint_path is not None

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

from Retinotopy.dataset.HCP_3sets_ROI import Retinotopy
from torch_geometric.loader import DataLoader

# Import models from separate modules
from models import (
    deepRetinotopy_Baseline,
    deepRetinotopy_OptionA,
    deepRetinotopy_OptionB,
    deepRetinotopy_OptionC
)

# Create Neptune run (optional)
run = None
neptune_run_for_test = None  # For loading checkpoint from Neptune

if test_only_mode:
    # Test-only mode: Load checkpoint from Neptune (only if neptune_run_id is provided)
    if args.neptune_run_id is not None:
        # Load checkpoint from Neptune
        if not NEPTUNE_AVAILABLE:
            raise RuntimeError("Neptune is required for test-only mode with Neptune but is not available.")
        
        try:
            api_token = args.api_token or os.environ.get('NEPTUNE_API_TOKEN')
            if not api_token:
                raise ValueError("Neptune API token is required for test-only mode with Neptune.")
            
            # Initialize Neptune run in read mode
            neptune_run_for_test = neptune.init_run(
                project=args.project,
                api_token=api_token,
                with_id=args.neptune_run_id,
                mode="read-only"
            )
            print(f"Loaded Neptune run: {args.neptune_run_id}")
            
            # Fetch config from Neptune to get model parameters
            try:
                neptune_config = neptune_run_for_test["config"].fetch()
                print("Loaded config from Neptune run:")
                for key, value in neptune_config.items():
                    print(f"  {key}: {value}")
                
                # Update args with config from Neptune (for model creation)
                for key, value in neptune_config.items():
                    if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                        setattr(args, key, value)
                
                # Ensure required args are set
                if 'model_type' in neptune_config:
                    args.model_type = neptune_config['model_type']
                if 'prediction' in neptune_config:
                    args.prediction = neptune_config['prediction']
                if 'hemisphere' in neptune_config:
                    args.hemisphere = neptune_config['hemisphere']
                    
            except Exception as e:
                print(f"Warning: Could not fetch config from Neptune: {e}")
                print("Using provided command-line arguments for model creation.")
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Neptune run for test-only mode: {e}")
    else:
        # Local checkpoint mode - no Neptune needed
        print("Test-only mode: Using local checkpoint (Neptune not required)")
        
elif args.use_neptune and NEPTUNE_AVAILABLE:
    # Normal training mode with Neptune logging
    try:
        api_token = args.api_token or os.environ.get('NEPTUNE_API_TOKEN')
        if api_token:
            run = neptune.init_run(
                project=args.project,
                api_token=api_token
            )
        else:
            print("Warning: Neptune API token not provided. Skipping Neptune logging.")
    except Exception as e:
        print(f"Warning: Failed to initialize Neptune: {e}. Continuing without logging.")

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Retinotopy', 'data')

pre_transform = T.Compose([T.FaceToEdge()])
train_dataset = Retinotopy(
    path, 'Train',
    transform=T.Cartesian(),
    pre_transform=pre_transform,
    n_examples=args.n_examples,
    prediction=args.prediction,
    myelination=args.myelination,
    hemisphere=args.hemisphere
)
dev_dataset = Retinotopy(
    path, 'Development',
    transform=T.Cartesian(),
    pre_transform=pre_transform,
    n_examples=args.n_examples,
    prediction=args.prediction,
    myelination=args.myelination,
    hemisphere=args.hemisphere
)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

# Test dataset (only if run_test is True)
test_dataset = None
test_loader = None
if args.run_test:
    test_dataset = Retinotopy(
        path, 'Test',
        transform=T.Cartesian(),
        pre_transform=pre_transform,
        n_examples=args.n_examples,
        prediction=args.prediction,
        myelination=args.myelination,
        hemisphere=args.hemisphere
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Neptune config logging
neptune_run_id = None
if run is not None:
    run["config"] = vars(args)
    run["config/architecture"] = args.model_type
    run["config/optimizer"] = args.optimizer
    run["config/scheduler"] = args.scheduler
    run["config/loss_fn"] = "SmoothL1Loss"
    run["config/dataset"] = "HCP_3sets_ROI"
    run["config/train_size"] = len(train_dataset)
    run["config/dev_size"] = len(dev_dataset)
    if args.run_test:
        run["config/test_size"] = len(test_dataset)
    # Get Neptune run ID for folder naming
    try:
        neptune_run_id = run["sys/id"].fetch()
    except Exception as e:
        print(f"Warning: Could not fetch Neptune run ID: {e}. Using default folder name.")
        neptune_run_id = None

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
model = create_model(args.model_type, num_features=2, args=args).to(device)

# Calculate and log model size
total_params, trainable_params = count_parameters(model)
model_size_mb = calculate_model_size_mb(model)
print(f"Model size:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size: {model_size_mb:.2f} MB")

# Log model size to Neptune
if run is not None:
    run["model/total_parameters"] = total_params
    run["model/trainable_parameters"] = trainable_params
    run["model/size_mb"] = model_size_mb

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

# Create output directory structure: {output_dir}/{neptune_run_id}/ or {output_dir}/{model_type}_{prediction}_{hemisphere}/
prediction_short = 'ecc' if args.prediction == 'eccentricity' else 'PA'
if neptune_run_id is not None:
    output_subdir = neptune_run_id
else:
    output_subdir = f"{args.model_type}_{prediction_short}_{args.hemisphere}"
output_path = osp.join(osp.dirname(osp.realpath(__file__)), args.output_dir, output_subdir)
if not osp.exists(output_path):
    os.makedirs(output_path)

# Load checkpoint if in test-only mode
if test_only_mode:
    print("\n" + "="*50)
    if args.neptune_run_id is not None:
        print("Test-only mode: Loading checkpoint from Neptune")
        print("="*50)
        
        try:
            # Download checkpoint from Neptune
            artifact_name = f"artifacts/{args.checkpoint_type}_model_state_dict"
            checkpoint_path = osp.join(output_path, f"{args.checkpoint_type}_model_from_neptune.pt")
            
            print(f"Downloading {args.checkpoint_type} model checkpoint from Neptune...")
            neptune_run_for_test[artifact_name].download(checkpoint_path)
            print(f"Checkpoint downloaded to: {checkpoint_path}")
            
            # Load checkpoint into model
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"Model loaded from {args.checkpoint_type} checkpoint.")
            
            # Get epoch info if available
            try:
                if args.checkpoint_type == 'best':
                    best_epoch = neptune_run_for_test["best_model/epoch"].fetch()
                    best_mae_thr = neptune_run_for_test["best_model/mae_thr"].fetch()
                    print(f"Best model info: epoch {best_epoch}, MAE_thr: {best_mae_thr:.4f}")
                else:
                    final_epoch = neptune_run_for_test["training/final_epoch"].fetch()
                    print(f"Final model epoch: {final_epoch}")
            except Exception as e:
                print(f"Warning: Could not fetch epoch info: {e}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from Neptune: {e}")
    
    elif args.checkpoint_path is not None:
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
        if run is not None:
            run["lr_current"] = args.lr_decay

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        R2 = data.R2.view(-1)
        threshold = R2.view(-1) > 2.2

        loss = torch.nn.SmoothL1Loss()
        output_loss = loss(R2 * model(data), R2 * data.y.view(-1))
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
            if run is not None:
                run["lr_current"] = scheduler.get_last_lr()[0]
    
    # CosineAnnealingLR scheduler steps per epoch
    if args.scheduler == 'cosine':
        scheduler.step()
        if run is not None:
            run["lr_current"] = scheduler.get_last_lr()[0]
    
    # Neptune logging
    if run is not None:
        run["train/loss"].append(output_loss.detach().cpu().item())
        run["train/mae"].append(MAE)
    
    return output_loss.detach(), MAE


def test():
    model.eval()
    MeanAbsError = 0
    MeanAbsError_thr = 0
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

        MAE = torch.mean(abs(data.to(device).y.view(-1)[threshold == 1] - pred[
            threshold == 1])).item()
        MAE_thr = torch.mean(abs(
            data.to(device).y.view(-1)[threshold2 == 1] - pred[
                threshold2 == 1])).item()
        MeanAbsError_thr += MAE_thr
        MeanAbsError += MAE

    test_MAE = MeanAbsError / len(dev_loader)
    test_MAE_thr = MeanAbsError_thr / len(dev_loader)
    output = {'Predicted_values': y_hat, 'Measured_values': y, 'R2': R2_plot,
              'MAE': test_MAE, 'MAE_thr': test_MAE_thr}
    
    # Neptune logging
    if run is not None:
        run["dev/mae"].append(test_MAE)
        run["dev/mae_thr"].append(test_MAE_thr)
    
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

# Skip training if in test-only mode
if test_only_mode:
    print("\n" + "="*50)
    print("Skipping training (test-only mode)")
    print("="*50)
    # Evaluate on dev set first
    test_output = test()
    print(f"Dev set evaluation:")
    print(f"  Dev MAE: {test_output['MAE']:.4f}")
    print(f"  Dev MAE_thr: {test_output['MAE_thr']:.4f}")
else:
    # Normal training loop
    for epoch in range(1, args.n_epochs + 1):
        loss, MAE = train(epoch)
        test_output = test()
        current_lr = get_current_lr(optimizer)
        print(
            'Epoch: {:02d}, Train_loss: {:.4f}, Train_MAE: {:.4f}, Test_MAE: {'
            ':.4f}, Test_MAE_thr: {:.4f}, LR: {:.6f}'.format(
                epoch, loss, MAE, test_output['MAE'], test_output['MAE_thr'], current_lr))
        
        # Neptune logging (moved inside loop to log every epoch)
        if run is not None:
            run["epoch"] = epoch
            run["monitor/loss"].append(loss.cpu().item())
            run["monitor/mae"].append(MAE)
            run["monitor/val_mae"].append(test_output['MAE'])
            run["monitor/val_mae_thr"].append(test_output['MAE_thr'])
            run["monitor/best_mae_thr"] = best_mae_thr
            run["monitor/best_epoch"] = best_epoch
            # Log learning rate for each epoch
            run["monitor/lr"].append(current_lr)
        
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
            if run is not None:
                run["early_stopping/triggered"] = True
                run["early_stopping/epoch"] = epoch
                run["early_stopping/best_epoch"] = best_epoch
                run["early_stopping/best_mae_thr"] = best_mae_thr
            break

        if epoch % args.interm_save_every == 0:
            output_file = osp.join(
                output_path,
                f'{args.model_type}_{prediction_short}_{args.hemisphere}_output_epoch{epoch}.pt'
            )
            torch.save({'Epoch': epoch,
                        'Predicted_values': test_output['Predicted_values'],
                        'Measured_values': test_output['Measured_values'],
                        'R2': test_output['R2'], 'Loss': loss,
                        'Dev_MAE': test_output['MAE']},
                       output_file)
            
            # Neptune logging
            if run is not None:
                run[f"intermediate/epoch_{epoch}/dev_mae"] = test_output["MAE"]
                run[f"intermediate/epoch_{epoch}/dev_mae_thr"] = test_output["MAE_thr"]

    # Set final_epoch if training completed normally (no early stopping)
    if not early_stopped:
        final_epoch = args.n_epochs

# Log training completion status (only for training mode)
if not test_only_mode:
    if run is not None:
        if early_stopped:
            run["training/completed"] = False
            run["training/early_stopped"] = True
            run["training/final_epoch"] = final_epoch
        else:
            run["training/completed"] = True
            run["training/early_stopped"] = False
            run["training/final_epoch"] = final_epoch

    # Saving model's learned parameters (only for training mode)
    # Save best model (lowest dev/mae_thr)
    if best_model_state is not None:
        best_model_file = osp.join(
            output_path,
            f'{args.model_type}_{prediction_short}_{args.hemisphere}_best_model_epoch{best_epoch}.pt'
        )
        torch.save(best_model_state, best_model_file)
        print(f"Best model saved to: {best_model_file} (epoch {best_epoch}, MAE_thr: {best_mae_thr:.4f})")
        
        if run is not None:
            run["artifacts/best_model_state_dict"].upload(best_model_file)
            run["best_model/epoch"] = best_epoch
            run["best_model/mae_thr"] = best_mae_thr

    # Save final model (most recent epoch)
    final_model_file = osp.join(
        output_path,
        f'{args.model_type}_{prediction_short}_{args.hemisphere}_final_model.pt'
    )
    torch.save(model.state_dict(), final_model_file)
    print(f"Final model saved to: {final_model_file}")
    if early_stopped:
        print(f"Training stopped early at epoch {final_epoch} (best model was at epoch {best_epoch})")
    else:
        print(f"Training completed all {args.n_epochs} epochs")

    if run is not None:
        run["artifacts/final_model_state_dict"].upload(final_model_file)

# =============================
# Save Test Results to NPZ
# =============================
def save_test_results_npz(test_output, model_name, output_path, args, prediction_short):
    """
    Save test set results to npz file
    
    Args:
        test_output: Dictionary containing 'Predicted_values', 'Measured_values', 'R2', 'MAE', 'MAE_thr'
        model_name: Name of the model ('best' or 'final')
        output_path: Directory to save npz file
        args: Arguments object
        prediction_short: Short name for prediction type ('ecc' or 'PA')
    """
    # Concatenate all batches into single arrays
    y_hat_all = torch.cat(test_output['Predicted_values'], dim=0).cpu().numpy()
    y_all = torch.cat(test_output['Measured_values'], dim=0).cpu().numpy()
    R2_all = torch.cat(test_output['R2'], dim=0).cpu().numpy()
    
    # Calculate errors
    errors = np.abs(y_all - y_hat_all)
    
    # Apply threshold for R2 > 2.2
    threshold = R2_all > 2.2
    y_hat_thr = y_hat_all[threshold]
    y_thr = y_all[threshold]
    R2_thr = R2_all[threshold]
    errors_thr = errors[threshold]
    
    # Apply threshold for R2 > 17
    threshold2 = R2_all > 17
    y_hat_thr2 = y_hat_all[threshold2]
    y_thr2 = y_all[threshold2]
    R2_thr2 = R2_all[threshold2]
    errors_thr2 = errors[threshold2]
    
    # Save to npz file
    npz_file = osp.join(output_path, f'{args.model_type}_{prediction_short}_{args.hemisphere}_{model_name}_test_results.npz')
    np.savez_compressed(
        npz_file,
        # All data
        predicted_values=y_hat_all,
        measured_values=y_all,
        R2=R2_all,
        errors=errors,
        # Thresholded data (R2 > 2.2)
        predicted_values_thr=y_hat_thr,
        measured_values_thr=y_thr,
        R2_thr=R2_thr,
        errors_thr=errors_thr,
        threshold_mask=threshold,
        # Thresholded data (R2 > 17)
        predicted_values_thr2=y_hat_thr2,
        measured_values_thr2=y_thr2,
        R2_thr2=R2_thr2,
        errors_thr2=errors_thr2,
        threshold2_mask=threshold2,
        # Metrics
        MAE=test_output['MAE'],
        MAE_thr=test_output['MAE_thr'],
        # Additional statistics
        mean_error=np.mean(errors),
        median_error=np.median(errors),
        mean_error_thr=np.mean(errors_thr),
        median_error_thr=np.median(errors_thr),
        mean_error_thr2=np.mean(errors_thr2),
        median_error_thr2=np.median(errors_thr2)
    )
    print(f"Saved test results to npz: {npz_file}")

# =============================
# Test Set Evaluation
# =============================
if args.run_test:
    print("\n" + "="*50)
    print("Starting test set evaluation...")
    print("="*50)
    
    def evaluate_test_set(model_to_evaluate, model_name="model"):
        """Evaluate model on test set"""
        model_to_evaluate.eval()
        MeanAbsError = 0
        MeanAbsError_thr = 0
        y = []
        y_hat = []
        R2_plot = []
        
        with torch.no_grad():
            for data in test_loader:
                pred = model_to_evaluate(data.to(device)).detach()
                y_hat.append(pred)
                y.append(data.to(device).y.view(-1))
                
                R2 = data.R2.view(-1)
                R2_plot.append(R2)
                threshold = R2.view(-1) > 2.2
                threshold2 = R2.view(-1) > 17
                
                MAE = torch.mean(abs(data.to(device).y.view(-1)[threshold == 1] - pred[
                    threshold == 1])).item()
                MAE_thr = torch.mean(abs(
                    data.to(device).y.view(-1)[threshold2 == 1] - pred[
                        threshold2 == 1])).item()
                MeanAbsError_thr += MAE_thr
                MeanAbsError += MAE
        
        test_MAE = MeanAbsError / len(test_loader)
        test_MAE_thr = MeanAbsError_thr / len(test_loader)
        output = {'Predicted_values': y_hat, 'Measured_values': y, 'R2': R2_plot,
                  'MAE': test_MAE, 'MAE_thr': test_MAE_thr}
        return output
    
    # Evaluate best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        best_test_output = evaluate_test_set(model, "best")
        print(f"\nBest Model (epoch {best_epoch}) Test Results:")
        print(f"  Test MAE: {best_test_output['MAE']:.4f}")
        print(f"  Test MAE_thr: {best_test_output['MAE_thr']:.4f}")
        
        # Save best model test results
        best_test_file = osp.join(
            output_path,
            f'{args.model_type}_{prediction_short}_{args.hemisphere}_best_test_results.pt'
        )
        torch.save({
            'Epoch': best_epoch,
            'Predicted_values': best_test_output['Predicted_values'],
            'Measured_values': best_test_output['Measured_values'],
            'R2': best_test_output['R2'],
            'Test_MAE': best_test_output['MAE'],
            'Test_MAE_thr': best_test_output['MAE_thr']
        }, best_test_file)
        
        if run is not None:
            run["test/best_model/mae"] = best_test_output['MAE']
            run["test/best_model/mae_thr"] = best_test_output['MAE_thr']
            run["test/best_model/epoch"] = best_epoch
            run["artifacts/best_test_results"].upload(best_test_file)
        
        # Save test results to npz for best model
        print("\nSaving test results to npz for best model...")
        save_test_results_npz(best_test_output, 'best', output_path, args, prediction_short)
    
    # Evaluate final model
    # In test_only_mode, model is already loaded from Neptune checkpoint
    if not test_only_mode:
        # Only load from file if not in test-only mode
        final_model_file = osp.join(
            output_path,
            f'{args.model_type}_{prediction_short}_{args.hemisphere}_final_model.pt'
        )
        if osp.exists(final_model_file):
            model.load_state_dict(torch.load(final_model_file, map_location=device))
        else:
            print(f"Warning: Final model file not found: {final_model_file}")
            print("Using current model state for final model evaluation.")
    
    final_test_output = evaluate_test_set(model, "final")
    epoch_str = f"epoch {final_epoch}" if final_epoch > 0 else "unknown epoch"
    print(f"\nFinal Model ({epoch_str}) Test Results:")
    print(f"  Test MAE: {final_test_output['MAE']:.4f}")
    print(f"  Test MAE_thr: {final_test_output['MAE_thr']:.4f}")
    
    # Save final model test results
    final_test_file = osp.join(
        output_path,
        f'{args.model_type}_{prediction_short}_{args.hemisphere}_final_test_results.pt'
    )
    torch.save({
        'Epoch': final_epoch,
        'Predicted_values': final_test_output['Predicted_values'],
        'Measured_values': final_test_output['Measured_values'],
        'R2': final_test_output['R2'],
        'Test_MAE': final_test_output['MAE'],
        'Test_MAE_thr': final_test_output['MAE_thr']
    }, final_test_file)
    
    if run is not None:
        run["test/final_model/mae"] = final_test_output['MAE']
        run["test/final_model/mae_thr"] = final_test_output['MAE_thr']
        run["test/final_model/epoch"] = final_epoch
        run["artifacts/final_test_results"].upload(final_test_file)
    
    # Save test results to npz for final model
    print("\nSaving test results to npz for final model...")
    save_test_results_npz(final_test_output, 'final', output_path, args, prediction_short)
    
    print("\nTest set evaluation completed.")

# Close Neptune runs
if run is not None:
    run.stop()

if neptune_run_for_test is not None:
    neptune_run_for_test.stop()

if test_only_mode:
    print(f"\nTest-only evaluation completed. Output directory: {output_path}")
else:
    print(f"\nTraining completed. Output directory: {output_path}")

