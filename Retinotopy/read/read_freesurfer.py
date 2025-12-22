import os.path as osp
import scipy.io
import numpy as np
import torch
import nibabel as nib
from torch_geometric.data import Data


def read_freesurfer(path, hemisphere=None, sub_id=None,
                    visual_mask_L=None, visual_mask_R=None,
                    faces_L=None, faces_R=None, myelination=False):
    """Read FreeSurfer data files and create a data object with attributes x, y, pos, faces and R2.

    Args:
        path (string): Path to FreeSurfer subjects directory (containing subject folders)
        hemisphere (string): 'Left' or 'Right' hemisphere
        sub_id (string): Subject ID (folder name)
        visual_mask_L (numpy array): Mask of the region of interest from left hemisphere (32492,)
        visual_mask_R (numpy array): Mask of the region of interest from right hemisphere (32492,)
        faces_L (numpy array): Triangular faces from the region of interest (number of faces, 3) in the left hemisphere
        faces_R (numpy array): Triangular faces from the region of interest (number of faces, 3) in the right hemisphere
        myelination (boolean): True if myelin values will be used as an additional feature

    Returns:
        data (object): object of class Data (from torch_geometric.data) with attributes x, y, pos, faces and R2.
    """
    # Defining number of nodes
    number_cortical_nodes = int(64984)
    number_hemi_nodes = int(number_cortical_nodes / 2)
    
    # Determine hemisphere string for file paths
    if hemisphere in ['Left', 'LH', 'left', 'lh']:
        hemi_str = 'lh'
        hemi_long = 'Left'
        faces = torch.tensor(faces_L.T, dtype=torch.long)
        visual_mask = visual_mask_L
    elif hemisphere in ['Right', 'RH', 'right', 'rh']:
        hemi_str = 'rh'
        hemi_long = 'Right'
        faces = torch.tensor(faces_R.T, dtype=torch.long)
        visual_mask = visual_mask_R
    else:
        raise ValueError(f"Invalid hemisphere: {hemisphere}. Must be 'Left'/'LH'/'left'/'lh' or 'Right'/'RH'/'right'/'rh'")
    
    # Coordinates of the hemisphere vertices (from template)
    # Template files are in Retinotopy/data/raw/converted/
    template_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'raw', 'converted', 
                            f'mid_pos_{hemi_long[0]}.mat')
    if not osp.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    pos = torch.tensor((scipy.io.loadmat(template_path)[f'mid_pos_{hemi_long[0]}'].reshape(
            (number_hemi_nodes, 3))[visual_mask == 1]),
        dtype=torch.float)
    
    # Load curvature file
    curvature_file = osp.join(path, sub_id, 'surf', 
                             f'{sub_id}.curvature-midthickness.{hemi_str}.32k_fs_LR.func.gii')
    
    if not osp.exists(curvature_file):
        raise FileNotFoundError(f"Curvature file not found: {curvature_file}")
    
    curvature = torch.tensor(
        np.array(nib.load(curvature_file).agg_data()).reshape(
            number_hemi_nodes, -1)[visual_mask == 1], 
        dtype=torch.float
    )
    
    # Remove NaN values
    nocurv = np.isnan(curvature)
    curvature[nocurv == 1] = 0
    
    # Prepare features
    if myelination:
        # Load myelin map file
        myelin_file = osp.join(path, sub_id, 'surf',
                              f'{sub_id}.myelin-midthickness.{hemi_str}.32k_fs_LR.func.gii')
        
        if not osp.exists(myelin_file):
            # Try alternative names
            alt_myelin_file = osp.join(path, sub_id, 'surf',
                                      f'{hemi_str}.SmoothedMyelinMap')
            if osp.exists(alt_myelin_file):
                # Convert to GIFTI and resample if needed
                print(f"Warning: Found {alt_myelin_file} but need resampled version. "
                      f"Please run preprocessing step to generate {myelin_file}")
                # For now, create dummy myelin values
                myelin_values = torch.zeros_like(curvature)
            else:
                print(f"Warning: Myelin map not found: {myelin_file}. Using zeros.")
                myelin_values = torch.zeros_like(curvature)
        else:
            myelin_values = torch.tensor(
                np.array(nib.load(myelin_file).agg_data()).reshape(
                    number_hemi_nodes, -1)[visual_mask == 1],
                dtype=torch.float
            )
        
        # Remove NaN values
        nomyelin = np.isnan(myelin_values)
        myelin_values[nomyelin == 1] = 0
        
        # Concatenate curvature and myelin as features
        x = torch.cat((curvature, myelin_values), 1)
    else:
        x = curvature
    
    # Create data object
    data = Data(x=x, pos=pos)
    data.face = faces
    
    # Create dummy R2 values (not available for new subjects)
    # Set to 1.0 for all vertices to indicate they should be included
    data.R2 = torch.ones((x.shape[0], 1), dtype=torch.float)
    
    return data

