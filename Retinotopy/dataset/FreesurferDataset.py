import os.path as osp
import scipy.io
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from Retinotopy.read.read_freesurfer import read_freesurfer
from Retinotopy.functions.labels import labels
from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi


class FreesurferDataset(InMemoryDataset):
    """Dataset for FreeSurfer subjects for inference with pre-trained models."""

    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 list_subs=None,
                 myelination=False,
                 prediction=None,
                 hemisphere=None):
        """
        Args:
            root (string): Path to FreeSurfer subjects directory
            transform: Transform to apply to data
            pre_transform: Pre-transform to apply before saving
            pre_filter: Pre-filter function
            list_subs (list): List of subject IDs to process
            myelination (boolean): True if myelin values will be used as an additional feature
            prediction (string): Prediction type ('polarAngle', 'eccentricity', or 'pRFsize')
            hemisphere (string): Hemisphere ('Left' or 'Right')
        """
        self.root = root
        self.list_subs = list_subs if list_subs is not None else []
        self.myelination = myelination
        self.prediction = prediction
        self.hemisphere = hemisphere
        super(FreesurferDataset, self).__init__(root, transform, pre_transform, pre_filter)
        # Explicitly load data after processing
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        """Generate processed file name based on parameters"""
        hemi_short = 'LH' if self.hemisphere in ['Left', 'LH', 'left', 'lh'] else 'RH'
        myelination_suffix = '_myelin' if self.myelination else ''
        pred_short = {
            'eccentricity': 'ecc',
            'polarAngle': 'PA',
            'pRFsize': 'pRFsize'
        }.get(self.prediction, 'unknown')
        
        return [f'freesurfer_{pred_short}_{hemi_short}{myelination_suffix}.pt']

    def process(self):
        """Process FreeSurfer data and create PyTorch Geometric data objects"""
        # Selecting all visual areas (Wang2015) plus V1-3 fovea
        label_primary_visual_areas = ['ROI']
        final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(label_primary_visual_areas)

        # Load template faces (from Retinotopy/data/raw/converted/)
        template_dir = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'raw', 'converted')
        faces_R = labels(
            scipy.io.loadmat(osp.join(template_dir, 'tri_faces_R.mat'))['tri_faces_R'] - 1, 
            index_R_mask
        )
        faces_L = labels(
            scipy.io.loadmat(osp.join(template_dir, 'tri_faces_L.mat'))['tri_faces_L'] - 1, 
            index_L_mask
        )

        data_list = []

        for i in range(len(self.list_subs)):
            try:
                data = read_freesurfer(
                    self.root,
                    hemisphere=self.hemisphere,
                    sub_id=self.list_subs[i],
                    visual_mask_L=final_mask_L,
                    visual_mask_R=final_mask_R,
                    faces_L=faces_L,
                    faces_R=faces_R,
                    myelination=self.myelination
                )
                
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                data_list.append(data)
            except Exception as e:
                print(f"Error processing subject {self.list_subs[i]}: {e}")
                continue

        if len(data_list) == 0:
            raise RuntimeError("No valid subjects processed. Please check your data files.")

        # Save processed data
        torch.save(self.collate(data_list), self.processed_paths[0])

