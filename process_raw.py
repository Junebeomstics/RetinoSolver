import os.path as osp
import torch_geometric.transforms as T
from Retinotopy.dataset.HCP_3sets_ROI import Retinotopy
# from Retinotopy.dataset.HCP_3sets_ROI_rotated import Retinotopy as RetinotopyRotated
# from Retinotopy.dataset.HCP_3sets_ROI_notwin import Retinotopy as RetinotopyNotwin
# from Retinotopy.dataset.HCP_3sets_ROI_splitHalves import Retinotopy as RetinotopySplitHalves

# Set data path
path = osp.join(osp.dirname(osp.realpath(__file__)), 'Retinotopy', 'data')

pre_transform = T.Compose([T.FaceToEdge()])

# Basic ROI variant - Eccentricity
print("Processing Eccentricity - Left Hemisphere...")
Retinotopy(path, 'Train', transform=T.Cartesian(), pre_transform=pre_transform, 
           n_examples=181, prediction='eccentricity', myelination=True, hemisphere='Left')
print("Processing Eccentricity - Right Hemisphere...")
Retinotopy(path, 'Train', transform=T.Cartesian(), pre_transform=pre_transform, 
           n_examples=181, prediction='eccentricity', myelination=True, hemisphere='Right')

# Basic ROI variant - Polar Angle
print("Processing Polar Angle - Left Hemisphere...")
Retinotopy(path, 'Train', transform=T.Cartesian(), pre_transform=pre_transform, 
           n_examples=181, prediction='polarAngle', myelination=True, hemisphere='Left')
print("Processing Polar Angle - Right Hemisphere...")
Retinotopy(path, 'Train', transform=T.Cartesian(), pre_transform=pre_transform, 
           n_examples=181, prediction='polarAngle', myelination=True, hemisphere='Right')

# # Rotated variant
# print("Processing Rotated - Left Hemisphere...")
# RetinotopyRotated(path, 'Train', transform=T.Cartesian(), pre_transform=pre_transform, 
#                   n_examples=181, prediction='polarAngle', myelination=True, hemisphere='Left')

# # Notwin variant
# print("Processing Notwin - Left Hemisphere...")
# RetinotopyNotwin(path, 'Train', transform=T.Cartesian(), pre_transform=pre_transform, 
#                  n_examples=181, prediction='polarAngle', myelination=True, hemisphere='Left')

# # Split Halves variant (fit2, fit3)
# print("Processing Split Halves fit2 - Left Hemisphere...")
# RetinotopySplitHalves(path, 'Train', transform=T.Cartesian(), pre_transform=pre_transform, 
#                       n_examples=181, prediction='polarAngle', myelination=True, 
#                       hemisphere='Left', fit=2)
# print("Processing Split Halves fit3 - Left Hemisphere...")
# RetinotopySplitHalves(path, 'Train', transform=T.Cartesian(), pre_transform=pre_transform, 
#                       n_examples=181, prediction='polarAngle', myelination=True, 
#                       hemisphere='Left', fit=3)

print("All processing completed!")