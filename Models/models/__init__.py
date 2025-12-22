"""
Model definitions for deepRetinotopy
"""
from models.baseline import deepRetinotopy_Baseline
from models.transolver_optionA import deepRetinotopy_OptionA
from models.transolver_optionB import deepRetinotopy_OptionB
from models.transolver_optionC import deepRetinotopy_OptionC
from models.fno import FNOForRetinotopy, SimpleFNO, GraphToGridFNO, FNO1d

__all__ = [
    'deepRetinotopy_Baseline',
    'deepRetinotopy_OptionA',
    'deepRetinotopy_OptionB',
    'deepRetinotopy_OptionC',
    'FNOForRetinotopy',
    'SimpleFNO',
    'GraphToGridFNO',
    'FNO1d',
]

