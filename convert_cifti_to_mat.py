import nibabel as nib
import numpy as np
import scipy.io
import os
import os.path as osp
from collections import OrderedDict

# ============================================================================
# 설정: 키워드에 따라 파일명과 matlab 키 이름이 결정됨
# ============================================================================
KEYWORD = 'PolarAngle'  # 'PolarAngle', 'Eccentricity', 'pRFsize', 'R2' 등으로 변경 가능
KEYMAP = {'PolarAngle': 'polarAngle', 'Eccentricity': 'eccentricity', 'ReceptiveFieldSize': 'pRFsize', 'R2': 'R2'}

# 파일 경로 설정
cifti_file = f'S1200_7T_Retinotopy181.All.Fit1_{KEYWORD}_MSMAll.32k_fs_LR.dscalar.nii'
mat_file_path = f'Retinotopy/data/raw/converted/cifti_{KEYMAP[KEYWORD]}_all.mat'
list_subj_path = 'Retinotopy/data/list_subj'
output_mat_path = f'Retinotopy/data/raw/converted/cifti_{KEYMAP[KEYWORD]}_all_new.mat'

# matlab 키 이름 설정 (기존 형식에 맞춤)
matlab_key = f'cifti_{KEYMAP[KEYWORD]}'
subject_key_suffix = f'_fit1_{KEYWORD.lower().replace("polarangle", "polarangle").replace("eccentricity", "eccentricity").replace("prfsize", "receptivefieldsize").replace("r2", "r2")}_msmall'

# ============================================================================
# 1. CIFTI 파일 로드 및 구조 확인
# ============================================================================
print("=" * 80)
print(f"Converting CIFTI file to MATLAB format: {KEYWORD}")
print("=" * 80)

print(f"\n1. Loading CIFTI file: {cifti_file}")
cifti_img = nib.load(cifti_file)
data = cifti_img.get_fdata()
print(f"   Data shape: {data.shape}")

# Brain axis 정보 가져오기
brain_axis = cifti_img.header.get_axis(1)

# CORTEX_LEFT와 CORTEX_RIGHT의 vertex indices 찾기
vertex_indices_L = None
vertex_indices_R = None
left_slice = None
right_slice = None

print("\n2. Extracting vertex indices for CORTEX_LEFT and CORTEX_RIGHT...")
for name, slice_obj, bm in brain_axis.iter_structures():
    if 'CORTEX_LEFT' in name:
        vertex_indices_L = brain_axis.vertex[slice_obj]
        left_slice = slice_obj
        print(f"   Found CORTEX_LEFT: {len(vertex_indices_L)} vertices")
        print(f"      Min vertex: {vertex_indices_L.min()}, Max vertex: {vertex_indices_L.max()}")
    elif 'CORTEX_RIGHT' in name:
        vertex_indices_R = brain_axis.vertex[slice_obj]
        right_slice = slice_obj
        print(f"   Found CORTEX_RIGHT: {len(vertex_indices_R)} vertices")
        print(f"      Min vertex: {vertex_indices_R.min()}, Max vertex: {vertex_indices_R.max()}")

if vertex_indices_L is None or vertex_indices_R is None:
    raise ValueError("Could not find CORTEX_LEFT or CORTEX_RIGHT in CIFTI file")

# ============================================================================
# 2. Subject 목록 로드
# ============================================================================
print("\n3. Loading subject list...")
with open(list_subj_path) as fp:
    subjects = fp.read().split("\n")
subjects = [s.strip() for s in subjects if s.strip()]  # 빈 줄 제거
print(f"   Total subjects: {len(subjects)}")
print(f"   First 5 subjects: {subjects[:5]}")

# ============================================================================
# 3. 각 subject에 대해 데이터 변환
# ============================================================================
print(f"\n4. Processing {len(subjects)} subjects...")
number_hemi_nodes = 32492
number_cortical_nodes = 64984

# matlab 구조체를 위한 딕셔너리 생성
matlab_list = []
subj_list = []

matlab_list.append(np.array(['pos'], dtype='<U3'))
subj_list.append(('dimord', 'O'))

for idx, subject_id in enumerate(subjects):
    if idx % 20 == 0:
        print(f"   Processing subject {idx+1}/{len(subjects)}: {subject_id}")
    
    # 해당 subject의 데이터 가져오기 (CIFTI 파일의 각 row가 하나의 subject)
    subject_data = data[idx, :]  # shape: (91282,)
    
    # 좌반구 데이터 추출 및 변환
    left_data_raw = subject_data[left_slice]
    full_left = np.full(number_hemi_nodes, np.nan, dtype=np.float32)
    full_left[vertex_indices_L] = left_data_raw
    
    # 우반구 데이터 추출 및 변환
    right_data_raw = subject_data[right_slice]
    full_right = np.full(number_hemi_nodes, np.nan, dtype=np.float32)
    full_right[vertex_indices_R] = right_data_raw

    rest = subject_data[number_cortical_nodes:]
    
    # 좌반구 + 우반구 결합 (64984, 1) 형태
    combined_data = np.concatenate([full_left, full_right, rest]).reshape(-1, 1)
    
    # matlab 키 이름 생성
    subject_key = f'x{subject_id}{subject_key_suffix}'
    
    matlab_list.append(np.array(combined_data))
    subj_list.append((subject_key, 'O'))

print(matlab_list[0].shape)


# 최상위 키로 래핑
final_dict = {matlab_key: np.array(np.array([[matlab_list]]), dtype=object)}

print(final_dict[matlab_key][0][0])

# ============================================================================
# 4. MATLAB 파일로 저장
# ============================================================================
print(f"\n5. Saving to MATLAB file: {output_mat_path}")
os.makedirs(osp.dirname(output_mat_path), exist_ok=True)
scipy.io.savemat(output_mat_path, final_dict, oned_as='column')
print("   Saved successfully!")

