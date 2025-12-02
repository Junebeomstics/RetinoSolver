# Unified Training Script for deepRetinotopy Models

이 스크립트는 baseline 모델과 Transolver 기반 모델들을 통합하여 하나의 스크립트로 다양한 실험을 실행할 수 있도록 합니다.

## 파일 구조

```
Models/
├── train_unified.py              # 통합 학습 스크립트
├── run_all_experiments.sh        # 모든 실험 조합을 실행하는 스크립트
├── run_single_experiment.sh      # 단일 실험을 실행하는 예제 스크립트
├── models/                       # 모델 클래스들
│   ├── __init__.py
│   ├── baseline.py                # Baseline 모델
│   ├── transolver_optionA.py     # Transolver Option A 모델
│   ├── transolver_optionB.py     # Transolver Option B 모델
│   ├── physics_attention.py      # Physics Attention 모듈들
│   └── utils.py                  # 유틸리티 함수들
└── README_unified_training.md    # 이 파일
```

## 출력 디렉토리 구조

실험 결과는 다음과 같은 구조로 저장됩니다:

```
output/
├── baseline_ecc_Left/
│   ├── baseline_ecc_Left_output_epoch25.pt
│   ├── baseline_ecc_Left_output_epoch50.pt
│   ├── ...
│   └── baseline_ecc_Left_model.pt
├── baseline_ecc_Right/
├── baseline_PA_Left/
├── transolver_optionA_ecc_Left/
├── transolver_optionA_ecc_Right/
├── transolver_optionA_PA_Left/
├── transolver_optionB_ecc_Left/
└── ...
```

각 실험은 `{model_type}_{prediction}_{hemisphere}` 형식의 별도 폴더에 저장됩니다.

## 사용법

### 1. 단일 실험 실행 (Docker 사용)

**Docker를 사용한 실행 (권장):**

```bash
# 기본 사용법
./run_single_experiment.sh baseline eccentricity Left

# Docker 이미지 지정
DOCKER_IMAGE=your_image:tag ./run_single_experiment.sh baseline eccentricity Left

# GPU 비활성화
USE_GPU=false ./run_single_experiment.sh baseline eccentricity Left

# Neptune 비활성화
USE_NEPTUNE=false ./run_single_experiment.sh baseline eccentricity Left
```

스크립트는 자동으로:
- Docker 이미지 존재 여부 확인
- 필요한 패키지 (neptune, einops) 설치
- 도커 컨테이너에서 학습 실행

**로컬에서 직접 실행:**

```bash
python train_unified.py \
    --model_type baseline \
    --prediction eccentricity \
    --hemisphere Left \
    --n_epochs 200 \
    --lr_init 0.01 \
    --lr_decay_epoch 100 \
    --lr_decay 0.005
```

### 2. 모든 실험 실행

```bash
./run_all_experiments.sh
```

이 스크립트는 다음 조합을 모두 실행합니다:
- Model types: `baseline`, `transolver_optionA`, `transolver_optionB`
- Predictions: `eccentricity`, `polarAngle`
- Hemispheres: `Left`, `Right`

총 12개의 실험이 실행됩니다 (3 × 2 × 2).

### 3. Neptune 로깅 활성화

Neptune을 사용하려면:

```bash
python train_unified.py \
    --model_type baseline \
    --prediction eccentricity \
    --hemisphere Left \
    --use_neptune \
    --project your_project_name \
    --api_token your_api_token
```

또는 환경 변수 사용:

```bash
export NEPTUNE_API_TOKEN=your_api_token
python train_unified.py \
    --model_type baseline \
    --prediction eccentricity \
    --hemisphere Left \
    --use_neptune \
    --project your_project_name
```

## 주요 Arguments

### 필수 Arguments

- `--model_type`: 모델 타입 선택
  - `baseline`: 기본 SplineConv 모델
  - `transolver_optionA`: Transolver Physics Attention (edge 정보 미사용)
  - `transolver_optionB`: Transolver Physics Attention (edge 정보 간접 활용)

- `--prediction`: 예측 타겟
  - `eccentricity`: Eccentricity 예측
  - `polarAngle`: Polar Angle 예측

- `--hemisphere`: 반구 선택
  - `Left`: 좌반구
  - `Right`: 우반구

### 선택적 Arguments

- `--n_epochs`: 학습 epoch 수 (기본값: 200)
- `--lr_init`: 초기 learning rate (기본값: 0.01)
- `--lr_decay_epoch`: Learning rate 감소 epoch (기본값: 100)
- `--lr_decay`: Learning rate 감소 후 값 (기본값: 0.005)
- `--interm_save_every`: 중간 결과 저장 주기 (기본값: 25)
- `--batch_size`: 배치 크기 (기본값: 1)
- `--n_examples`: 예제 수 (기본값: 181)
- `--output_dir`: 출력 디렉토리 (기본값: ./output)
- `--myelination`: Myelination 특징 사용 여부 (기본값: True)

### Neptune Arguments

- `--use_neptune`: Neptune 로깅 활성화
- `--project`: Neptune 프로젝트 이름
- `--api_token`: Neptune API 토큰 (환경 변수 `NEPTUNE_API_TOKEN` 사용 가능)

## 모델 설명

### Baseline Model
- 순수 SplineConv 기반 모델
- 12개의 SplineConv 레이어로 구성
- Edge 정보를 직접 활용

### Transolver Option A
- SplineConv + Physics Attention 하이브리드 모델
- Edge 정보를 사용하지 않음
- Physics Attention은 노드 특징만 사용하여 물리적 상태를 학습

### Transolver Option B
- SplineConv + Physics Attention 하이브리드 모델
- Edge 정보를 특징으로 인코딩하여 활용
- K-NN 거리, 노드 degree, local density 등을 특징으로 변환

## 예제

### Eccentricity 예측 (Left Hemisphere, Baseline)

```bash
python train_unified.py \
    --model_type baseline \
    --prediction eccentricity \
    --hemisphere Left
```

### Polar Angle 예측 (Right Hemisphere, Transolver Option A)

```bash
python train_unified.py \
    --model_type transolver_optionA \
    --prediction polarAngle \
    --hemisphere Right
```

### Transolver Option B with Neptune

```bash
python train_unified.py \
    --model_type transolver_optionB \
    --prediction eccentricity \
    --hemisphere Left \
    --use_neptune \
    --project your_project \
    --api_token your_token
```

## 주의사항

### Docker 사용 시

1. **Docker 이미지 준비:**
   ```bash
   # 프로젝트 루트에서 이미지 빌드
   docker build -t vnmd/deepretinotopy_1.0.18:latest .
   ```

2. **필요한 패키지 자동 설치:**
   - 스크립트가 자동으로 `neptune`과 `einops`를 설치합니다
   - 매 실행 시 설치하므로 첫 실행 시 약간의 시간이 소요될 수 있습니다

3. **Docker 이미지 이름 변경:**
   - 환경 변수로 지정 가능: `DOCKER_IMAGE=your_image:tag`

4. **GPU 사용:**
   - 기본적으로 GPU를 사용합니다 (`USE_GPU=true`)
   - GPU를 사용하지 않으려면: `USE_GPU=false`

### 일반 주의사항

1. Neptune을 사용하지 않으면 `USE_NEPTUNE=false`로 설정하거나 스크립트에서 수정하세요.

2. 각 실험은 독립적으로 실행되며, 결과는 별도의 폴더에 저장됩니다.

3. Docker를 사용하지 않는 경우, 로컬에 다음 패키지가 필요합니다:
   ```bash
   pip install einops neptune
   ```

