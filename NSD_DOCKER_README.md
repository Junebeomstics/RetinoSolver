# deepRetinotopy NSD Evaluation - Docker Version

Docker를 사용한 NSD 데이터셋 평가 파이프라인입니다. Singularity 대신 Docker를 사용하여 더 널리 사용되는 환경에서 실행할 수 있습니다.

## 🐳 Docker 버전 vs Singularity 버전

| Feature | Singularity 버전 | Docker 버전 |
|---------|-----------------|------------|
| 컨테이너 기술 | Singularity | Docker |
| 권장 환경 | HPC 클러스터 | 개인 워크스테이션, 서버 |
| GPU 지원 | ✓ | ✓ (nvidia-docker 필요) |
| 파일 시스템 | 자동 마운트 | 명시적 볼륨 마운트 |
| 스크립트 | `run_nsd_inference.sh` | `run_nsd_inference_docker.sh` |

## 📋 요구사항

### 필수 소프트웨어

1. **Docker** (20.10+)
   ```bash
   # 설치 확인
   docker --version
   ```

2. **NVIDIA Docker** (GPU 사용 시)
   ```bash
   # GPU 지원 확인
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

3. **Python 3.7+** (평가 스크립트용, 호스트에서 실행)
   ```bash
   pip install numpy scipy nibabel matplotlib seaborn pandas
   ```

### 데이터 요구사항

동일하게 다음 데이터가 필요합니다:
- NSD FreeSurfer 데이터
- Model checkpoints
- HCP surface templates

자세한 내용은 `NSD_EVALUATION_README.md` 참조

## 🚀 사용법

### Quick Start

**1. 테스트 실행 (권장)**

```bash
./test_nsd_docker.sh
```

**2. 단일 평가**

```bash
./run_nsd_inference_docker.sh -s subj01 -h lh -p eccentricity
```

**3. 전체 평가**

```bash
./run_nsd_full_evaluation_docker.sh -s subj01
```

### 주요 옵션

#### `run_nsd_inference_docker.sh`

```bash
./run_nsd_inference_docker.sh [options]

Options:
  -s SUBJECT      Subject ID (default: subj01)
  -h HEMISPHERE   Hemisphere: lh or rh (default: lh)
  -p PREDICTION   Prediction: eccentricity, polarAngle, pRFsize (default: eccentricity)
  -m MODEL        Model type (default: baseline)
  -y MYELINATION  Use myelination: True or False (default: False)
  -r R2_THRESHOLD R2 threshold (default: 0.1)
  -j N_JOBS       Parallel jobs (default: auto)
  -o OUTPUT_DIR   Output directory (default: ./nsd_evaluation)
  -g USE_GPU      Use GPU: true or false (default: true)
```

#### `run_nsd_full_evaluation_docker.sh`

```bash
./run_nsd_full_evaluation_docker.sh [options]

Options:
  -s SUBJECT      Subject ID (default: subj01)
  -m MODEL        Model type (default: baseline)
  -y MYELINATION  Use myelination: True or False (default: False)
  -r R2_THRESHOLD R2 threshold (default: 0.1)
  -o OUTPUT_DIR   Output directory (default: ./nsd_evaluation)
  -g USE_GPU      Use GPU: true or false (default: true)
  -c CONCURRENT   Max concurrent jobs (default: 2)
```

### 예제

**예제 1**: GPU 없이 실행

```bash
./run_nsd_inference_docker.sh -s subj01 -h lh -p eccentricity -g false
```

**예제 2**: Polar angle 평가 (myelination 포함)

```bash
./run_nsd_inference_docker.sh -s subj01 -h rh -p polarAngle -y True
```

**예제 3**: 전체 평가 (병렬 4개)

```bash
./run_nsd_full_evaluation_docker.sh -s subj01 -c 4
```

**예제 4**: 여러 피험자 평가

```bash
for subj in subj01 subj02 subj03; do
    ./run_nsd_full_evaluation_docker.sh -s $subj -o ./results_${subj}
done
```

**예제 5**: 사용자 정의 Docker 이미지

```bash
export DOCKER_IMAGE="my-docker-registry/deepretinotopy:custom"
./run_nsd_inference_docker.sh -s subj01 -h lh -p eccentricity
```

## 🔧 Docker 설정

### 컨테이너 관리

**컨테이너 상태 확인**
```bash
docker ps --filter name=deepretinotopy_nsd_eval
```

**컨테이너 중지**
```bash
docker stop deepretinotopy_nsd_eval
```

**컨테이너 제거**
```bash
docker rm -f deepretinotopy_nsd_eval
```

**새로 시작 (컨테이너 재생성)**
```bash
docker rm -f deepretinotopy_nsd_eval
./run_nsd_inference_docker.sh -s subj01 -h lh -p eccentricity
```

### 볼륨 마운트

스크립트는 자동으로 다음 디렉토리를 마운트합니다:

| 호스트 경로 | 컨테이너 경로 | 설명 |
|-----------|-------------|-----|
| `$PROJECT_ROOT` | `/workspace` | deepRetinotopy 프로젝트 |
| `$NSD_DIR` | `/mnt/nsd_freesurfer` | NSD FreeSurfer 데이터 |
| `$HCP_SURFACE_DIR` | `/mnt/hcp_surface` | HCP surface templates |

### 사용자 정의 설정

**환경 변수로 Docker 이미지 변경**
```bash
export DOCKER_IMAGE="vnmd/deepretinotopy:1.0.19"
./run_nsd_inference_docker.sh -s subj01 -h lh -p eccentricity
```

**NSD 데이터 경로 변경**
스크립트 상단의 `NSD_DIR` 변수 수정:
```bash
# run_nsd_inference_docker.sh 내부
NSD_DIR="/your/custom/path/to/nsd/freesurfer"
```

## 📊 출력 구조

출력은 Singularity 버전과 동일합니다:

```
nsd_evaluation/
├── plots/                  # 시각화
│   ├── *_scatter.png
│   └── *_distribution.png
├── results/                # JSON 메트릭
│   └── *_metrics.json
├── summary_table.csv       # 요약 테이블
├── summary_report.txt      # 텍스트 리포트
├── comparison_plot.png     # 비교 플롯
├── correlation_heatmap.png # 히트맵
└── *.log                   # 개별 작업 로그 (전체 평가 시)
```

## 🐛 문제 해결

### Docker 관련 이슈

**Issue: "docker: command not found"**

**Solution**: Docker 설치
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
# Log out and log back in
```

**Issue: "permission denied while trying to connect to Docker daemon"**

**Solution**: 사용자를 docker 그룹에 추가
```bash
sudo usermod -aG docker $USER
newgrp docker  # Or log out and log back in
```

**Issue: "could not select device driver with capabilities: [[gpu]]"**

**Solution**: NVIDIA Docker 설치
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

또는 GPU 없이 실행:
```bash
./run_nsd_inference_docker.sh -s subj01 -h lh -p eccentricity -g false
```

**Issue: "Docker image not found"**

**Solution**: 이미지 pull
```bash
docker pull vnmd/deepretinotopy_1.0.18:latest
```

### 실행 이슈

**Issue: "No checkpoint file found"**

**Solution**: Checkpoint 파일 확인
```bash
ls -lah Models/checkpoints/eccentricity_Left_baseline_noMyelin/
```

**Issue: "Subject directory not found"**

**Solution**: NSD 데이터 경로 확인
```bash
ls -lah /mnt/external_storage1/natural-scenes-dataset/nsddata/freesurfer/subj01/
```

**Issue: 평가 단계에서 Python 패키지 에러**

**Solution**: 호스트에 필요한 패키지 설치
```bash
pip install numpy scipy nibabel matplotlib seaborn pandas
```

### 성능 이슈

**Issue: 너무 느림**

**Solution**: 병렬 작업 수 증가
```bash
./run_nsd_full_evaluation_docker.sh -s subj01 -c 4  # 동시 4개
```

**Issue: 메모리 부족**

**Solution**: 병렬 작업 수 감소 또는 Docker 메모리 제한 증가
```bash
# 병렬 작업 감소
./run_nsd_full_evaluation_docker.sh -s subj01 -c 1

# Docker 메모리 설정 (Docker Desktop)
# Settings → Resources → Advanced → Memory 증가
```

## 📈 성능 비교

| 방식 | 속도 | 메모리 | GPU 지원 | 사용 편의성 |
|-----|------|--------|---------|----------|
| Singularity | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (HPC) |
| Docker | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (개인) |

## 🔄 Singularity 버전으로 전환

Singularity 버전을 사용하려면:

```bash
# Singularity 버전 사용
./run_nsd_inference.sh -s subj01 -h lh -p eccentricity
./run_nsd_full_evaluation.sh -s subj01
```

## 💡 고급 사용

### 컨테이너 내부 접속

```bash
# 컨테이너가 실행 중일 때
docker exec -it deepretinotopy_nsd_eval bash

# 내부에서 명령 실행
cd /workspace
ls Models/checkpoints/
```

### 로그 확인

```bash
# 전체 평가 시 생성된 로그 확인
cat nsd_evaluation/subj01_lh_eccentricity.log

# Docker 컨테이너 로그
docker logs deepretinotopy_nsd_eval
```

### 수동 단계 실행

```bash
# 컨테이너 시작
docker run -d --gpus all --name my_eval \
    -v $(pwd):/workspace \
    -v /mnt/external_storage1/natural-scenes-dataset/nsddata/freesurfer:/mnt/nsd_freesurfer \
    vnmd/deepretinotopy_1.0.18:latest tail -f /dev/null

# Step 1만 실행
docker exec my_eval bash -c "cd /workspace/run_from_freesurfer && ./1_native2fsaverage.sh -s /mnt/nsd_freesurfer -t /mnt/hcp_surface -h lh -i subj01"

# 정리
docker rm -f my_eval
```

## 📚 추가 자료

- 전체 문서: `NSD_EVALUATION_README.md`
- Docker 공식 문서: https://docs.docker.com/
- NVIDIA Docker: https://github.com/NVIDIA/nvidia-docker
- deepRetinotopy 논문: (논문 링크)

## 🤝 기여 및 피드백

문제가 발생하거나 개선 사항이 있으면 이슈를 열어주세요.

## 📄 라이센스

deepRetinotopy 프로젝트와 동일한 라이센스를 따릅니다.
