# Docker 사용 가이드

이 프로젝트는 PyTorch 0.4.1을 실행하기 위한 Docker 환경을 제공합니다.

## 사전 요구사항

- Docker 설치 (버전 20.10 이상 권장)
- Docker Compose 설치 (버전 1.29 이상 권장)

## Docker 설치 확인

```bash
docker --version
docker-compose --version
```

## Docker 이미지 빌드 및 실행

### 방법 1: Docker Compose 사용 (권장)

```bash
# 이미지 빌드 및 컨테이너 시작
docker-compose up -d

# 컨테이너에 접속
docker-compose exec deepretinotopy bash

# 컨테이너 중지
docker-compose down
```

### 방법 2: Docker 명령어 직접 사용

```bash
# 이미지 빌드
docker build -t deepretinotopy:pytorch0.4.1 .

# 컨테이너 실행 및 접속
docker run -it --name deepretinotopy \
  -v $(pwd):/workspace \
  -v $(pwd)/Retinotopy/data:/workspace/Retinotopy/data \
  deepretinotopy:pytorch0.4.1 bash

# 컨테이너 중지 및 제거
docker stop deepretinotopy
docker rm deepretinotopy
```

## GPU 지원 (NVIDIA)

GPU를 사용하려면:

1. NVIDIA Docker 설치 확인:
```bash
nvidia-docker --version
```

2. `docker-compose.yml` 파일에서 GPU 관련 주석을 해제하고 수정

3. Docker Compose로 실행:
```bash
docker-compose up -d
```

또는 Docker 명령어로:
```bash
docker run -it --gpus all --name deepretinotopy \
  -v $(pwd):/workspace \
  -v $(pwd)/Retinotopy/data:/workspace/Retinotopy/data \
  deepretinotopy:pytorch0.4.1 bash
```

## 컨테이너 내에서 작업

컨테이너에 접속한 후:

```bash
# Python 버전 확인
python --version

# PyTorch 버전 확인
python -c "import torch; print(torch.__version__)"

# 프로젝트 코드 실행
python Models/deepRetinotopy_ecc_LH.py
```

## 문제 해결

### 컨테이너가 시작되지 않는 경우
```bash
# 로그 확인
docker-compose logs

# 컨테이너 재빌드
docker-compose build --no-cache
```

### 의존성 설치 오류
```bash
# 컨테이너 내에서 수동 설치
docker-compose exec deepretinotopy pip install <package_name>
```

### 볼륨 마운트 문제
- 호스트와 컨테이너 간 파일 권한 확인
- 절대 경로 사용 권장







