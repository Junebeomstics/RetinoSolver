#!/bin/bash
# GPU를 사용하여 Docker 컨테이너 실행 스크립트

# 기존 컨테이너가 있으면 제거
docker rm -f deepretinotopy 2>/dev/null

# GPU를 사용하여 컨테이너 실행
docker run -it --gpus all \
  --name deepretinotopy \
  -v $(pwd):/workspace \
  -v $(pwd)/Retinotopy/data:/workspace/Retinotopy/data \
  -w /workspace \
  deepretinotopy:pytorch0.4.1 \
  bash

