#!/bin/bash
# nvidia-container-toolkit 설치 스크립트

set -e

echo "Installing nvidia-container-toolkit..."

# 1. GPG key 추가
echo "Step 1: Adding GPG key..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 2. Repository 추가
echo "Step 2: Adding repository..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3. 패키지 목록 업데이트
echo "Step 3: Updating package list..."
sudo apt-get update

# 4. nvidia-container-toolkit 설치
echo "Step 4: Installing nvidia-container-toolkit..."
sudo apt-get install -y nvidia-container-toolkit

# 5. Docker daemon 설정
echo "Step 5: Configuring Docker daemon..."
sudo nvidia-ctk runtime configure --runtime=docker

# 6. Docker 재시작
echo "Step 6: Restarting Docker..."
sudo systemctl restart docker

# 7. 설치 확인
echo "Step 7: Verifying installation..."
nvidia-container-runtime --version

echo ""
echo "Installation completed!"
echo "Test GPU access with: docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi"

