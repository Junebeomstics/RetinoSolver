#!/bin/bash
# nvidia-container-toolkit installation script

set -e

echo "Installing nvidia-container-toolkit..."

# 1. Add GPG key
echo "Step 1: Adding GPG key..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 2. Add repository
echo "Step 2: Adding repository..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3. Update package list
echo "Step 3: Updating package list..."
sudo apt-get update

# 4. Install nvidia-container-toolkit
echo "Step 4: Installing nvidia-container-toolkit..."
sudo apt-get install -y nvidia-container-toolkit

# 5. Configure Docker daemon
echo "Step 5: Configuring Docker daemon..."
sudo nvidia-ctk runtime configure --runtime=docker

# 6. Restart Docker
echo "Step 6: Restarting Docker..."
sudo systemctl restart docker

# 7. Verify installation
echo "Step 7: Verifying installation..."
nvidia-container-runtime --version

echo ""
echo "Installation completed!"
echo "Test GPU access with: docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi"

