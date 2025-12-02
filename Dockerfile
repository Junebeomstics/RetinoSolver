# Use official PyTorch 2.x image with CUDA 12.1 support
# For CPU version, use: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FORCE_CUDA="1" \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch and torchvision (already in base image, but ensure version)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0

# Verify torch installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Install torch-geometric with all extensions
# torch-geometric[full] will attempt to install extensions automatically
# If that fails, we'll install them manually with proper CUDA settings
RUN export CUDA_HOME=/usr/local/cuda && \
    export PATH=$CUDA_HOME/bin:$PATH && \
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH && \
    pip install --no-cache-dir torch-geometric[full] || \
    (pip install --no-cache-dir torch-geometric && \
     FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
     pip install --no-cache-dir --no-build-isolation \
     torch-scatter torch-sparse torch-cluster torch-spline-conv)

# Copy requirements file
COPY requirements.txt .

# Install remaining requirements (excluding torch packages to avoid conflicts)
# Filter out torch-related packages from requirements.txt
RUN grep -v "^torch" requirements.txt > /tmp/requirements_filtered.txt && \
    pip install --no-cache-dir --ignore-installed -r /tmp/requirements_filtered.txt || \
    echo "Warning: Some packages may have failed to install"

# Copy project files
COPY . .

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Verify installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')" || true

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Set default command
CMD ["/bin/bash"]
