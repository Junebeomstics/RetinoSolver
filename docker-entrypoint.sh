#!/bin/bash
# Docker entrypoint script to ensure CUDA extensions are built

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1

# Check if torch-spline-conv is installed
if ! python -c "import torch_spline_conv" 2>/dev/null; then
    echo "torch-spline-conv not found. Installing now..."
    
    # Install torch-geometric extensions
    pip install --no-cache-dir --no-build-isolation \
        torch-scatter \
        torch-sparse \
        torch-cluster \
        torch-spline-conv 2>&1 | tee /tmp/cuda_build.log || \
    echo "Warning: Some extensions may have failed to install. Check /tmp/cuda_build.log"
    
    # Verify installation
    if python -c "import torch_spline_conv" 2>/dev/null; then
        echo "torch-spline-conv installed successfully!"
    else
        echo "Warning: torch-spline-conv installation may have failed."
    fi
fi

# Execute the command
exec "$@"
