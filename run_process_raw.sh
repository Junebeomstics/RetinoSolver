#!/bin/bash

# Script to run process_raw.py in Docker container

# Get absolute path of script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Docker settings
DOCKER_IMAGE=${DOCKER_IMAGE:-"vnmd/deepretinotopy_1.0.18:latest"}
USE_GPU=${USE_GPU:-"true"}

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Verify paths exist
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Error: Project root directory does not exist: $PROJECT_ROOT"
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/Retinotopy/data" ]; then
    echo "Warning: Data directory does not exist: $PROJECT_ROOT/Retinotopy/data"
    echo "It will be created automatically if needed."
fi

echo "Project root: $PROJECT_ROOT"
echo "Docker image: $DOCKER_IMAGE"
echo ""

# Build Docker command using --mount (more reliable than -v)
DOCKER_ARGS=("run" "--rm")

if [ "$USE_GPU" = "true" ]; then
    DOCKER_ARGS+=("--gpus" "all")
fi

# Use --mount instead of -v for better path handling
DOCKER_ARGS+=("--mount" "type=bind,source=$PROJECT_ROOT,target=/workspace")
DOCKER_ARGS+=("--mount" "type=bind,source=$PROJECT_ROOT/Retinotopy/data,target=/workspace/Retinotopy/data")
DOCKER_ARGS+=("-w" "/workspace")
DOCKER_ARGS+=("$DOCKER_IMAGE")
DOCKER_ARGS+=("python" "process_raw.py")

# Pass all script arguments to process_raw.py
if [ $# -gt 0 ]; then
    DOCKER_ARGS+=("$@")
fi

# Execute the command
echo "Executing Docker command..."
echo ""
docker "${DOCKER_ARGS[@]}"
