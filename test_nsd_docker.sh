#!/bin/bash
#
# Quick test script for NSD evaluation with Docker
# Tests a single prediction (eccentricity, left hemisphere) to verify setup
#

set -e

echo "==============================================="
echo "Quick Test: NSD Evaluation with Docker"
echo "==============================================="
echo ""
echo "This script will test the evaluation pipeline with:"
echo "  - Subject: subj01"
echo "  - Hemisphere: lh"
echo "  - Prediction: eccentricity"
echo "  - Model: baseline"
echo "  - Docker: enabled"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run single evaluation
bash "$SCRIPT_DIR/run_nsd_inference_docker.sh" \
    -s subj01 \
    -h lh \
    -p eccentricity \
    -m baseline \
    -y False \
    -r 0.1 \
    -o ./nsd_evaluation_test_docker \
    -g true

if [ $? -eq 0 ]; then
    echo ""
    echo "==============================================="
    echo "✓ Test Completed Successfully!"
    echo "==============================================="
    echo ""
    echo "Check the results in: ./nsd_evaluation_test_docker/"
    echo ""
    echo "To run full evaluation with Docker, use:"
    echo "  ./run_nsd_full_evaluation_docker.sh -s subj01"
    echo ""
    echo "Docker container status:"
    docker ps --filter name=deepretinotopy_nsd_eval --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "To stop the Docker container:"
    echo "  docker stop deepretinotopy_nsd_eval"
    echo ""
else
    echo ""
    echo "==============================================="
    echo "✗ Test Failed"
    echo "==============================================="
    echo ""
    echo "Please check the error messages above."
    echo "Common issues:"
    echo "  1. Docker not installed - install Docker first"
    echo "  2. Docker image not found - check DOCKER_IMAGE variable"
    echo "  3. NSD data not found - check path in script"
    echo "  4. Model checkpoints missing - check Models/checkpoints/"
    echo "  5. HCP surfaces missing - check surface/ directory"
    echo "  6. GPU not available - try with: -g false"
    echo ""
    exit 1
fi
