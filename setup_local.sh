#!/bin/bash
# Setup local conda environment for E-Textile Validation GUI
# Run this on your LOCAL Linux machine after mounting Triton via SSHFS:
#   sshfs zhangy47@triton.aalto.fi:/scratch/elec/t412-aiwo t412-aiwo
#   cd t412-aiwo/yaozhang/etextile_validation_gui
#   bash setup_local.sh

set -e

ENV_NAME="etextile"

echo "=== E-Textile Validation GUI — Local Setup ==="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create environment
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists. Updating..."
    conda activate ${ENV_NAME}
    pip install -r requirements.txt
    pip install PyQt6
else
    echo "Creating conda environment '${ENV_NAME}'..."
    conda create -n ${ENV_NAME} python=3.10 -y
    conda activate ${ENV_NAME}
    pip install -r requirements.txt
    pip install PyQt6
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the GUI:"
echo "  conda activate ${ENV_NAME}"
echo "  python app.py"
echo ""
