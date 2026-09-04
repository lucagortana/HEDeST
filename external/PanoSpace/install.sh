#!/bin/bash
# PanoSpace One-Click Installation Script
# This script installs PanoSpace with all dependencies

set -e  # Exit on error

echo "=========================================="
echo "PanoSpace Installation Script"
echo "=========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found"
echo ""

# Auto-detect GPU
echo "Detecting GPU..."
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected!"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
    echo ""
    echo "GPU-accelerated version is recommended for better performance."
    echo ""
    read -p "Install GPU-enabled version? (Y/n): " install_gpu
    install_gpu=${install_gpu:-Y}  # Default to Yes

    if [[ "$install_gpu" =~ ^[Yy]$ ]]; then
        env_file="environment-gpu.yml"
        install_gpu_pytorch=true
        echo ""
        echo "Installing GPU-enabled version (CUDA 12.1)..."
        echo "Note: PyTorch with CUDA will be installed separately via pip"
    else
        env_file="environment.yml"
        install_gpu_pytorch=false
        echo ""
        echo "Installing CPU-only version..."
        echo "Note: PyTorch will be installed separately via pip"
    fi
else
    echo "No NVIDIA GPU detected or nvidia-smi not available."
    echo "Installing CPU-only version..."
    env_file="environment.yml"
    install_gpu_pytorch=false
fi

# Check if environment file exists
if [ ! -f "$env_file" ]; then
    echo "Error: $env_file not found!"
    exit 1
fi

# Create conda environment
echo ""
echo "Creating conda environment (this may take 10-20 minutes)..."
echo "Please wait..."
conda env create -f "$env_file"

# Activate environment
echo ""
echo "Activating PanoSpace environment..."
eval "$(conda shell.bash hook)"
conda activate PanoSpace

# Install PyTorch via pip (separate from conda to ensure proper CUDA support)
echo ""
echo "Installing PyTorch via pip..."
echo "Note: Removing any existing PyTorch installation first..."
pip uninstall -y torch torchvision 2>/dev/null || true

if [ "$install_gpu_pytorch" = true ]; then
    echo "Installing PyTorch with CUDA support (this may take 5-10 minutes)..."
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  "torch>=2.1" "torchvision>=0.15"
else
    echo "Installing CPU-only PyTorch..."
    pip install "torch>=2.1" "torchvision>=0.15"
fi

# Install PanoSpace
echo ""
echo "Installing PanoSpace package..."
pip install .

# Verify installation
echo ""
echo "Verifying installation..."
if [ -f "scripts/verify_install.py" ]; then
    python scripts/verify_install.py
else
    python -c "import panospace as ps; print('PanoSpace imported successfully!')"
fi

# Test GPU if GPU version was installed
if [ "$env_file" = "environment-gpu.yml" ]; then
    echo ""
    echo "Testing GPU support..."
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To use PanoSpace:"
echo "  1. Activate the environment:"
echo "     conda activate PanoSpace"
echo ""
echo "  2. Start Python and import PanoSpace:"
echo "     python"
echo "     >>> import panospace as ps"
echo ""
echo "  3. Check out the Quick Start guide in README.md"
echo ""
echo "For GPU users:"
if [ "$env_file" = "environment-gpu.yml" ]; then
    echo "  ✓ GPU support is enabled"
    echo "  Your GPU will be used automatically for cell detection"
else
    echo "  Note: You installed the CPU-only version"
    echo "  To enable GPU support later, see README.md"
fi
echo ""
