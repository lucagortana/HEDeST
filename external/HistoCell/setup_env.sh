#!/bin/bash
# ---------------------------------------------------------------------------
# Create the conda environment used by run_histocell.py.
#
#   bash external/HistoCell/setup_env.sh [ENV_NAME]
#
# Upstream ships requirements.txt pinned to python 3.7 / torch 1.12.1+cu113.
# That stack cannot read the .h5ad and pyramidal .tiff files this benchmark
# uses, so the environment below is the same software one major version newer.
# Nothing in the *method* depends on the version: HistoCell is plain
# torch.nn modules, and `resnet18(pretrained=True)` resolves to the very same
# ImageNet checkpoint (resnet18-f37072fd.pth, IMAGENET1K_V1) on torchvision
# 0.13 and 0.20 alike, so the encoder starts from identical weights.
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="${1:-histocell-env}"
PY_VERSION=3.10

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup_env] '$ENV_NAME' already exists -- reusing it."
else
    echo "[setup_env] creating '$ENV_NAME' (python $PY_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi

conda activate "$ENV_NAME"

echo "[setup_env] installing torch 2.5.1 + torchvision 0.20.1 (cu121)"
pip install --quiet torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

echo "[setup_env] installing the rest"
pip install --quiet \
    "numpy<2" \
    einops \
    yacs \
    opencv-python-headless \
    tifffile \
    imagecodecs \
    anndata \
    h5py \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    tqdm \
    pillow

python - <<'PY'
import torch, torchvision, cv2, tifffile, anndata, einops, yacs
print(f"[setup_env] torch {torch.__version__}  torchvision {torchvision.__version__}")
print(f"[setup_env] cuda available: {torch.cuda.is_available()}")
print("[setup_env] OK")
PY

echo "[setup_env] done -- activate with: conda activate $ENV_NAME"
