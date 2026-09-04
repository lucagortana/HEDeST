#!/bin/bash
# ---------------------------------------------------------------------------
# PanoSpace environment setup for the HEDeST benchmark (CBIO cluster).
#
# Replaces the upstream interactive `install.sh` with a non-interactive,
# pip-driven install that matches the other envs of this repo
# (plugin-env, cellvit-env, lemon-env, ...).
#
#   bash external/PanoSpace/setup_env.sh [env_name]
#
# Default env name: panospace-env
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="${1:-panospace-env}"
PY_VER="3.11"
CUDA_INDEX="https://download.pytorch.org/whl/cu121"

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup_env] env '$ENV_NAME' already exists -- reusing it"
else
    echo "[setup_env] creating conda env '$ENV_NAME' (python $PY_VER)"
    conda create -y -n "$ENV_NAME" "python=$PY_VER" pip
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# --- 1. core: everything the annotation pipeline needs -----------------------
echo "[setup_env] installing core dependencies"
python -m pip install \
    "numpy<2" "pandas>=2.0" "scipy>=1.11" "scikit-learn>=1.3" "scikit-image>=0.22" \
    "matplotlib>=3.7" "pillow>=10" "opencv-python-headless>=4.8" "tqdm>=4.65" \
    "requests>=2.31" "shapely>=2.0" "einops>=0.6" "tifffile>=2024.1" "imagecodecs" \
    "h5py" "hdf5plugin" "anndata>=0.10" "scanpy>=1.10" "python-igraph>=0.10" "leidenalg>=0.9"

echo "[setup_env] installing torch (cu121)"
python -m pip install --extra-index-url "$CUDA_INDEX" "torch>=2.1,<2.6" "torchvision>=0.16"

echo "[setup_env] installing DL / OT / solver stack"
python -m pip install \
    "pytorch-lightning>=2.1" "lightning>=2.1" "transformers>=4.33" \
    "pot>=0.9" "pyscipopt" "ortools>=9.7"

# --- 2. deconvolution backends ---------------------------------------------
# RCTD + spatialDWLS need qpsolvers (+ ray for RCTD's parallel loop).
echo "[setup_env] installing RCTD / spatialDWLS dependencies"
# osqp is the QP backend both RCTD and spatialDWLS ask qpsolvers for.
python -m pip install "qpsolvers>=4.0" "osqp" "quadprog" "ray>=2.7"

# cell2location is vendored inside panospace and needs the scvi-tools stack.
# It is the heaviest / most fragile dependency, so a failure here is tolerated:
# the RCTD + spatialDWLS backends (and --proportions) keep working without it.
echo "[setup_env] installing cell2location stack (scvi-tools) -- optional"
python -m pip install "scvi-tools>=1.1,<1.3" "pyro-ppl>=1.8" \
    || echo "[setup_env] WARNING: scvi-tools install failed -- the 'cell2location' backend will be unavailable"

# --- 3. panospace itself ----------------------------------------------------
echo "[setup_env] installing panospace (editable)"
python -m pip install -e "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" --no-deps

echo "[setup_env] done. Verify with:"
echo "    conda activate $ENV_NAME && python external/PanoSpace/run_panospace.py --help"
