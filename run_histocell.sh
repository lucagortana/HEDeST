#!/bin/bash

#SBATCH --job-name=histocell
#SBATCH --output=/cluster/CBIO/home/lgortana/HEDeST/log/histocell_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/HEDeST/log/histocell_%j.err
#SBATCH --gres=gpu:1
#SBATCH -p cbio-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclude=node005,node009

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate histocell-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/histocell-env/lib:${LD_LIBRARY_PATH:-}

# HistoCell starts from an ImageNet ResNet-18; cache it once instead of
# re-downloading on every node.
export TORCH_HOME=/cluster/CBIO/home/lgortana/.histocell_cache/torch

# ---------------------------------------------------------------------------
# Defaults, used only when the script is submitted without arguments.
# ---------------------------------------------------------------------------
SAMPLE=lung_s3
LEVEL=level2
BENCH=/cluster/CBIO/data1/lgortana/STHELAR/bench_data

DEFAULT_ARGS=(
    --he           "$BENCH/$SAMPLE/he.tiff"
    --st           "$BENCH/$SAMPLE/pseudovisium.h5ad"
    --seg-dict     "$BENCH/$SAMPLE/hovernet.json"
    --proportions  "$BENCH/$SAMPLE/sim/$LEVEL/proportions.csv"
    --output       "/cluster/CBIO/data1/lgortana/STHELAR/histocell/$SAMPLE/$LEVEL"
    --sample-name  "${SAMPLE}_${LEVEL}"
    --num-workers  "${SLURM_CPUS_PER_TASK:-6}"
)

if [ "$#" -gt 0 ]; then
    ARGS=("$@")
else
    ARGS=("${DEFAULT_ARGS[@]}")
fi

python3 -u external/HistoCell/run_histocell.py "${ARGS[@]}"
