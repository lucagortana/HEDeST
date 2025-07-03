#!/bin/bash

#SBATCH --job-name=HEDeST
#SBATCH --output=/cluster/CBIO/home/lgortana/deconv-plugin/log/gridsearch_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/deconv-plugin/log/gridsearch_%j.err
#SBATCH --gres=gpu:1
#SBATCH -p cbio-gpu
#SBATCH --exclude=node005,node006,node009
#SBATCH --cpus-per-task=8

echo "Found a place!"

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate plugin-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/plugin-env/lib:$LD_LIBRARY_PATH

IMAGE_DICT=$1
SIM_CSV=$2
SPOT_DICT=$3
SPOT_DICT_GLOBAL=$4
shift 4

EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
    EXTRA_ARGS+=("$1")
    shift
done

python3 -u hedest/gridsearch.py \
    "$IMAGE_DICT" \
    "$SIM_CSV" \
    "$SPOT_DICT" \
    "$SPOT_DICT_GLOBAL" \
    "${EXTRA_ARGS[@]}"
