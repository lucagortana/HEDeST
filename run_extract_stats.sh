#!/bin/bash

#SBATCH --job-name=extract_stats
#SBATCH --output=/cluster/CBIO/home/lgortana/deconv-plugin/log/extract_stats_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/deconv-plugin/log/extract_stats_%j.err
#SBATCH -p cbio-cpu
#SBATCH --cpus-per-task=8

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate plugin-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/plugin-env/lib:$LD_LIBRARY_PATH

FOLDER=$1
GT_CSV=$2

python3 -u deconvplugin/compute_stats.py "$FOLDER" "$GT_CSV"
