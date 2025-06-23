#!/bin/bash

#SBATCH --job-name=histocell
#SBATCH --output=/cluster/CBIO/home/lgortana/deconv-plugin/log/histocell_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/deconv-plugin/log/histocell_%j.err
#SBATCH --gres=gpu:1
#SBATCH -p cbio-gpu
#SBATCH --exclude=node005,node006,node009
#SBATCH --cpus-per-task=8

echo "Found a place!"

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate histocell

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/histocell/lib:$LD_LIBRARY_PATH

for seed in {0..9}; do
  python3 -u infer.py \
    --seed $seed \
    --epoch 40 \
    --tissue BRCA \
    --deconv Xenium \
    --origin Rep1_256 \
    --prefix Rep1_256 \
    --k_class 7 \
    --tissue_compartment /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/histocell/BRCA/tissue_compartment_BRCA.json \
    --omit_gt
done
