#!/bin/bash

#SBATCH --job-name=moco-v3
#SBATCH --output=/cluster/CBIO/home/lgortana/HEDeST/log/ssl_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/HEDeST/log/ssl_%j.err
#SBATCH --gres=gpu:P100:1
#SBATCH -p cbio-gpu
#SBATCH --exclude=node005,node009
#SBATCH --cpus-per-task=4

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate CellST

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/CellST/lib:$LD_LIBRARY_PATH

python -u external/mocov3/run_moco.py \
    --image_path /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/image_dict_cellvit_64px_20um.pt \
    --save_path /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer \
    --tag BRCA_cellvit_64px_20um \
    --batch_size_infer 2048 \
    --num_workers 4 \
