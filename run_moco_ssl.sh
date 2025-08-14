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
    --image_path /cluster/CBIO/data1/lgortana/Visium_Human_Breast_Cancer/image_dict.pt \
    --save_path /cluster/CBIO/data1/lgortana/Visium_Human_Breast_Cancer \
    --tag moco-TENXHB2-rn50 \
    --json_path /cluster/CBIO/data1/lgortana/Visium_Human_Breast_Cancer/seg_json/pannuke_fast_mask_lvl3.json \
    --batch_size_infer 2048 \
    --num_workers 4 \
