#!/bin/bash

#SBATCH --job-name=extract_emb
#SBATCH --output=/cluster/CBIO/home/lgortana/simulation-embeddings/log/extract_emb_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/simulation-embeddings/log/extract_emb_%j.err
#SBATCH --gres=gpu:P100:1
#SBATCH -p cbio-gpu
#SBATCH --exclude=node009
#SBATCH --cpus-per-task=4

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate plugin-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/plugin-env/lib:$LD_LIBRARY_PATH

python extract_features.py \
    --model_name 'h-optimus-0' \
    --image_dict_path /cluster/CBIO/data1/lgortana/Xenium_V1_humanLung_Cancer_FFPE/image_dict_64.pt \
    --out_dir /cluster/CBIO/data1/lgortana/Xenium_V1_humanLung_Cancer_FFPE/emb_hoptimus0.pt \
    --batch_size 64
