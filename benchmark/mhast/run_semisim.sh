#!/bin/bash

#SBATCH --job-name=mhast_classif
#SBATCH --output=/cluster/CBIO/home/lgortana/deconv-plugin/log/mhast_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/deconv-plugin/log/mhast_%j.err
#SBATCH -p cbio-cpu
#SBATCH --cpus-per-task=8

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate mhast

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/mhast/lib:$LD_LIBRARY_PATH

python3 semisim.py \
    --data_path /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim \
    --gt_filename sim_Xenium_FFPE_Human_Breast_Cancer_Rep1_gt.csv \
    --spot_dict_filename spot_dict_small.json \
    --embeddings_filename moco_embed_moco-XENHBrep1-rn50.pt \
    --n_iter 8 \
    --output_csv mhast_results.csv
