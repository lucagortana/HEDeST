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

python3 -u sim.py \
    --data_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/sim \
    --gt_filename 4_moco_clusters_30spots_balanced_5mean_5var_gt.csv \
    --spot_dict_filename 4_moco_clusters_30spots_balanced_5mean_5var_spot_dict.json \
    --embeddings_filename 4_moco_clusters_30spots_balanced_5mean_5var_emb_dict.pt \
    --n_iter 2 \
    --output_xlsx ../results/mhast/4_moco_clusters_30spots_balanced_5mean_5var_mhast.xlsx
