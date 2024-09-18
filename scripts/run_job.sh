#!/bin/bash

#SBATCH --job-name=plugin
#SBATCH --output=/cluster/CBIO/home/lgortana/deconv-plugin/log/classif_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/deconv-plugin/log/classif_%j.err
#SBATCH --mem 80000
#SBATCH --gres=gpu:1
#SBATCH -p cbio-gpu
#SBATCH --cpus-per-task=8

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate plugin-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/plugin-env/lib:$LD_LIBRARY_PATH

python3 -u run.py \
    --adata_name CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma \
    --json_path ../data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json \
    --path_st_adata ../data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/ \
    --proportions_file ../data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv \
    --batch_size 8 \
    --lr 1e-5 \
    --agg_loss mean \
    --alpha 0.5 \
    --epochs 50 \
    --train_size 0.5 \
    --val_size 0.25 \
    --out_dir ../out/model_bs8_lr1e-5_agglossmean_alpha05_epochs50 \
    --rs 42 \
    --image_dict_path ../data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict.pt
