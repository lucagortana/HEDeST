#!/bin/bash

#SBATCH --job-name=HEDeST
#SBATCH --output=/cluster/CBIO/home/lgortana/HEDeST/log/hedest_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/HEDeST/log/hedest_%j.err
#SBATCH --gres=gpu:1
#SBATCH -p cbio-gpu
#SBATCH --cpus-per-task=8

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate plugin-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/plugin-env/lib:$LD_LIBRARY_PATH

for seed in 42 43 44 45 46; do
  python3 -u hedest/main.py \
    /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/moco_embed_moco-TENXHB-rn50.pt \
    /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/props/DestVI_2000_hvg_squash_06_02_no_endo_Visium_FFPE_Human_Breast_Cancer_prop.csv \
    /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/seg_json/pannuke_fast_mask_lvl3.json \
    /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/ST/ \
    Visium_FFPE_Human_Breast_Cancer \
    --spot-dict-file /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/spot_dict.json \
    --model-name quick \
    --batch-size 64 \
    --lr 3e-4 \
    --divergence l2 \
    --alpha 0 \
    --beta 0.0 \
    --epochs 100 \
    --train-size 0.8 \
    --val-size 0.1 \
    --out-dir models/Breast_cancer_FFPE/DestVI_2000_hvg_squash_06_02_no_endo/model_quick_alpha_0.0_lr_0.0003_divergence_l2_beta_0.0_seed_${seed} \
    --tb-dir models/TBruns \
    --rs $seed
done
