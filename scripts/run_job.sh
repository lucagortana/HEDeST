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
    --json_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json \
    --image_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict_32.pt \
    --path_st_adata /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/ \
    --proportions_file /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv \
    --lr 1e-5 \
    --agg_loss mean \
    --alpha 0.5 \
    --epochs 60 \
    --train_size 0.5 \
    --val_size 0.25 \
    --out_dir ../out/new/model_lr1e-5_agglossmean_alpha05_epochs60_se32 \
    --rs 42 \

python3 -u run.py \
    --adata_name CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma \
    --json_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json \
    --image_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict_32.pt \
    --path_st_adata /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/ \
    --proportions_file /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv \
    --lr 1e-5 \
    --agg_loss mean \
    --alpha 0.25 \
    --epochs 60 \
    --train_size 0.5 \
    --val_size 0.25 \
    --out_dir ../out/new/model_lr1e-5_agglossmean_alpha025_epochs60_se32 \
    --rs 42 \

python3 -u run.py \
    --adata_name CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma \
    --json_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json \
    --image_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict_32.pt \
    --path_st_adata /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/ \
    --proportions_file /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv \
    --lr 1e-5 \
    --agg_loss mean \
    --alpha 0 \
    --epochs 60 \
    --train_size 0.5 \
    --val_size 0.25 \
    --out_dir ../out/new/model_lr1e-5_agglossmean_alpha0_epochs60_se32 \
    --rs 42 \

python3 -u run.py \
    --adata_name CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma \
    --json_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json \
    --image_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict_32.pt \
    --path_st_adata /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/ \
    --proportions_file /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv \
    --lr 1e-5 \
    --agg_loss onehot \
    --alpha 0.5 \
    --epochs 60 \
    --train_size 0.5 \
    --val_size 0.25 \
    --out_dir ../out/new/model_lr1e-5_agglossonehot_alpha05_epochs60_se32 \
    --rs 42 \

python3 -u run.py \
    --adata_name CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma \
    --json_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json \
    --image_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict_32.pt \
    --path_st_adata /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/ \
    --proportions_file /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv \
    --lr 1e-5 \
    --agg_loss onehot \
    --alpha 0.25 \
    --epochs 60 \
    --train_size 0.5 \
    --val_size 0.25 \
    --out_dir ../out/new/model_lr1e-5_agglossonehot_alpha025_epochs60_se32 \
    --rs 42 \

python3 -u run.py \
    --adata_name CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma \
    --json_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json \
    --image_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict_32.pt \
    --path_st_adata /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/ \
    --proportions_file /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv \
    --lr 1e-5 \
    --agg_loss onehot \
    --alpha 0 \
    --epochs 60 \
    --train_size 0.5 \
    --val_size 0.25 \
    --out_dir ../out/new/model_lr1e-5_agglossonehot_alpha0_epochs60_se32 \
    --rs 42 \

#

python3 -u run.py \
    --adata_name CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma \
    --json_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json \
    --image_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict_32.pt \
    --path_st_adata /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/ \
    --proportions_file /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv \
    --lr 1e-5 \
    --weights \
    --agg_loss mean \
    --alpha 0.5 \
    --epochs 60 \
    --train_size 0.5 \
    --val_size 0.25 \
    --out_dir ../out/new/model_lr1e-5_agglossmean_weight_alpha05_epochs60_se32 \
    --rs 42 \

python3 -u run.py \
    --adata_name CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma \
    --json_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json \
    --image_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict_32.pt \
    --path_st_adata /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/ \
    --proportions_file /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv \
    --lr 1e-5 \
    --weights \
    --agg_loss mean \
    --alpha 0.25 \
    --epochs 60 \
    --train_size 0.5 \
    --val_size 0.25 \
    --out_dir ../out/new/model_lr1e-5_agglossmean_weight_alpha025_epochs60_se32 \
    --rs 42 \

python3 -u run.py \
    --adata_name CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma \
    --json_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json \
    --image_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict_32.pt \
    --path_st_adata /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/ \
    --proportions_file /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv \
    --lr 1e-5 \
    --weights \
    --agg_loss mean \
    --alpha 0 \
    --epochs 60 \
    --train_size 0.5 \
    --val_size 0.25 \
    --out_dir ../out/new/model_lr1e-5_agglossmean_weight_alpha0_epochs60_se32 \
    --rs 42 \

python3 -u run.py \
    --adata_name CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma \
    --json_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json \
    --image_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict_32.pt \
    --path_st_adata /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/ \
    --proportions_file /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv \
    --lr 1e-5 \
    --weights \
    --agg_loss onehot \
    --alpha 0.5 \
    --epochs 60 \
    --train_size 0.5 \
    --val_size 0.25 \
    --out_dir ../out/new/model_lr1e-5_agglossonehot_weight_alpha05_epochs60_se32 \
    --rs 42 \

python3 -u run.py \
    --adata_name CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma \
    --json_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json \
    --image_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict_32.pt \
    --path_st_adata /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/ \
    --proportions_file /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv \
    --lr 1e-5 \
    --weights \
    --agg_loss onehot \
    --alpha 0.25 \
    --epochs 60 \
    --train_size 0.5 \
    --val_size 0.25 \
    --out_dir ../out/new/model_lr1e-5_agglossonehot_weight_alpha025_epochs60_se32 \
    --rs 42 \

python3 -u run.py \
    --adata_name CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma \
    --json_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json \
    --image_path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/images_dict_32.pt \
    --path_st_adata /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/ \
    --proportions_file /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv \
    --lr 1e-5 \
    --weights \
    --agg_loss onehot \
    --alpha 0 \
    --epochs 60 \
    --train_size 0.5 \
    --val_size 0.25 \
    --out_dir ../out/new/model_lr1e-5_agglossonehot_weight_alpha0_epochs60_se32 \
    --rs 42 \
