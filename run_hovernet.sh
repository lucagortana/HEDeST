#!/bin/bash

#SBATCH --job-name=hovernet
#SBATCH --output=/cluster/CBIO/home/lgortana/deconv-plugin/log/segment_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/deconv-plugin/log/segment_%j.err
#SBATCH --mem 80000
#SBATCH --gres=gpu:P100:1
#SBATCH -p cbio-gpu
#SBATCH --cpus-per-task=4

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate hovernet

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/hovernet/lib:$LD_LIBRARY_PATH

python ../hover_net/run_infer.py \
    --gpu='0,1' \
    --nr_types=0 \
    --type_info_path=../hover_net/type_info.json \
    --model_mode=original \
    --model_path=data/pretrained/seg/hovernet_original_consep_notype_tf2pytorch.tar \
    wsi \
    --input_dir=data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/tif/ \
    --output_dir=out/results_hovernet/Ovarian_nolabel_consep/ \
    --input_mask_dir=data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/mask/lvl3/ \

python ../hover_net/run_infer.py \
    --gpu='0,1' \
    --nr_types=0 \
    --type_info_path=../hover_net/type_info.json \
    --model_mode=original \
    --model_path=data/pretrained/seg/hovernet_original_cpm17_notype_tf2pytorch.tar \
    wsi \
    --input_dir=data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/tif/ \
    --output_dir=out/results_hovernet/Ovarian_nolabel_cpm17/ \
    --input_mask_dir=data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/mask/lvl3/ \

python ../hover_net/run_infer.py \
    --gpu='0,1' \
    --nr_types=0 \
    --type_info_path=../hover_net/type_info.json \
    --model_mode=original \
    --model_path=data/pretrained/seg/hovernet_original_kumar_notype_tf2pytorch.tar \
    wsi \
    --input_dir=data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/tif/ \
    --output_dir=out/results_hovernet/Ovarian_nolabel_kumar/ \
    --input_mask_dir=data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/mask/lvl3/ \
