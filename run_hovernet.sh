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

python hovernet/run_infer.py \
    --gpu='0,1' \
    --nr_types=6 \
    --type_info_path=hovernet/type_info.json \
    --model_mode=fast \
    --model_path=/cluster/CBIO/data1/lgortana/pretrained/seg_classif/hovernet_fast_pannuke_type_tf2pytorch.tar \
    wsi \
    --input_dir=/cluster/CBIO/data1/lgortana/CytAssist_FFPE_Sagittal_Mouse_Brain/tif/ \
    --output_dir=/cluster/CBIO/data1/lgortana/CytAssist_FFPE_Sagittal_Mouse_Brain/seg_json/ \
    --input_mask_dir=data/CytAssist_FFPE_Sagittal_Mouse_Brain/mask/lvl3/ \
