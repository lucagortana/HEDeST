#!/bin/bash

#SBATCH --job-name=hovernet
#SBATCH --output=/cluster/CBIO/home/lgortana/hover_net/myrepo/log/segment_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/hover_net/myrepo/log/segment_%j.err
#SBATCH --mem 80000
#SBATCH --gres=gpu:1
#SBATCH -p cbio-gpu
#SBATCH --cpus-per-task=4

echo 'Found a place!'
python run_infer.py --gpu='0,1' --nr_types=6 --type_info_path=type_info.json --model_path=myrepo/pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar wsi --input_dir=myrepo/input/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ --output_dir=myrepo/out/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/new_mask/ --input_mask_dir=myrepo/mask/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_lvl125_1/ --save_mask