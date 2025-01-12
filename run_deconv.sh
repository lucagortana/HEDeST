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

divergence_options=("rot" "kl" "l2" "l1")
reduction_options=("sum" "mean")
alpha_values=(0 0.25 0.5)

for divergence in "${divergence_options[@]}"; do
  for reduction in "${reduction_options[@]}"; do
    for alpha in "${alpha_values[@]}"; do

      out_dir="../models/new/model_probas_div${divergence}_red${reduction}_alpha${alpha}_lr1e-5_weighted"

      python3 -u deconvplugin/run.py \
        CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma \
        /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/seg_json/pannuke_fast_mask_lvl3.json \
        /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/image_dict_64.pt \
        /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/ST/ \
        /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv \
        --lr 1e-5 \
        --weights \
        --agg "proba" \
        --divergence "$divergence" \
        --reduction "$reduction" \
        --alpha "$alpha" \
        --epochs 80 \
        --out_dir "$out_dir" \

    done
  done
done
