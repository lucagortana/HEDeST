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

for seed in 42; do
  python3 -u hedest/main.py \
    /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/sim/4_moco_clusters_30spots_balanced_5mean_5var_emb_dict.pt \
    /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/sim/4_moco_clusters_30spots_balanced_5mean_5var_prop.csv \
    --spot-dict-file /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/sim/4_moco_clusters_30spots_balanced_5mean_5var_spot_dict.json \
    --model-name quick \
    --batch-size 64 \
    --lr 3e-4 \
    --divergence l2 \
    --alpha 0 \
    --beta 0.0 \
    --epochs 100 \
    --train-size 0.8 \
    --val-size 0.1 \
    --out-dir models/quick-sim-30spots/balanced/model_quick_alpha_0.0_lr_0.0003_divergence_l2_beta_0.0_seed_${seed} \
    --tb-dir models/TBruns \
    --rs $seed
done

for seed in 42; do
  python3 -u hedest/main.py \
    /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/sim/4_moco_clusters_30spots_reallike_5mean_5var_emb_dict.pt \
    /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/sim/4_moco_clusters_30spots_reallike_5mean_5var_prop.csv \
    --spot-dict-file /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/sim/4_moco_clusters_30spots_reallike_5mean_5var_spot_dict.json \
    --model-name quick \
    --batch-size 64 \
    --lr 3e-4 \
    --divergence l2 \
    --alpha 0 \
    --beta 0.0 \
    --epochs 100 \
    --train-size 0.8 \
    --val-size 0.1 \
    --out-dir models/quick-sim-30spots/realistic/model_quick_alpha_0.0_lr_0.0003_divergence_l2_beta_0.0_seed_${seed} \
    --tb-dir models/TBruns \
    --rs $seed
done
