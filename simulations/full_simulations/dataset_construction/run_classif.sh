#!/bin/bash

#SBATCH --job-name=classif
#SBATCH --output=/cluster/CBIO/home/lgortana/simulation-embeddings/log/classif_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/simulation-embeddings/log/classif_%j.err
#SBATCH -p cbio-cpu
#SBATCH --cpus-per-task=4

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate xgb_env

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/xgb_env/lib:$LD_LIBRARY_PATH

#xgboost
python cell_classifier.py \
    --features_file /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/moco_embed_moco-XENHBrep1-rn50.pt \
    --ground_truth_file /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/sim_Xenium_FFPE_Human_Breast_Cancer_Rep1_gt.csv \
    --out_path /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/classif \
    --classifier xgboost \
    --balance \
    --num_seeds 5

python cell_classifier.py \
    --features_file /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/moco_embed_moco-XENHBrep1-rn50.pt \
    --ground_truth_file /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/sim_Xenium_FFPE_Human_Breast_Cancer_Rep1_gt.csv \
    --out_path /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/classif \
    --classifier xgboost \
    --num_seeds 5

#LogReg
python cell_classifier.py \
    --features_file /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/moco_embed_moco-XENHBrep1-rn50.pt \
    --ground_truth_file /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/sim_Xenium_FFPE_Human_Breast_Cancer_Rep1_gt.csv \
    --out_path /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/classif \
    --classifier logistic \
    --balance \
    --penalty l1 \
    --C 1.0 \
    --num_seeds 5

python cell_classifier.py \
    --features_file /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/moco_embed_moco-XENHBrep1-rn50.pt \
    --ground_truth_file /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/sim_Xenium_FFPE_Human_Breast_Cancer_Rep1_gt.csv \
    --out_path /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/classif \
    --classifier logistic \
    --penalty l1 \
    --C 1.0 \
    --num_seeds 5

python cell_classifier.py \
    --features_file /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/moco_embed_moco-XENHBrep1-rn50.pt \
    --ground_truth_file /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/sim_Xenium_FFPE_Human_Breast_Cancer_Rep1_gt.csv \
    --out_path /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/classif \
    --classifier logistic \
    --balance \
    --penalty l2 \
    --C 1.0 \
    --num_seeds 5
