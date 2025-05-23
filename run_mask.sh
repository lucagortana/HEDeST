#!/bin/bash

#SBATCH --job-name=create_mask
#SBATCH --output=/cluster/CBIO/home/lgortana/deconv-plugin/log/mask_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/deconv-plugin/log/mask_%j.err
#SBATCH -p cbio-cpu
#SBATCH --cpus-per-task=4

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate hovernet

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/hovernet/lib:$LD_LIBRARY_PATH

python3 -u hovernet/run_mask.py \
        /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/pyr_tif/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif \
        /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/mask/lvl3_pyr/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.png \
        --mask-level 3 \
