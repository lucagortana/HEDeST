#!/bin/bash

#SBATCH --job-name=create_mask
#SBATCH --output=/cluster/CBIO/home/lgortana/HEDeST/log/mask_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/HEDeST/log/mask_%j.err
#SBATCH -p cbio-cpu
#SBATCH --cpus-per-task=4

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate hovernet

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/hovernet/lib:$LD_LIBRARY_PATH

python3 -u external/hovernet/run_mask.py \
        /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/pyr_tif/Visium_FFPE_Human_Breast_Cancer_image.tif \
        /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/mask/lvl3/Visium_FFPE_Human_Breast_Cancer_image.png \
        --mask-level 3 \
