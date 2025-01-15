#!/bin/bash

#SBATCH --job-name=create_mask
#SBATCH --output=/cluster/CBIO/home/lgortana/deconv-plugin/log/mask_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/deconv-plugin/log/mask_%j.err
#SBATCH --mem 80000
#SBATCH --gres=gpu:P100:1
#SBATCH -p cbio-gpu
#SBATCH --cpus-per-task=4

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate hovernet

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/hovernet/lib:$LD_LIBRARY_PATH

python3 -u hovernet/run_mask.py \
        /cluster/CBIO/data1/lgortana/CytAssist_FFPE_Sagittal_Mouse_Brain/tif/CytAssist_FFPE_Sagittal_Mouse_Brain_tissue_image.tif \
        /cluster/CBIO/data1/lgortana/CytAssist_FFPE_Sagittal_Mouse_Brain/mask/CytAssist_FFPE_Sagittal_Mouse_Brain_tissue_image.png \
        --mask-level 3 \
