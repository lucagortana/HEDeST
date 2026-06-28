#!/bin/bash

#SBATCH --job-name=cellvit
#SBATCH --output=/cluster/CBIO/home/lgortana/HEDeST/log/cellvit_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/HEDeST/log/cellvit_%j.err
#SBATCH --gres=gpu:A40:1
#SBATCH -p cbio-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate cellvit-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/cellvit-env/lib:$LD_LIBRARY_PATH

python -u external/CellViT/cellvit/detect_cells.py \
    --model SAM \
    --nuclei_taxonomy pannuke \
    --batch_size 8 \
    --outdir /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/cellvit/ \
    --export_cells \
    --save_geojson \
    --size_px 64 \
    --size_um 20 \
    --image_dict_path /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/image_dict_cellvit_64px_20um.pt \
    process_wsi \
    --wsi_path /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/pyr_tif/Visium_FFPE_Human_Breast_Cancer_image.tif \
    --wsi_mpp 0.345 \
    --wsi_magnification 40
