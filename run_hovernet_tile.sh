#!/bin/bash

#SBATCH --job-name=hovernet
#SBATCH --output=/cluster/CBIO/home/lgortana/HEDeST/log/segment_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/HEDeST/log/segment_%j.err
#SBATCH --gres=gpu:1
#SBATCH -p cbio-gpu
#SBATCH --cpus-per-task=4

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate hovernet

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/hovernet/lib:$LD_LIBRARY_PATH

python external/hovernet/run_infer.py \
    --gpu='1' \
    --nr_types=6 \
    --type_info_path=external/hovernet/type_info.json \
    --batch_size=16 \
    --model_mode=fast \
    --model_path=/cluster/CBIO/data1/lgortana/pretrained/seg_classif/hovernet_fast_pannuke_type_tf2pytorch.tar \
    --nr_inference_workers=8 \
    --nr_post_proc_workers=16 \
    tile \
    --input_dir=/cluster/CBIO/data1/lgortana/Xenium_V1_humanLung_Cancer_FFPE/histocell/LCA/tiles/LCA_256 \
    --output_dir=/cluster/CBIO/data1/lgortana/Xenium_V1_humanLung_Cancer_FFPE/histocell/LCA/seg/LCA_256 \
    --mem_usage=0.1 \
    --draw_dot \
    --save_qupath
