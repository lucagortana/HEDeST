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

python3 -u run2.py