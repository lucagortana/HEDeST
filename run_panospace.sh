#!/bin/bash

#SBATCH --job-name=panospace
#SBATCH --output=/cluster/CBIO/home/lgortana/HEDeST/log/panospace_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/HEDeST/log/panospace_%j.err
#SBATCH --gres=gpu:1
#SBATCH -p cbio-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate panospace-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/panospace-env/lib:$LD_LIBRARY_PATH

# RCTD asks Ray for 22 workers (upstream's hard-coded value). Cap it at the
# cores Slurm actually granted -- this changes wall time only, never the result.
export PANOSPACE_RCTD_CORES=${SLURM_CPUS_PER_TASK:-8}

# ---------------------------------------------------------------------------
# Defaults, used only when the script is submitted without arguments.
# ---------------------------------------------------------------------------
SAMPLE=breast_s6
LEVEL=level0
BENCH=/cluster/CBIO/data1/lgortana/STHELAR/bench_data

DEFAULT_ARGS=(
    --he           "$BENCH/$SAMPLE/he.tiff"
    --st           "$BENCH/$SAMPLE/pseudovisium.h5ad"
    --seg-dict     "$BENCH/$SAMPLE/hovernet.json"
    --proportions  "$BENCH/$SAMPLE/sim/$LEVEL/proportions.csv"
    --output       "/cluster/CBIO/data1/lgortana/STHELAR/panospace/$SAMPLE/$LEVEL"
    --sample-name  "${SAMPLE}_${LEVEL}"
)
# NB: no method parameters are set here on purpose. alpha, ot-mode, neighb,
# epochs, the crop radius and the solver all keep PanoSpace's own defaults;
# see `python external/PanoSpace/run_panospace.py --help`.

if [ "$#" -gt 0 ]; then
    ARGS=("$@")
else
    ARGS=("${DEFAULT_ARGS[@]}")
fi

python3 -u external/PanoSpace/run_panospace.py "${ARGS[@]}"
