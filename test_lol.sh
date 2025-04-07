#!/bin/bash

# ---------------------
# Arguments for GPU job
# ---------------------

IMAGE_DICT="/cluster/CBIO/data1/lgortana/Xenium_V1_humanLung_Cancer_FFPE/sim/LuCA/moco_embed_moco-XENHL-rn50.pt"
SIM_CSV="/cluster/CBIO/data1/lgortana/Xenium_V1_humanLung_Cancer_FFPE/sim/LuCA/sim_Xenium_V1_humanLung_Cancer_FFPE_prop_real.csv"
SPOT_DICT="/cluster/CBIO/data1/lgortana/Xenium_V1_humanLung_Cancer_FFPE/sim/LuCA/spot_dict_real.json"
SPOT_DICT_GLOBAL="/cluster/CBIO/data1/lgortana/Xenium_V1_humanLung_Cancer_FFPE/sim/LuCA/spot_dict_global_real.json"

GPU_EXTRA_ARGS=(
    --models quick
    --alphas 0 1e-4
    --learning_rates 3e-4 1e-3
    --weights_options 0
    --divergences kl
    --seeds 42 43
    --batch_size 64
    --out_dir models/quick-semi-sim-emb-v2-2/test
)

# ---------------------
# Arguments for CPU job
# ---------------------

FOLDER="models/quick-semi-sim-emb-v2-2/test"
GT_CSV="/cluster/CBIO/data1/lgortana/Xenium_V1_humanLung_Cancer_FFPE/sim/LuCA/sim_Xenium_V1_humanLung_Cancer_FFPE_gt.csv"

# ---------------------
# Submit GPU job
# ---------------------

GPU_JOB_ID=$(sbatch --parsable run_gridsearch.sh "$IMAGE_DICT" "$SIM_CSV" "$SPOT_DICT" "$SPOT_DICT_GLOBAL" "${GPU_EXTRA_ARGS[@]}")

# ---------------------
# Submit CPU job (depends on GPU)
# ---------------------

CPU_JOB_ID=$(sbatch --parsable --dependency=afterok:$GPU_JOB_ID run_extract_stats.sh "$FOLDER" "$GT_CSV")
