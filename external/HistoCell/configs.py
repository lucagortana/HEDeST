from __future__ import annotations

import json

from yacs.config import CfgNode as CN


def _get_config(tissue_type, deconv, subtype, k_class, tissue_dir):
    config = CN()
    config.train = CN()
    config.train.lr = 0.0005
    config.train.epoch = 41
    config.train.val_iter = 10
    config.train.val_min_iter = 9

    config.data = CN()
    config.data.deconv = deconv
    config.data.save_model = f"./train_log/{tissue_type}/models"  # model saved
    config.data.ckpt = f"./train_log/{tissue_type}/ckpts"  # eval results saved
    config.data.tile_dir = f"./demo/data/{tissue_type}/tiles"  # path to tiles
    config.data.mask_dir = f"./demo/data/{tissue_type}/seg"  # path to json segmentation file
    config.data.batch_size = 32
    config.data.tissue_dir = tissue_dir  # tissue compartment directory
    config.data.max_cell_num = 256  # max cell number in a single tile for batch learning
    config.data.cell_dir = (
        f"./demo/data/{tissue_type}/cell_proportion/type/{config.data.deconv}"  # path to cell proportion label
    )

    config.model = CN()
    config.model.tissue_class = 3
    config.model.pretrained = True
    config.model.channels = 3
    config.model.k_class = k_class

    return config


def _get_cell_state_config(tissue_type, deconv, subtype, tissue_dir):
    config = CN()
    config.train = CN()
    config.train.lr = 0.0005
    config.train.epoch = 41
    config.train.val_iter = 5
    config.train.val_min_iter = 9
    config.train.state_epoch = 41

    config.data = CN()
    config.data.deconv = deconv
    config.data.save_model = f"./train_log/{tissue_type}/models"
    config.data.ckpt = f"./train_log/{tissue_type}/ckpts"
    config.data.tile_dir = f"./demo/data/{tissue_type}/tiles"
    config.data.mask_dir = f"./demo/data/{tissue_type}/seg"
    config.data.batch_size = 32
    config.data.tissue_dir = tissue_dir
    config.data.max_cell_num = 256
    with open(tissue_dir, "r") as tissue_file:
        tc = json.load(tissue_file)

    config.data.cell_dir = f"./demo/data/{tissue_type}/cell_proportion/type/{config.data.deconv}"
    config.data.state_dir = (
        f"./demo/data/{tissue_type}/cell_proportion/state/{config.data.deconv}"  # path to cell state directory
    )

    config.model = CN()
    config.model.tissue_class = len(tc["list"])
    config.model.pretrained = True
    config.model.channels = 3
    config.model.k_class = len(tc["dict"])

    k_state = 0
    for key, value in tc["state"].items():
        k_state += int(value)
    config.model.k_state = k_state
    print(k_state)

    return config


# ---------------------------------------------------------------------------
# Benchmark configuration (see run_histocell.py)
# ---------------------------------------------------------------------------
# The two blocks above read tiles, segmentations and proportions from a fixed
# ./demo/data/... layout.  run_histocell.py is given the four benchmark files
# directly, so the config below carries only the *method* settings; every path
# arrives as a command-line argument.
#
# Where the released code and the paper disagree, the paper wins, matching how
# the rest of this benchmark treats published methods:
#
#   epochs   paper: 50 for the cell-type stage ("we trained the model for 50
#            epochs with the supervision of cell compartment loss and KL
#            divergence loss")   -- released configs.py: 41
#   lr       paper: 1e-4 ("Adam optimizer with a learning rate of 1 x 10-4")
#            -- released configs.py: 5e-4
#
# Both are exposed as --epochs / --lr, so the released values are one flag away.
PAPER_EPOCHS, PAPER_LR = 50, 1e-4
RELEASED_EPOCHS, RELEASED_LR = 41, 0.0005


def _get_bench_config(
    k_class,
    tissue_class=3,
    epochs=PAPER_EPOCHS,
    lr=PAPER_LR,
    batch_size=32,
    max_cell_num=256,
    pretrained=True,
    channels=3,
):
    config = CN()
    config.train = CN()
    config.train.lr = lr
    config.train.epoch = epochs
    config.train.val_iter = 10  # checkpoint cadence, as released
    config.train.val_min_iter = 9

    config.data = CN()
    config.data.batch_size = batch_size
    config.data.max_cell_num = max_cell_num  # affects the result, not just the
    # batch shape: padded slots keep a
    # non-zero attention weight in the
    # GAT, so this is a real parameter

    config.model = CN()
    config.model.tissue_class = tissue_class
    config.model.pretrained = pretrained
    config.model.channels = channels
    config.model.k_class = k_class

    return config
