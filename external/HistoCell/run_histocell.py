#!/usr/bin/env python
"""HistoCell on one benchmark slide: train on its spots, predict every nucleus.

HistoCell was published as a set of pre-trained models, one per cancer type,
but no weights were released -- so the model is trained here from scratch on
the slide it will annotate, using that slide's spot proportions as the weak
supervision label, and then applied to every nucleus of the same slide.

    train    one tile per spot (side = spot diameter), supervised by
             proportions.csv through the symmetric KL loss, plus the tissue
             compartment cross-entropy
    predict  the slide is partitioned into tiles of the same size, so that
             every nucleus is seen once, at the magnification the model was
             trained on

Inputs are the benchmark's own files -- the same four PanoSpace takes -- and no
single-cell reference is needed, because the proportions are already given.

Outputs
-------
histocell_predictions.csv   cell_id x cell type, shaped like HEDeST's
                            `pred_best_adjusted`; rows are softmax
                            probabilities, so `df.idxmax(axis=1)` is the call
histocell_proportions.csv   spot x cell type, shaped like
                            bench_data/{sample}/sim/{level}/proportions.csv
tissue_compartment.json     the tcs file used (generated unless one is passed)
run_info.json, run.log      provenance
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench  # noqa: E402
from configs import _get_bench_config, PAPER_EPOCHS, PAPER_LR  # noqa: E402
from data import SlideTileDataset  # noqa: E402
from model.arch import HistoCell  # noqa: E402
from utils.utils import setup_seed, save_checkpoint  # noqa: E402

logger = logging.getLogger("run_histocell")


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    io = p.add_argument_group("inputs / outputs")
    io.add_argument("--he", required=True, help="whole-slide H&E image (.tiff)")
    io.add_argument("--st", required=True, help="spot AnnData (.h5ad), for spot coordinates")
    io.add_argument("--seg-dict", required=True, help="whole-slide HoVer-Net segmentation (.json)")
    io.add_argument(
        "--proportions", required=True, help="spot x cell-type proportions (.csv), the weak supervision label"
    )
    io.add_argument("--output", required=True, help="output directory")
    io.add_argument("--sample-name", default=None, help="label used in logs and run_info.json")
    io.add_argument(
        "--spot-dict",
        default=None,
        help="optional spot_id -> [cell_id] map. Restricts each training tile to "
        "the spot's own cells and defines spot membership when clumping. "
        "Off by default: upstream keeps every nucleus of the square tile.",
    )

    m = p.add_argument_group("method (defaults are HistoCell's own)")
    m.add_argument(
        "--epochs",
        type=int,
        default=PAPER_EPOCHS,
        help=f"training epochs (default {PAPER_EPOCHS}, the paper's cell-type stage; "
        "the released configs.py says 41)",
    )
    m.add_argument(
        "--lr",
        type=float,
        default=PAPER_LR,
        help=f"Adam learning rate (default {PAPER_LR}, the paper's; " "the released configs.py says 5e-4)",
    )
    m.add_argument("--batch-size", type=int, default=32, help="tiles per batch (default 32)")
    m.add_argument(
        "--max-cell-num", type=int, default=256, help="cells per tile fed to the model (default 256, as released)"
    )
    m.add_argument(
        "--tile-px",
        type=float,
        default=None,
        help="tile side in slide pixels (default: the spot diameter, per the paper)",
    )
    m.add_argument(
        "--tissue-compartment",
        default=None,
        help="upstream-style tcs .json; generated from the cell-type names if omitted",
    )
    m.add_argument("--seed", type=int, default=47, help="random seed (default 47, as released)")

    r = p.add_argument_group("run")
    r.add_argument(
        "--infer-tiling",
        choices=["grid", "spots"],
        default="grid",
        help="grid: partition the slide so every nucleus is predicted (default). "
        "spots: predict only inside the spot tiles.",
    )
    r.add_argument(
        "--eval-mode",
        choices=["train", "eval"],
        default="train",
        help="module mode at inference. 'train' is what upstream infer.py does "
        "(batch-norm on batch statistics, dropout on); 'eval' is the usual "
        "deterministic choice. Default follows upstream.",
    )
    r.add_argument("--num-workers", type=int, default=6, help="DataLoader workers (default 6)")
    r.add_argument("--resume", default=None, help="skip training and load this checkpoint")
    r.add_argument("--no-plots", action="store_true", help="skip the diagnostic figures")
    return p.parse_args(argv)


def setup_logging(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    fh = logging.FileHandler(os.path.join(out_dir, "run.log"), mode="w")
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ---------------------------------------------------------------------------
def train(model, loader, config, device):
    """Upstream ``train.py::train_loop``, verbatim, wrapped in the epoch loop."""
    loss_func = torch.nn.KLDivLoss()
    aux_loss = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.train.lr)
    model.to(device)

    for epoch in range(config.train.epoch):
        model.train()
        running, seen = 0.0, 0
        bar = tqdm(
            loader,
            desc=f"epoch {epoch}",
            total=len(loader),
            unit="batch",
            dynamic_ncols=True,
            file=sys.stdout,
            mininterval=60.0,
        )
        for idx, data in enumerate(bar):
            tissue = data["tissue"].to(torch.float32).to(device)
            images = data["image"].to(torch.float32).to(device)
            cell_proportion = data["cells"].to(torch.float32).to(device)
            valid_mask = data["mask"].to(torch.long).to(device)
            cell_size = data["size"].to(torch.float32).to(device)
            adj_mat = data["adj"].to(torch.float32).to(device)
            gt_tissue_cat = data["tissue_cat"].to(device)
            if torch.sum(data["mask"]) <= 0:
                continue
            batch, cells, channels, height, width = images.shape
            images = images.reshape(batch * cells, channels, height, width)
            probs, pred_proportion, tissue_cat, _ = model(
                tissue, images, adj_mat, cell_size, valid_mask, {"batch": batch, "cells": cells}
            )
            cell_prop_list = []
            for single_props, valid_index in zip(cell_proportion, valid_mask):
                if valid_index <= 0:
                    continue
                cell_prop_list.append(single_props)
            cell_proportion = torch.stack(cell_prop_list, dim=0)

            loss_out = (
                loss_func((cell_proportion + 1e-10).log(), pred_proportion + 1e-10)
                + loss_func((pred_proportion + 1e-10).log(), cell_proportion + 1e-10)
                + aux_loss(tissue_cat, gt_tissue_cat)
            )
            loss_out.backward()
            optimizer.step()
            optimizer.zero_grad()

            running += loss_out.item()
            seen += 1
            # refresh=False: upstream's per-iteration description, without one
            # progress-bar line per batch in a Slurm log.
            bar.set_description(f"epoch:{epoch} iter:{idx} loss:{loss_out.item():.4f}", refresh=False)
        bar.close()
        logger.info("epoch %d/%d  mean loss %.5f", epoch, config.train.epoch - 1, running / max(seen, 1))

    return model, optimizer


@torch.no_grad()
def predict(model, loader, device, n_nuclei, k_class, eval_mode):
    """Upstream ``infer.py::val_loop``, keeping only what maps back to cell ids.

    Upstream leaves the module in train mode here (``model.train()``), so
    batch-norm uses batch statistics and dropout stays on; ``--eval-mode eval``
    switches that off.  Two of the dropouts live inside ``arch.py`` with
    ``training=True`` hard-coded and are active either way -- untouched.
    """
    model.train() if eval_mode == "train" else model.eval()

    probs = np.full((n_nuclei, k_class), np.nan, dtype=np.float32)
    n_tiles = 0
    bar = tqdm(
        loader, total=len(loader), unit="batch", dynamic_ncols=True, file=sys.stdout, mininterval=60.0, desc="predict"
    )
    for data in bar:
        tissue = data["tissue"].to(torch.float32).to(device)
        images = data["image"].to(torch.float32).to(device)
        cell_size = data["size"].to(torch.float32).to(device)
        adj_mat = data["adj"].to(torch.float32).to(device)
        valid_mask = data["mask"].to(torch.long).to(device)
        if torch.sum(data["mask"]) <= 0:
            continue
        batch, cells, channels, height, width = images.shape
        images = images.reshape(batch * cells, channels, height, width)
        prob_list, _, _, _ = model(tissue, images, adj_mat, cell_size, valid_mask, {"batch": batch, "cells": cells})

        kept = [(vi, ci) for vi, ci in zip(valid_mask, data["cell_index"]) if vi > 0]
        for probs_i, (valid_index, cell_index) in zip(prob_list, kept):
            n = int(valid_index)
            rows = cell_index[:n].numpy()
            probs[rows] = probs_i.detach().cpu().numpy()[:n]
            n_tiles += 1
    bar.close()

    covered = int(np.isfinite(probs[:, 0]).sum())
    logger.info("Predicted %d/%d nuclei over %d tiles", covered, n_nuclei, n_tiles)
    return probs


# ---------------------------------------------------------------------------
def _palette(sim_dir, level, cell_types, sample):
    """Colours for the cell types, in the benchmark's own hierarchical code.

    Reproduced inside this package (`bench.level_palette`) and derived from the
    slide's own `proportions.csv` files -- nothing outside this repository is
    imported.  Falls back to tab20 if the levels cannot be read.
    """
    try:
        pal = bench.level_palette(sim_dir, level)
        missing = [c for c in cell_types if c not in pal]
        if not missing:
            logger.info("Palette: hierarchical colour code for %s level %s", sample, level)
            return {c: tuple(pal[c]) for c in cell_types}
        logger.warning("Hierarchical palette misses %s; falling back to tab20", missing)
    except Exception as exc:
        logger.warning("Hierarchical palette unavailable (%s: %s); falling back to tab20", type(exc).__name__, exc)
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    return {c: cmap(i % 20) for i, c in enumerate(cell_types)}


def _thumbnail(path):
    """Lowest-resolution level of the pyramid, for plot backgrounds."""
    import tifffile

    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        levels = getattr(series, "levels", None)
        level = levels[-1] if levels and len(levels) > 1 else series
        return level.asarray()


def _draw_pies(ax, xy, values, cell_types, colours, radius):
    """One pie per spot, drawn at the spot's own location."""
    from matplotlib.patches import Wedge
    from matplotlib.collections import PatchCollection

    wedges, face = [], []
    for (x, y), row in zip(xy, values):
        total = row.sum()
        if total <= 0:
            continue
        angle = 90.0
        for k, ct in enumerate(cell_types):
            frac = row[k] / total
            if frac <= 0:
                continue
            nxt = angle - 360.0 * frac
            wedges.append(Wedge((x, y), radius, nxt, angle))
            face.append(colours[ct])
            angle = nxt
    ax.add_collection(PatchCollection(wedges, facecolors=face, edgecolors="none", linewidths=0, rasterized=True))


def make_plots(out_dir, reader, nuclei, pred, truth, fitted, coords, cell_types, sample, level, sim_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.spatial import cKDTree

    colours = _palette(sim_dir, level, cell_types, sample)
    thumb = _thumbnail(reader.path)
    H, W = reader.height, reader.width

    # 1 -- every predicted nucleus on the slide
    calls = pred.idxmax(axis=1)
    xy_cells = nuclei.loc[pred.index, ["cx", "cy"]].to_numpy()
    fig, ax = plt.subplots(figsize=(13, 13 * H / W + 1))
    ax.imshow(thumb, extent=[0, W, H, 0], alpha=0.45, interpolation="bilinear")
    for ct in cell_types:
        sel = (calls == ct).to_numpy()
        if sel.sum():
            ax.scatter(xy_cells[sel, 0], xy_cells[sel, 1], s=0.7, linewidths=0, color=colours[ct], rasterized=True)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"{sample} level {level} — HistoCell single-nucleus predictions, " f"{len(pred):,} nuclei", fontsize=14
    )
    fig.legend(
        handles=[Patch(facecolor=colours[c], label=c) for c in cell_types],
        loc="lower center",
        ncol=min(len(cell_types), 6),
        frameon=False,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(os.path.join(out_dir, "01_annotation_slide.png"), dpi=130)
    plt.close(fig)

    # 2 -- the paper's own metric: predicted spot proportions vs the labels
    common = truth.index.intersection(fitted.index)
    T = truth.loc[common, cell_types].to_numpy(float)
    P = fitted.loc[common, cell_types].to_numpy(float)
    n = len(cell_types)
    ncol = min(n, 6)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.9 * nrow), squeeze=False)
    pccs = []
    for k, ct in enumerate(cell_types):
        r = float(np.corrcoef(T[:, k], P[:, k])[0, 1])
        pccs.append(r)
        ax = axes[k // ncol][k % ncol]
        ax.scatter(T[:, k], P[:, k], s=6, alpha=0.3, linewidths=0, color=colours[ct], rasterized=True)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"{ct}\nPCC = {r:.3f}", fontsize=10)
        ax.set_xlabel("deconvolved (label)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        if k % ncol == 0:
            ax.set_ylabel("HistoCell (clumped)")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(
        f"{sample} level {level} — spot proportions, {len(common):,} spots, " f"mean PCC = {np.nanmean(pccs):.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_spot_proportion_fit.png"), dpi=140)
    plt.close(fig)

    # 3 -- the same spots side by side: truth on the left, HistoCell on the right
    xy = coords.loc[common, ["x", "y"]].to_numpy(float)
    dist, _ = cKDTree(xy).query(xy, k=2)
    pitch = float(np.median(dist[:, 1]))
    radius = 0.48 * pitch
    fig, axes = plt.subplots(1, 2, figsize=(21, 11 * H / W + 2.0))
    for ax, values, title in ((axes[0], T, "ground truth"), (axes[1], P, "HistoCell (predicted)")):
        ax.imshow(thumb, extent=[0, W, H, 0], alpha=0.35, interpolation="bilinear")
        _draw_pies(ax, xy, values, cell_types, colours, radius)
        ax.set_xlim(xy[:, 0].min() - 3 * pitch, xy[:, 0].max() + 3 * pitch)
        ax.set_ylim(xy[:, 1].max() + 3 * pitch, xy[:, 1].min() - 3 * pitch)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=15)
    fig.legend(
        handles=[Patch(facecolor=colours[c], label=c) for c in cell_types],
        loc="lower center",
        ncol=min(len(cell_types), 7),
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle(
        f"{sample} — level {level}   ·   {len(common):,} spots, "
        f"{len(cell_types)} cell types   ·   mean PCC = {np.nanmean(pccs):.3f}, "
        f"MAE = {np.abs(P - T).mean():.3f}   ·   pies at 0.48x the spot pitch",
        fontsize=16,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.965])
    fig.savefig(os.path.join(out_dir, "03_spot_pies_true_vs_pred.png"), dpi=130)
    plt.close(fig)

    return float(np.nanmean(pccs)), {c: float(r) for c, r in zip(cell_types, pccs)}


# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.abspath(args.output)
    setup_logging(out_dir)
    started = time.time()
    sample = args.sample_name or os.path.basename(out_dir)
    logger.info("HistoCell benchmark run: %s", sample)
    logger.info("arguments: %s", vars(args))

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    # -- inputs ------------------------------------------------------------
    truth = bench.load_proportions(args.proportions)
    cell_types = list(truth.columns)

    # The benchmark's colour code is keyed by (dataset, level); both are recorded
    # in the proportions path (.../bench_data/{sample}/sim/level{L}/proportions.csv).
    prop_path = os.path.abspath(args.proportions)
    sim_dir = os.path.dirname(os.path.dirname(prop_path))
    parts = prop_path.split(os.sep)
    try:
        palette_level = int(parts[-2].replace("level", ""))
        palette_sample = parts[-4]
    except (ValueError, IndexError):
        palette_level, palette_sample = -1, sample
    adata = bench.load_st_adata(args.st)
    coords = bench.spot_coordinates(adata)
    tile_px = args.tile_px if args.tile_px is not None else bench.infer_tile_size(adata)
    nuclei, seg_meta = bench.load_seg_dict(args.seg_dict)

    spots = [s for s in truth.index if s in coords.index]
    dropped = len(truth) - len(spots)
    if dropped:
        logger.warning("%d spot(s) of the proportions table are absent from %s and are skipped", dropped, args.st)
    truth = truth.loc[spots]
    keep = truth.to_numpy(float).sum(axis=1) > 0
    if not keep.all():
        logger.warning("%d spot(s) have all-zero proportions and cannot supervise; dropped", int((~keep).sum()))
        truth = truth.loc[keep]
        spots = list(truth.index)
    coords = coords.loc[spots]

    # -- tissue compartments ------------------------------------------------
    if args.tissue_compartment:
        tcs = bench.load_tissue_compartment(args.tissue_compartment, len(cell_types))
        logger.info("Tissue compartments read from %s", args.tissue_compartment)
    else:
        tcs = bench.make_tissue_compartment(cell_types)
    bench.write_tissue_compartment(tcs, os.path.join(out_dir, "tissue_compartment.json"))
    assert len(tcs["dict"]) == len(cell_types)

    # -- tiles --------------------------------------------------------------
    reader = bench.SlideReader(args.he)
    train_tiles = bench.spot_tiles(coords, tile_px)
    spot_dict = bench.load_spot_dict(args.spot_dict) if args.spot_dict else None
    train_members = bench.assign_nuclei(nuclei, train_tiles, tile_px, restrict=spot_dict)
    n_per = np.array([len(v) for v in train_members.values()])
    non_empty = [t for t in train_tiles.index if len(train_members[t]) > 0]
    logger.info(
        "Training tiles: %d spots, %d px, %d with >=1 nucleus " "(median %d cells, max %d, %d nuclei total)",
        len(train_tiles),
        int(round(tile_px)),
        len(non_empty),
        int(np.median(n_per)),
        int(n_per.max()),
        int(n_per.sum()),
    )
    if n_per.max() > args.max_cell_num:
        logger.warning(
            "%d training tile(s) hold more than max_cell_num=%d nuclei; " "the surplus is dropped, as upstream does",
            int((n_per > args.max_cell_num).sum()),
            args.max_cell_num,
        )
    train_tiles = train_tiles.loc[non_empty]

    config = _get_bench_config(
        k_class=len(cell_types),
        tissue_class=len(tcs["list"]),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_cell_num=args.max_cell_num,
    )
    logger.info(
        "config: k_class=%d tissue_class=%d epochs=%d lr=%g batch=%d max_cell_num=%d",
        config.model.k_class,
        config.model.tissue_class,
        config.train.epoch,
        config.train.lr,
        config.data.batch_size,
        config.data.max_cell_num,
    )

    model = HistoCell(config.model)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        logger.info("Loaded checkpoint %s -- training skipped", args.resume)
    else:
        train_set = SlideTileDataset(
            reader,
            train_tiles,
            nuclei,
            train_members,
            tcs,
            proportions=truth.loc[non_empty],
            tile_px=tile_px,
            aug=True,
            max_cell_num=args.max_cell_num,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        logger.info("Training on %d tiles for %d epochs", len(train_set), config.train.epoch)
        t0 = time.time()
        model, optimizer = train(model, train_loader, config, device)
        logger.info("Training done in %.1f min", (time.time() - t0) / 60)
        os.makedirs(os.path.join(out_dir, "model"), exist_ok=True)
        save_checkpoint(model, optimizer, os.path.join(out_dir, "model", f"epoch_{config.train.epoch - 1}.ckpt"))

    # -- inference ----------------------------------------------------------
    if args.infer_tiling == "grid":
        infer_tiles = bench.grid_tiles(nuclei, tile_px, reader.width, reader.height)
    else:
        infer_tiles = train_tiles
    infer_members = bench.assign_nuclei(nuclei, infer_tiles, tile_px)
    infer_tiles = infer_tiles.loc[[t for t in infer_tiles.index if len(infer_members[t]) > 0]]
    over = sum(1 for t in infer_tiles.index if len(infer_members[t]) > args.max_cell_num)
    if over:
        logger.warning(
            "%d inference tile(s) hold more than max_cell_num=%d nuclei; " "the surplus stays unpredicted",
            over,
            args.max_cell_num,
        )

    infer_set = SlideTileDataset(
        reader,
        infer_tiles,
        nuclei,
        infer_members,
        tcs,
        proportions=None,
        tile_px=tile_px,
        aug=False,
        max_cell_num=args.max_cell_num,
    )
    infer_loader = DataLoader(infer_set, batch_size=16, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logger.info("Predicting over %d tiles (%s tiling)", len(infer_set), args.infer_tiling)
    t0 = time.time()
    probs = predict(model, infer_loader, device, len(nuclei), len(cell_types), args.eval_mode)
    logger.info("Inference done in %.1f min", (time.time() - t0) / 60)

    got = np.isfinite(probs[:, 0])
    pred = pd.DataFrame(probs[got], index=nuclei.index[got], columns=cell_types)
    missing = int((~got).sum())
    if missing:
        logger.warning(
            "%d/%d nuclei (%.2f%%) are absent from the predictions; " "align on the cell ids of %s",
            missing,
            len(nuclei),
            100 * missing / len(nuclei),
            os.path.basename(args.seg_dict),
        )
    bench.write_predictions(pred, os.path.join(out_dir, "histocell_predictions.csv"))

    fitted = bench.clump_to_spots(pred, coords, radius=tile_px / 2.0, spot_dict=spot_dict, nuclei=nuclei)
    fitted = fitted[cell_types]
    bench.write_proportions(fitted, os.path.join(out_dir, "histocell_proportions.csv"))

    mean_pcc, per_type_pcc = (None, None)
    if not args.no_plots:
        try:
            mean_pcc, per_type_pcc = make_plots(
                out_dir, reader, nuclei, pred, truth, fitted, coords, cell_types, palette_sample, palette_level, sim_dir
            )
            logger.info("Spot-level fit to the supervision labels: mean PCC = %.4f", mean_pcc)
        except Exception:
            logger.exception("Plotting failed; the result files are unaffected")

    info = {
        "sample": sample,
        "arguments": vars(args),
        "segmentation_meta": {k: v for k, v in seg_meta.items() if np.isscalar(v)},
        "cell_types": cell_types,
        "tile_px": float(tile_px),
        "n_spots_supervising": int(len(truth)),
        "n_training_tiles": int(len(train_tiles)),
        "n_inference_tiles": int(len(infer_tiles)),
        "n_nuclei_total": int(len(nuclei)),
        "n_nuclei_predicted": int(got.sum()),
        "tissue_compartment": tcs,
        "spot_pcc_mean": mean_pcc,
        "spot_pcc_per_type": per_type_pcc,
        "runtime_seconds": round(time.time() - started, 1),
        "torch": torch.__version__,
    }
    with open(os.path.join(out_dir, "run_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, default=str)

    elapsed = time.time() - started
    logger.info("Done in %dm%02ds -> %s", int(elapsed // 60), int(elapsed % 60), out_dir)


if __name__ == "__main__":
    main()
