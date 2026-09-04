#!/usr/bin/env python
"""
run_panospace.py -- PanoSpace as an external benchmark method for HEDeST.

One command takes the same inputs HEDeST uses (H&E TIFF, spot-level ST AnnData,
single-cell reference *or* pre-computed proportions, HoVer-Net segmentation) and
produces the single-cell cell-type assignment in HEDeST's output shape.

Typical calls
-------------
Benchmark run -- reuse the existing segmentation and the ground-truth spot
proportions, so only PanoSpace's own cell-level assignment is being measured::

    python external/PanoSpace/run_panospace.py \\
        --he        .../bench_data/breast_s6/he.tiff \\
        --st        .../bench_data/breast_s6/pseudovisium.h5ad \\
        --seg-dict  .../bench_data/breast_s6/hovernet.json \\
        --proportions .../bench_data/breast_s6/sim/level0/proportions.csv \\
        --output    results/panospace/breast_s6/level0

Full pipeline -- PanoSpace segments *and* deconvolves on its own::

    python external/PanoSpace/run_panospace.py \\
        --he .../he.tiff --st .../pseudovisium.h5ad \\
        --sc-ref .../references/breast.h5ad --celltype-key cell_type \\
        --output results/panospace/breast_s6/full

Deconvolution only -- write ``proportions.csv`` and stop, so several PanoSpace
runs can reuse it afterwards via ``--proportions``::

    python external/PanoSpace/run_panospace.py \\
        --st .../pseudovisium.h5ad --sc-ref .../references/breast.h5ad \\
        --celltype-key cell_type --deconv-only \\
        --output results/panospace/breast_s6/deconv

Outputs (in ``--output``)
-------------------------
``panospace_predictions.csv``
    ``cell_id`` x cell type, one row per annotated nucleus, one-hot.  Same
    shape as HEDeST's ``pred_best_adjusted``, so ``df.idxmax(axis=1)`` yields
    the predicted label.  Cell ids are the ids of the segmentation dict.
``proportions.csv``
    Only when PanoSpace computed the deconvolution itself. ``spot_id`` x cell
    type, rows summing to 1 -- same layout as
    ``bench_data/{sample}/sim/{level}/proportions.csv``.
``segmentation.json``
    Only when PanoSpace ran the segmentation itself (HoVer-Net format).
``run_info.json`` / ``run.log``
    Parameters, per-stage timings and cell/spot counts.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# Make the vendored `panospace` package importable when this file is run as a
# script from the repository root (no install required).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logger = logging.getLogger("run_panospace")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_panospace.py",
        description="Run PanoSpace end-to-end and emit HEDeST-shaped outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("inputs / outputs")
    io.add_argument(
        "--he", default=None, help="High-resolution H&E image (TIFF/PNG/JPEG). Required unless --deconv-only."
    )
    io.add_argument(
        "--st", required=True, help="Spot-level spatial transcriptomics AnnData (.h5ad) with .obsm['spatial']."
    )
    io.add_argument("--output", required=True, help="Directory where results are written.")
    io.add_argument(
        "--sample-name",
        default=None,
        help="Name used for cache namespacing and in run_info.json (default: --output basename).",
    )

    seg = p.add_argument_group("segmentation")
    seg.add_argument(
        "--seg-dict",
        default=None,
        help="Existing segmentation in HoVer-Net JSON format. When given, no "
        "segmentation model is run and the output cell ids are the ids of this file.",
    )
    seg.add_argument(
        "--seg-model",
        default="HIPT",
        choices=["HIPT", "SAM"],
        help="CellViT variant used when --seg-dict is absent (upstream default: HIPT).",
    )
    seg.add_argument("--seg-overlap", type=int, default=64, help="Tile overlap for CellViT, in pixels.")
    seg.add_argument(
        "--no-morphology",
        action="store_true",
        help="Ignore the PanNuke class of each nucleus (disables the morphology " "prior; alpha then has no effect).",
    )

    dec = p.add_argument_group("cell-type proportions")
    dec.add_argument(
        "--proportions",
        default=None,
        help="Pre-computed spot x cell-type proportions CSV. When given, the "
        "deconvolution step is skipped and no proportions file is written.",
    )
    dec.add_argument("--sc-ref", default=None, help="Single-cell reference AnnData (.h5ad) used for deconvolution.")
    dec.add_argument(
        "--celltype-key", default="cell_type", help="Column of --sc-ref .obs holding the cell-type labels."
    )
    dec.add_argument("--sc-layer", default=None, help="Layer of --sc-ref to use as raw counts (default: .X).")
    dec.add_argument(
        "--sc-min-cells-per-type", type=int, default=0, help="Drop reference cell types with fewer cells than this."
    )
    dec.add_argument(
        "--sc-max-cells-per-type",
        type=int,
        default=0,
        help="Sub-sample the reference to at most this many cells per type "
        "(0 = no sub-sampling). RCTD densifies the reference, so large "
        "atlases need this.",
    )
    dec.add_argument("--sc-drop-types", default=None, help="Comma-separated reference cell types to drop.")
    dec.add_argument(
        "--deconv-methods",
        default="RCTD,cell2location,spatialDWLS",
        help="Comma-separated deconvolution backends to ensemble with EnDecon.",
    )
    dec.add_argument(
        "--deconv-only", action="store_true", help="Stop after deconvolution and write proportions.csv only."
    )
    dec.add_argument(
        "--deconv-stop-on-error",
        action="store_true",
        help="Abort when a deconvolution backend fails. Upstream keeps going as "
        "long as at least one backend succeeds, which is the default here too.",
    )
    dec.add_argument(
        "--deconv-allow-negative",
        action="store_true",
        help="Accept negative values from a backend. Upstream rejects them "
        "(require_nonnegative=True), which is the default here too.",
    )
    dec.add_argument(
        "--sc-drop-na-labels",
        action="store_true",
        help="Drop reference cells whose label is NaN/unknown. Off by default: "
        "upstream passes the reference through untouched.",
    )

    geo = p.add_argument_group("geometry")
    geo.add_argument(
        "--spot-radius",
        type=float,
        default=None,
        help="Spot radius in full-resolution pixels. Auto-detected from the ST " "object when omitted.",
    )
    geo.add_argument("--mpp", type=float, default=None, help="Microns per pixel of the H&E image.")
    geo.add_argument("--spot-diameter-um", type=float, default=None, help="Spot diameter in microns (used with --mpp).")

    sr = p.add_argument_group("super-resolution")
    sr.add_argument(
        "--sr-crop-radius",
        type=int,
        default=None,
        help="Half-size, in pixels, of the DINOv2 centre crop. Defaults to the "
        "slide's spot radius, which is the paper's definition (local patch "
        "2r x 2r, neighbourhood 6r x 6r, r = spot radius). Pass 129 to "
        "reproduce the hard-coded constant of the released code instead.",
    )
    sr.add_argument(
        "--neighb", type=int, default=3, help="Neighbourhood crop is --neighb times wider than the centre crop."
    )
    sr.add_argument("--epochs", type=int, default=50, help="Training epochs of the super-resolution head.")
    sr.add_argument("--lr", type=float, default=1e-4, help="Learning rate of the super-resolution head.")
    sr.add_argument("--sr-batch-size", type=int, default=32, help="Batch size of the super-resolution head.")
    sr.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early-stopping patience on val_loss (0 = disabled, the upstream default).",
    )
    sr.add_argument(
        "--mask-mode",
        default="largest",
        choices=["largest", "spots", "none"],
        help="Which tissue fragments define the sub-spot grid. 'largest' is upstream: "
        "the single biggest contour only, so every other piece of a "
        "multi-fragment section is silently discarded. 'spots' keeps every "
        "fragment carrying at least one measured spot -- needed on sections cut "
        "into several pieces. 'none' keeps the whole image.",
    )
    sr.add_argument(
        "--mask-min-spots",
        type=int,
        default=1,
        help="With --mask-mode spots, the minimum number of measured spots a "
        "fragment must carry to be kept. 1 keeps anything with a spot on it, "
        "including specks of debris; a handful keeps only real tissue pieces.",
    )
    sr.add_argument(
        "--mask-downscale",
        default="1",
        help="Downscale factor for tissue-contour detection. 1 = upstream "
        "(full-resolution Canny); 'auto' keeps the proxy under ~16 MP, which "
        "is 10x faster and 3x lighter for a 0.06%% difference in the grid.",
    )

    ann = p.add_argument_group("annotation")
    ann.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Weight of the morphology prior against the super-resolved proportions.",
    )
    ann.add_argument(
        "--ot-mode",
        default="emd",
        choices=["emd", "sinkhorn"],
        help="Optimal-transport variant aligning cell types to PanNuke classes.",
    )
    ann.add_argument("--sinkhorn-reg", type=float, default=0.01, help="Sinkhorn regularisation.")
    ann.add_argument(
        "--solver",
        default="auto",
        choices=["auto", "flow", "gurobi", "scip"],
        help="Assignment solver. 'auto' reproduces the published method: the MILP, "
        "with Gurobi if licensed else SCIP. 'flow' is an exact min-cost-flow "
        "reformulation of the same model -- same optimum, seconds instead of "
        "hours -- for when the MILP is intractable.",
    )
    ann.add_argument("--no-global-quota", action="store_true", help="Drop the global cell-type quota.")
    ann.add_argument("--no-spot-quota", action="store_true", help="Drop the per-spot cell-type quotas.")

    run = p.add_argument_group("runtime")
    run.add_argument(
        "--device", default=None, choices=["cuda", "cpu"], help="Compute device (default: cuda when available)."
    )
    run.add_argument(
        "--cache-dir",
        default="~/.panospace_cache",
        help="Where DINOv2 weights, sub-spot grids and super-resolution "
        "checkpoints are cached (outside the repository).",
    )
    run.add_argument(
        "--deconv-cache-dir", default=None, help="Deconvolution checkpoint directory (default: <cache-dir>/deconv)."
    )
    run.add_argument("--seed", type=int, default=42, help="Random seed.")
    run.add_argument("--overwrite", action="store_true", help="Recompute even when the output files already exist.")
    run.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    run.add_argument(
        "--scip-verbose",
        action="store_true",
        help="Let SCIP print its progress table (nodes, primal/dual bound, gap) "
        "while it solves the MILP. Diagnostics only -- the model and its "
        "optimum are unchanged.",
    )
    run.add_argument(
        "--scip-time-limit",
        type=float,
        default=0,
        help="Seconds after which SCIP gives up (0 = no limit, as upstream). "
        "On timeout the run fails rather than returning a suboptimal "
        "assignment.",
    )
    return p


# Third-party loggers that are unusable at DEBUG level: PIL emits one record
# per TIFF tile (tens of thousands for a whole slide), matplotlib per font.
_NOISY_LOGGERS = ("PIL", "matplotlib", "urllib3", "numba", "filelock", "h5py", "fsspec", "asyncio", "jax", "absl")


def setup_logging(out_dir: Path, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    fh = logging.FileHandler(out_dir / "run.log")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(level, logging.INFO))


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, args.verbose)

    t_start = time.time()
    timings: Dict[str, float] = {}
    sample = args.sample_name or out_dir.name

    # ---- argument sanity -------------------------------------------------
    if args.proportions and args.sc_ref:
        raise SystemExit("--proportions and --sc-ref are mutually exclusive.")
    if not args.proportions and not args.sc_ref:
        raise SystemExit("Provide either --proportions (pre-computed) or --sc-ref (compute them).")
    if args.deconv_only and args.proportions:
        raise SystemExit("--deconv-only needs --sc-ref: there is nothing to compute from --proportions.")
    if not args.deconv_only and not args.he:
        raise SystemExit("--he is required (the super-resolution stage reads the H&E image).")
    if args.he and not os.path.exists(args.he):
        raise SystemExit(f"H&E image not found: {args.he}")

    import panospace as ps
    from panospace import bench

    if args.scip_verbose:
        os.environ["PANOSPACE_SCIP_VERBLEVEL"] = "4"
    if args.scip_time_limit:
        os.environ["PANOSPACE_SCIP_TIMELIMIT"] = str(args.scip_time_limit)

    cache_dir = os.path.expanduser(args.cache_dir)
    deconv_cache = args.deconv_cache_dir or os.path.join(cache_dir, "deconv")

    pred_path = out_dir / "panospace_predictions.csv"
    prop_path = out_dir / "proportions.csv"
    seg_path = out_dir / "segmentation.json"

    logger.info("=" * 70)
    logger.info("PanoSpace | sample=%s | output=%s", sample, out_dir)
    logger.info("=" * 70)

    # ---- 1. spatial transcriptomics --------------------------------------
    t0 = time.time()
    adata_vis = bench.load_st_adata(args.st)
    radius = bench.infer_spot_radius(
        adata_vis,
        radius=args.spot_radius,
        mpp=args.mpp,
        spot_diameter_um=args.spot_diameter_um,
    )
    timings["load_st"] = time.time() - t0

    # ---- 2. cell-type proportions ----------------------------------------
    wrote_proportions = False
    t0 = time.time()
    if args.proportions:
        prop_df = bench.load_proportions(args.proportions)
        deconv_adata = bench.attach_proportions(adata_vis, prop_df)
    else:
        if args.deconv_only and prop_path.exists() and not args.overwrite:
            logger.info("%s already exists and --overwrite was not given; nothing to do.", prop_path)
            return 0
        drop = [t.strip() for t in args.sc_drop_types.split(",")] if args.sc_drop_types else None
        sc_adata = bench.load_sc_reference(
            args.sc_ref,
            celltype_key=args.celltype_key,
            layer=args.sc_layer,
            min_cells_per_type=args.sc_min_cells_per_type,
            max_cells_per_type=args.sc_max_cells_per_type or None,
            drop_types=drop,
            drop_na_labels=args.sc_drop_na_labels,
            seed=args.seed,
        )
        methods = [m.strip() for m in args.deconv_methods.split(",") if m.strip()]
        logger.info("Running deconvolution with %s", methods)
        deconv_adata = ps.deconv_celltype(
            adata_vis=adata_vis,
            sc_adata=sc_adata,
            celltype_key=args.celltype_key,
            methods=methods,
            cache_dir=deconv_cache,
            project_name=sample,
            resume=not args.overwrite,
            continue_on_error=not args.deconv_stop_on_error,
            require_nonnegative=not args.deconv_allow_negative,
        )
        bench.write_proportions(bench.proportions_from_deconv(deconv_adata), str(prop_path))
        wrote_proportions = True
        del sc_adata
    timings["deconvolution"] = time.time() - t0

    deconv_adata.uns["radius"] = int(radius)
    cell_types = [str(c) for c in deconv_adata.uns["celltype"]]
    logger.info("%d cell types: %s", len(cell_types), cell_types)

    if args.deconv_only:
        _write_run_info(
            out_dir,
            args,
            sample,
            radius,
            cell_types,
            timings,
            t_start,
            n_spots=deconv_adata.n_obs,
            wrote_proportions=wrote_proportions,
        )
        logger.info("Deconvolution-only run finished in %s", _fmt(time.time() - t_start))
        return 0

    if pred_path.exists() and not args.overwrite:
        logger.info("%s already exists and --overwrite was not given; nothing to do.", pred_path)
        return 0

    # ---- 3. segmentation --------------------------------------------------
    t0 = time.time()
    use_morphology = not args.no_morphology
    if args.seg_dict:
        seg_adata, _contours = ps.detect_cells(seg_dict=args.seg_dict, use_morphology=use_morphology)
        # The contour lists keep the whole parsed JSON alive (GBs for a whole
        # slide) and nothing downstream needs them: only centroids and types.
        del _contours
        wrote_segmentation = False
    else:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        logger.info("No --seg-dict given: running CellViT-%s on %s", args.seg_model, args.he)
        img = Image.open(args.he)
        seg_adata, _contours, new_seg_dict = ps.detect_cells(
            img,
            model="cellvit",
            model_name=args.seg_model,
            device=args.device,
            tile_size=256,
            overlap=args.seg_overlap,
            use_morphology=use_morphology,
            return_seg_dict=True,
            mpp=args.mpp,
        )
        bench.write_seg_dict(new_seg_dict, str(seg_path))
        wrote_segmentation = True
        del img, new_seg_dict, _contours
    n_cells_in = seg_adata.n_obs
    timings["segmentation"] = time.time() - t0

    # ---- 4. super-resolution ---------------------------------------------
    t0 = time.time()
    accelerator = _accelerator(args.device)
    # Paper (Nat Comput Sci 2026, Methods): "a local patch centered on the spot
    # (size 2r x 2r, where r is the spot radius) and a larger neighborhood patch
    # (size 6r x 6r)". The released code hard-codes r = 129 px instead.
    sr_crop_radius = args.sr_crop_radius if args.sr_crop_radius is not None else int(radius)
    logger.info(
        "DINOv2 crop radius: %d px (%s)",
        sr_crop_radius,
        "explicit --sr-crop-radius" if args.sr_crop_radius is not None else "spot radius, per the paper",
    )
    sr_adata = ps.superres_celltype(
        deconv_adata=deconv_adata,
        img_dir=args.he,
        neighb=args.neighb,
        radius=sr_crop_radius,
        epoch=args.epochs,
        learning_rate=args.lr,
        batch_size=args.sr_batch_size,
        local_path=os.path.join(cache_dir, "dinov2-base"),
        cache_dir=cache_dir,
        accelerator=accelerator,
        precision="16-mixed" if accelerator == "gpu" else "32",
        seed=args.seed,
        patience=args.patience or None,
        mask_mode=args.mask_mode,
        mask_min_spots=args.mask_min_spots,
        mask_downscale=args.mask_downscale if args.mask_downscale == "auto" else int(args.mask_downscale),
    )
    timings["superres"] = time.time() - t0
    logger.info("Super-resolution grid: %d sub-spots", sr_adata.n_obs)

    # The decoded slide (several GB) is no longer needed once the DINOv2
    # features are extracted; the assignment stage is pure numpy/flow.
    from panospace._core.annotation._superres_backend.superres_utils import release_image

    release_image()

    # ---- 5. single-cell annotation ---------------------------------------
    t0 = time.time()
    seg_adata_pred, _annotator = ps.celltype_annotator(
        decov_adata=deconv_adata,
        sr_deconv_adata=sr_adata,
        seg_adata=seg_adata,
        alpha=args.alpha,
        ot_mode=args.ot_mode,
        sinkhorn_reg=args.sinkhorn_reg,
        _global_quota=not args.no_global_quota,
        _spot_quota=not args.no_spot_quota,
        solver=args.solver,
    )
    timings["annotation"] = time.time() - t0

    # ---- 6. outputs -------------------------------------------------------
    pred_df = bench.predictions_to_frame(seg_adata_pred, cell_types)
    bench.write_predictions(pred_df, str(pred_path))

    n_dropped = n_cells_in - pred_df.shape[0]
    if n_dropped:
        logger.warning(
            "%d/%d nuclei (%.2f%%) were not covered by any sub-spot and are absent from the "
            "predictions; align on the CSV index when scoring.",
            n_dropped,
            n_cells_in,
            100.0 * n_dropped / max(n_cells_in, 1),
        )

    _write_run_info(
        out_dir,
        args,
        sample,
        radius,
        cell_types,
        timings,
        t_start,
        n_spots=deconv_adata.n_obs,
        wrote_proportions=wrote_proportions,
        n_cells_in=n_cells_in,
        n_cells_out=int(pred_df.shape[0]),
        n_subspots=int(sr_adata.n_obs),
        wrote_segmentation=wrote_segmentation,
        label_counts={k: int(v) for k, v in seg_adata_pred.obs["pred_cell_type"].value_counts().items()},
    )
    logger.info("Done in %s -> %s", _fmt(time.time() - t_start), pred_path)
    return 0


def _accelerator(device: Optional[str]) -> str:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "gpu"
    try:
        import torch

        return "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _write_run_info(
    out_dir: Path,
    args,
    sample: str,
    radius: int,
    cell_types: List[str],
    timings: Dict[str, float],
    t_start: float,
    **extra: Any,
) -> None:
    info: Dict[str, Any] = {
        "sample": sample,
        "params": vars(args),
        "spot_radius_px": int(radius),
        "cell_types": cell_types,
        "n_cell_types": len(cell_types),
        "timings_sec": {k: round(v, 2) for k, v in timings.items()},
        "total_sec": round(time.time() - t_start, 2),
    }
    info.update(extra)
    with open(out_dir / "run_info.json", "w") as fh:
        json.dump(info, fh, indent=2, default=str)
    logger.info("Wrote %s", out_dir / "run_info.json")


if __name__ == "__main__":
    raise SystemExit(main())
