#!/usr/bin/env python
"""
End-to-end smoke test for the DINOv2 super-resolution deconvolution pipeline.

Synthesizes a small fake H&E image + fake deconv AnnData, then exercises the
full superres_core pipeline: feature pre-extraction → MLP train → checkpoint →
predict. Run twice to verify the on-disk feature cache short-circuits the ViT
on subsequent invocations.

Usage:
    python scripts/smoke_test_superres.py                 # GPU, 16-mixed (default)
    python scripts/smoke_test_superres.py --cpu           # CPU, fp32
    python scripts/smoke_test_superres.py --dinov2 /path/to/dinov2-base
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# Allow running this script directly (e.g. `python scripts/smoke_test_superres.py`)
# without a editable install — make the project root importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

import anndata as ad


CELLTYPES = ["T", "B", "NK", "Macro", "Fibro", "Endo", "Tumor", "Epith", "Stem"]


def _save_synthetic_image(img_path: str, img_size: int = 600, seed: int = 0):
    """Write a deterministic random H&E-like image. Call ONCE per smoke run so
    the file's mtime/size — which feed the cache key — stay stable across the
    two pipeline invocations."""
    rng = np.random.default_rng(seed)
    img = (rng.random((img_size, img_size, 3)) * 255).astype("uint8")
    Image.fromarray(img).save(img_path)


def _build_fake_dataset(n_spots: int = 20, img_size: int = 600, radius: int = 30, seed: int = 0) -> ad.AnnData:
    """Build a small synthetic deconv AnnData (no image I/O)."""
    rng = np.random.default_rng(seed + 1)
    margin = radius * 3
    xs = rng.integers(margin, img_size - margin, size=n_spots)
    ys = rng.integers(margin, img_size - margin, size=n_spots)
    spatial = np.stack([xs, ys], axis=1).astype("float64")
    probs = rng.dirichlet(np.ones(len(CELLTYPES)), size=n_spots)

    adata = ad.AnnData(np.zeros((n_spots, 1)))
    adata.obs_names = [f"spot_{i}" for i in range(n_spots)]
    adata.obsm["spatial"] = spatial
    for i, ct in enumerate(CELLTYPES):
        adata.obs[ct] = probs[:, i]
    adata.uns["celltype"] = CELLTYPES
    adata.uns["radius"] = radius
    return adata


def _run_once(
    adata: ad.AnnData, img_path: str, cache_dir: str, accelerator: str, precision: str, dinov2_path: str, epoch: int = 3
) -> tuple[float, "ad.AnnData", str]:
    """Invoke superres_core once and return (elapsed_seconds, sr_adata, cache_path)."""
    from panospace._core.annotation.superres import superres_core

    t0 = time.time()
    sr = superres_core(
        deconv_adata=adata,
        img_dir=img_path,
        neighb=2,
        radius=adata.uns["radius"],
        epoch=epoch,
        batch_size=8,
        cache_dir=cache_dir,
        local_path=dinov2_path,
        accelerator=accelerator,
        precision=precision,
        seed=42,
    )
    return time.time() - t0, sr, cache_dir


def _assert_outputs(sr: "ad.AnnData", n_spots: int):
    """Sanity-check the predicted super-resolution AnnData."""
    assert sr.n_obs > n_spots, f"Expected subspot count > spot count ({n_spots}), got {sr.n_obs}"
    proba = sr.obs[CELLTYPES].to_numpy()
    assert np.isfinite(proba).all(), "Predictions contain NaN/Inf"
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), (
        f"Per-subspot probabilities must sum to 1; got range " f"[{row_sums.min()}, {row_sums.max()}]"
    )
    assert (proba >= 0).all(), "Probabilities must be non-negative"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU + fp32 (default: GPU + 16-mixed if CUDA available)"
    )
    parser.add_argument(
        "--dinov2", default=os.path.expanduser("~/.panospace_cache/dinov2-base"), help="Local path to DINOv2 weights"
    )
    parser.add_argument("--n-spots", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=600)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--cache-dir", default=None, help="Cache dir (default: a fresh tempdir)")
    args = parser.parse_args()

    import torch

    use_cpu = args.cpu or not torch.cuda.is_available()
    accelerator = "cpu" if use_cpu else "gpu"
    precision = "32" if use_cpu else "16-mixed"

    if not Path(args.dinov2).exists():
        print(
            f"ERROR: DINOv2 weights not found at {args.dinov2}. " f"Pass --dinov2 /path/to/dinov2-base", file=sys.stderr
        )
        sys.exit(2)

    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="panospace_smoke_")
    print(f"cache_dir = {cache_dir}")
    print(f"accelerator = {accelerator}, precision = {precision}")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
    try:
        # Write the synthetic image ONCE so its mtime/size (which feed the
        # cache key) stay identical across both pipeline passes.
        _save_synthetic_image(img_path, img_size=args.img_size, seed=0)
        adata = _build_fake_dataset(n_spots=args.n_spots, img_size=args.img_size, radius=30, seed=0)
        print(f"synthetic: {args.n_spots} spots, image {args.img_size}x{args.img_size}")

        # First pass: must extract features + train + predict.
        elapsed1, sr1, _ = _run_once(adata, img_path, cache_dir, accelerator, precision, args.dinov2, epoch=args.epoch)
        print(f"[pass 1] full pipeline OK in {elapsed1:.1f}s; subspots = {sr1.n_obs}")
        _assert_outputs(sr1, args.n_spots)

        # Locate the per-config cache subdir where features.pt / ckpt live.
        # Discovered rather than recomputed: CacheManager's key includes
        # implementation details (grid pitch, mask settings, ...) that this
        # test has no business mirroring.
        feats = sorted(Path(cache_dir).glob("*/features.pt"))
        assert len(feats) == 1, f"Expected exactly one cache dir, found {feats}"
        cache_path = str(feats[0].parent)
        feat_path = str(feats[0])
        ckpt_path = os.path.join(cache_path, "superres_model.ckpt")
        assert os.path.exists(ckpt_path), f"Missing {ckpt_path}"
        print(
            f"[pass 1] features.pt ({os.path.getsize(feat_path)//1024} KB), "
            f"superres_model.ckpt ({os.path.getsize(ckpt_path)//1024} KB)"
        )

        # Second pass with the SAME cache_dir + image: features load from disk,
        # existing checkpoint skips retraining → predictions must match pass 1.
        adata2 = _build_fake_dataset(n_spots=args.n_spots, img_size=args.img_size, radius=30, seed=0)
        elapsed2, sr2, _ = _run_once(adata2, img_path, cache_dir, accelerator, precision, args.dinov2, epoch=args.epoch)
        print(f"[pass 2] cached pipeline OK in {elapsed2:.1f}s " f"(should be much faster than pass 1)")
        _assert_outputs(sr2, args.n_spots)

        p1 = sr1.obs[CELLTYPES].to_numpy()
        p2 = sr2.obs[CELLTYPES].to_numpy()
        assert np.allclose(p1, p2, atol=1e-4), f"Cached-run predictions diverged: max diff {np.abs(p1-p2).max()}"

        print("\nALL_OK: superres pipeline train→ckpt→predict→cache verified.")
    finally:
        try:
            os.unlink(img_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
