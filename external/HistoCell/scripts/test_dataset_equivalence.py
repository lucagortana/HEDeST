#!/usr/bin/env python
"""Check that `SlideTileDataset` reproduces upstream's `TileBatchDataset`.

The benchmark adapter cuts tiles in memory from a whole-slide image instead of
reading a directory of pre-cut tiles plus one HoVer-Net json per tile.  That is
only legitimate if the tensors the model receives are the same ones.  This
script builds both:

    slide + whole-slide hovernet.json --> SlideTileDataset      (the adapter)
    the same tiles written to disk    --> TileBatchDataset      (upstream)

and compares every field of every item, elementwise.  Augmentation is disabled
on both sides, since ColorJitter/RandomGrayscale are random by construction.

    python scripts/test_dataset_equivalence.py \
        --he .../he.tiff --st .../pseudovisium.h5ad \
        --seg-dict .../hovernet.json --n-tiles 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench  # noqa: E402
from data import SlideTileDataset, TileBatchDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--he", required=True)
    ap.add_argument("--st", required=True)
    ap.add_argument("--seg-dict", required=True)
    ap.add_argument("--n-tiles", type=int, default=24)
    ap.add_argument("--max-cell-num", type=int, default=256)
    args = ap.parse_args()

    import logging

    logging.basicConfig(level=logging.WARNING)

    nuclei, _ = bench.load_seg_dict(args.seg_dict)
    adata = bench.load_st_adata(args.st)
    coords = bench.spot_coordinates(adata)
    tile_px = bench.infer_tile_size(adata)
    size = int(round(tile_px))

    tiles = bench.spot_tiles(coords, tile_px)
    members = bench.assign_nuclei(nuclei, tiles, tile_px)
    keep = [t for t in tiles.index if len(members[t]) > 0][: args.n_tiles]
    tiles = tiles.loc[keep]
    members = {t: members[t] for t in keep}

    tcs = bench.make_tissue_compartment(["A", "B", "C"])
    reader = bench.SlideReader(args.he)

    mine = SlideTileDataset(
        reader,
        tiles,
        nuclei,
        members,
        tcs,
        proportions=None,
        tile_px=tile_px,
        aug=False,
        max_cell_num=args.max_cell_num,
    )

    # Write the very same tiles the way upstream expects to find them: the tile
    # image already resized to 256 (upstream segmented 256 px tiles, so its json
    # coordinates live in 256-space), and one json per tile in that same space.
    with tempfile.TemporaryDirectory() as tmp:
        sample = "sample"
        tdir = os.path.join(tmp, "tiles", sample)
        mdir = os.path.join(tmp, "seg", sample, "json")
        os.makedirs(tdir), os.makedirs(mdir)
        tcs_path = os.path.join(tmp, "tcs.json")
        with open(tcs_path, "w") as fh:
            json.dump(tcs, fh)

        for tid in keep:
            x0, y0 = (int(v) for v in tiles.loc[tid, ["x0", "y0"]])
            crop = reader.crop(x0, y0, size)
            pic = Image.fromarray(crop).convert("RGB").resize((256, 256))
            pic.save(os.path.join(tdir, f"{tid}.png"))
            sx, sy = 256.0 / crop.shape[1], 256.0 / crop.shape[0]
            nuc = {}
            for j, row in enumerate(members[tid]):  # same order as the adapter
                r0, c0, r1, c1 = nuclei.iloc[row][["r0", "c0", "r1", "c1"]]
                cx, cy = nuclei.iloc[row][["cx", "cy"]]
                box = [
                    [int(min(max(round((r0 - y0) * sy), 0), 256)), int(min(max(round((c0 - x0) * sx), 0), 256))],
                    [int(min(max(round((r1 - y0) * sy), 0), 256)), int(min(max(round((c1 - x0) * sx), 0), 256))],
                ]
                nuc[str(j)] = {
                    "bbox": box,
                    "centroid": [(cx - x0) * sx, (cy - y0) * sy],
                    "type": int(nuclei.iloc[row]["htype"]),
                }
            with open(os.path.join(mdir, f"{tid}.json"), "w") as fh:
                json.dump({"mag": None, "nuc": nuc}, fh)

        theirs = TileBatchDataset(
            os.path.join(tmp, "tiles"),
            os.path.join(tmp, "seg"),
            tcs_path,
            cell_dir=None,
            prefix=[sample],
            aug=False,
            val=True,
            ext="png",
            max_cell_num=args.max_cell_num,
        )

        # upstream shuffles its file list, so match on the tile name
        theirs_by_name = {}
        for i in range(len(theirs)):
            item = theirs[i]
            theirs_by_name[item["name"].split("_", 1)[1]] = item

        fields = ["tissue", "image", "mask", "size", "adj", "cell_coords", "cell_types"]
        worst = {f: 0.0 for f in fields}
        n = 0
        for i in range(len(mine)):
            a = mine[i]
            b = theirs_by_name[a["name"]]
            for f in fields:
                x, y = a[f], b[f]
                if not torch.is_tensor(x):
                    assert x == y, f"{a['name']}.{f}: {x} != {y}"
                    continue
                d = float((x.to(torch.float64) - y.to(torch.float64)).abs().max())
                worst[f] = max(worst[f], d)
            n += 1

        print(f"compared {n} tiles from {os.path.basename(args.he)}")
        for f in fields:
            print(f"  max |adapter - upstream|  {f:<12s} {worst[f]:.3e}")
        bad = [f for f, d in worst.items() if d > 0]
        if bad:
            print(f"\nFAIL: {bad} differ")
            sys.exit(1)
        print("\nOK: every tensor the model receives is bit-identical.")


if __name__ == "__main__":
    main()
