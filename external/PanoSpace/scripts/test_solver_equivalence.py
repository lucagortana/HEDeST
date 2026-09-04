#!/usr/bin/env python
"""Check that the OR-Tools flow solver reproduces the MILP optimum.

The fork replaces upstream's MILP with a min-cost flow because the MILP does not
scale to a whole slide (see README, "Scalability"). The two formulations are
mathematically the same problem, and this script checks that empirically: it
builds synthetic slides with the same geometry as a real one (disjoint spot
discs, a sub-spot lattice, nuclei inside and outside the spots), solves the
assignment with both backends, and compares the objective values.

Usage:
    python scripts/test_solver_equivalence.py

Requires `pyscipopt` (the MILP reference); it is installed by setup_env.sh.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.WARNING)
from panospace._core.annotation.annotator import annotator_core

K = 3
CT = ["A", "B", "C"]


def build(n_spots_side=4, radius=10, n_seg=200, seed=0):
    rng = np.random.default_rng(seed)
    # spots on a coarse grid, pitch 3*radius so their discs are disjoint
    pitch = 3 * radius
    sc_ = np.array(
        [
            (x, y)
            for x in range(pitch, pitch * (n_spots_side + 1), pitch)
            for y in range(pitch, pitch * (n_spots_side + 1), pitch)
        ],
        float,
    )
    spot = ad.AnnData(np.ones((len(sc_), 1)))
    spot.obsm["spatial"] = sc_
    P = rng.dirichlet(np.ones(K), size=len(sc_))
    for i, c in enumerate(CT):
        spot.obs[c] = P[:, i]
    spot.uns["celltype"] = CT
    spot.uns["radius"] = radius

    # SR grid covering everything, pitch = radius
    lo, hi = 0, pitch * (n_spots_side + 2)
    g = np.array([(x, y) for x in range(lo, hi, radius) for y in range(lo, hi, radius)], float)
    sr = ad.AnnData(np.zeros((len(g), 1)))
    sr.obsm["spatial"] = g
    Q = rng.dirichlet(np.ones(K), size=len(g))
    for i, c in enumerate(CT):
        sr.obs[c] = Q[:, i]

    # segments: half inside spots, half scattered
    inside = []
    for s in sc_:
        for _ in range(rng.integers(2, 7)):
            th, rr = rng.uniform(0, 2 * np.pi), radius * np.sqrt(rng.uniform(0, 0.8))
            inside.append(s + [rr * np.cos(th), rr * np.sin(th)])
    inside = np.array(inside)
    outside = rng.uniform(lo, hi - 1, size=(n_seg, 2))
    # drop scattered points that landed inside a spot disc
    d = np.linalg.norm(outside[:, None, :] - sc_[None, :, :], axis=2)
    outside = outside[(d > radius * 1.2).all(axis=1)]
    pts = np.vstack([inside, outside])
    seg = ad.AnnData(np.ones((len(pts), 1)))
    seg.obsm["spatial"] = pts
    seg.obs_names = [str(i) for i in range(len(pts))]
    seg.obs["img_type"] = rng.integers(0, 6, size=len(pts))
    return spot, sr, seg


for seed in range(3):
    spot, sr, seg = build(seed=seed)
    res = {}
    for solver in ("flow", "scip"):
        p, a = annotator_core(spot, sr, seg.copy(), alpha=0.3, solver=solver)
        # recompute the score matrix exactly as infer_cell_types does
        sr2seg = a._affil_sr2seg_norm_csr.transpose().tocsr()
        sc_scores = np.asarray(sr2seg @ a.sr_ct_ratios)
        morph = np.asarray(a._seg_imgtype_onehot_csr @ a.type_transfer_prop.T)
        S = 0.7 * sc_scores + 0.3 * morph
        onehot = p.obs[a.cell_types].to_numpy()
        res[solver] = dict(
            obj=float((S * onehot).sum()), n=p.n_obs, labels=p.obs["pred_cell_type"].to_numpy(), quota=onehot.sum(0)
        )
    f, s = res["flow"], res["scip"]
    same = (f["labels"] == s["labels"]).mean()
    print(
        f"seed={seed}  n={f['n']:4d}  obj_flow={f['obj']:.8f}  obj_scip={s['obj']:.8f}  "
        f"delta={abs(f['obj']-s['obj']):.2e}  identical_labels={same:.3f}  "
        f"quota_match={bool((f['quota']==s['quota']).all())}"
    )
    assert abs(f["obj"] - s["obj"]) < 1e-6 * max(1.0, abs(s["obj"])), "objective mismatch!"
print("\nOK: the flow solver reaches the same optimum as the SCIP MILP.")
