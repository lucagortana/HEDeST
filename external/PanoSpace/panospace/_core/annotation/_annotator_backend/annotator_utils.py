from __future__ import annotations

import logging
import os
import time

import numpy as np
import ot  # POT for OT (EMD / Sinkhorn)
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# ---------- SCIP availability check ----------
def probe_scip() -> bool:
    """
    Safely check whether SCIP is usable.

    Returns
    -------
    bool
        True if SCIP is usable.
    """

    from pyscipopt import Model, SCIP_RESULT

    # Simple test: create a model and add a variable
    model = Model("probe_test")
    model.setParam("display/verblevel", 0)
    x = model.addVar("x", vtype="B")
    model.setObjective(x, sense="maximize")
    return True


try:
    _SCIP_AVAILABLE = probe_scip()
except Exception as e:
    logger.info(f"SCIP probe failed: {str(e)}")
    _SCIP_AVAILABLE = False

if _SCIP_AVAILABLE:
    from pyscipopt import Model, SCIP_RESULT, quicksum
# else:
#     logger.info("SCIP disabled: PySCIPOpt not installed or SCIP not available.")


def probe_gurobi() -> bool:
    """
    Safely check whether Gurobi is usable.

    Returns
    -------
    bool
        True if Gurobi is usable and license is valid.
    """
    env = None
    model = None
    try:
        import gurobipy as gp

        # --- Version check ---
        major, minor, *_ = gp.gurobi.version()
        if major < 10:
            return False

        # --- Minimal model lifecycle test ---
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()

        model = gp.Model(env=env)
        model.dispose()
        env.dispose()
        return True

    except Exception as e:
        logger.info(f"Gurobi probe exception: {str(e)}")
        return False

    finally:

        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass

        if env is not None:
            try:
                env.close() if hasattr(env, "close") else env.dispose()
            except Exception:
                pass


try:
    _GUROBI_AVAILABLE = probe_gurobi()
except Exception as e:
    logger.info(f"Gurobi probe failed: {str(e)}")
    _GUROBI_AVAILABLE = False

if _GUROBI_AVAILABLE:
    import gurobipy as gp
    from gurobipy import GRB
# else:
#     logger.info("Gurobi disabled: probe failed or license invalid.")

import warnings


def probe_ortools() -> bool:
    """Check whether the OR-Tools min-cost-flow solver is usable."""
    from ortools.graph.python import min_cost_flow  # noqa: F401

    return True


try:
    _ORTOOLS_AVAILABLE = probe_ortools()
except Exception as e:  # pragma: no cover - depends on the environment
    logger.info(f"OR-Tools probe failed: {str(e)}")
    _ORTOOLS_AVAILABLE = False

if _ORTOOLS_AVAILABLE:
    from ortools.graph.python import min_cost_flow as _ortools_mcf


from ...._utils.utils import radius_membership_sparse


class CellTypeAnnotator:
    """
    CellTypeAnnotator

    Assigns a cell type to each segmented unit (segment/nucleus) by combining:
      • deconvolved cell-type proportions at spot level and super-resolved spot level,
      • spatial memberships between spots/SR-spots and segments,
      • optional morphology priors from image-derived categories, and
      • a global assignment solved via Optimal Transport (OT) followed by either
        a convex relaxation (QP + rounding) or an exact MILP when available.

    The result is a globally consistent mapping from segments to cell types that
    respects local proportions and optional morphology preferences.
    """

    def __init__(
        self,
        spot_adata,
        sr_spot_adata,
        seg_adata,
        priori_type_affinities=None,
        alpha=0.3,
        ot_mode="emd",  # "sinkhorn" or "emd"
        sinkhorn_reg=0.01,  # Sinkhorn entropy regularization
        _global_quota=True,  # Whether to enforce global quotas in MILP
        _spot_quota=True,  # Whether to enforce spot-level quotas in MILP
        solver="auto",  # "auto" | "flow" | "gurobi" | "scip"
        dedup_overlapping_spots=False,
    ):
        """
        Parameters
        ----------
        spot_adata : AnnData
            Spot-level deconvolution. Requirements:
            - .uns['celltype'] : list of cell type names
            - .uns['radius']   : spot radius
            - .obsm['spatial'] : spot coordinates (S x 2)
            - .obs[ctype]      : a column for each cell type with its proportion
        sr_spot_adata : AnnData
            Super-resolved spot-level deconvolution. Same columns as above; .obsm['spatial'] required.
        seg_adata : AnnData
            Segmented units (segments or nuclei). Requires .obsm['spatial'].
            Optionally .obs['img_type'] with integer morphology labels enables the morphology branch ("mor" mode).
        priori_type_affinities : dict[str, list[str]] or None
            Optional prior: morphology-category name → preferred cell types.
        alpha : float
            Fusion weight. When morphology is active, blends (SR spatial propagation) with (morphology prior via OT).
        ot_mode : str
            "sinkhorn" for fast/stable entropic OT or "emd" for exact discrete OT.
        sinkhorn_reg : float
            Entropic regularization for Sinkhorn.

        Notes
        -----
        The final assignment is solved via MILP using:
          - Gurobi (if available, fastest)
          - SCIP (open-source fallback)
        Both solvers produce mathematically equivalent results with spot-level quota constraints.
        """
        self.spot_adata = spot_adata
        self.sr_spot_adata = sr_spot_adata
        self.seg_adata = seg_adata
        self.alpha = float(alpha)
        logger.info(f"CellTypeAnnotator alpha: {self.alpha}")
        self.ot_mode = ot_mode
        logger.info(f"CellTypeAnnotator OT mode: {self.ot_mode}")
        self.sinkhorn_reg = float(sinkhorn_reg)
        logger.info(f"CellTypeAnnotator Sinkhorn reg: {self.sinkhorn_reg}")

        self._global_quota = bool(_global_quota)
        logger.info(f"CellTypeAnnotator global_quota: {self._global_quota}")
        self._spot_quota = bool(_spot_quota)
        logger.info(f"CellTypeAnnotator spot_quota: {self._spot_quota}")

        self.solver = str(solver).lower()
        if self.solver not in {"auto", "flow", "gurobi", "scip"}:
            raise ValueError(f"Unknown solver {solver!r}; expected 'auto', 'flow', 'gurobi' or 'scip'.")
        self.dedup_overlapping_spots = bool(dedup_overlapping_spots)

        # Log solver availability
        logger.info(
            f"CellTypeAnnotator solver request={self.solver} "
            f"(available: flow/OR-Tools={_ORTOOLS_AVAILABLE}, Gurobi={_GUROBI_AVAILABLE}, SCIP={_SCIP_AVAILABLE})"
        )

        # Mode: enable morphology branch if seg_adata has img_type labels
        self.mode = "mor" if "img_type" in self.seg_adata.obs.columns else None
        logger.info(f"CellTypeAnnotator mode: {self.mode if self.mode else 'default (no morphology)'}")

        # Cell types and column names
        self.cell_types = list(self.spot_adata.uns["celltype"])
        logger.info(f"Cell types: {self.cell_types}")
        self.ct_cols = self.cell_types

        # Super-resolved spot proportions (S_sr x K), row-normalized
        self.sr_ct_ratios = self._safe_row_normalize(self.sr_spot_adata.obs[self.ct_cols].to_numpy(copy=True))

        # Spot proportions (S x K), clipped to nonnegative then row-normalized
        self.spot_ct_ratios = self._safe_row_normalize(
            np.clip(self.spot_adata.obs[self.ct_cols].to_numpy(copy=True), 0, None)
        )

        # Spatial radius
        self.radius = float(self.spot_adata.uns["radius"])
        logger.info(f"CellTypeAnnotator spatial radius: {self.radius}")

        # Prior morphology→cell-type affinities
        self.priori_type_affinities = priori_type_affinities or {}
        self.img_type_names = list(self.priori_type_affinities.keys()) if self.priori_type_affinities else []
        logger.info(f"CellTypeAnnotator priori_type_affinities: {self.priori_type_affinities}")

        # Cache spatial coordinates (NumPy)
        self._sr_spatial = np.asarray(self.sr_spot_adata.obsm["spatial"])
        self._spot_spatial = np.asarray(self.spot_adata.obsm["spatial"])
        self._seg_spatial = np.asarray(self.seg_adata.obsm["spatial"])

        # Sparse matrix caches (filled later)
        self._affil_sr2seg_csr = None  # (S_sr x Nseg)
        self._affil_spot2seg_csr = None  # (S x Nseg)
        self._affil_sr2seg_norm_csr = None  # row-normalized SR-spot→segment
        self._affil_sr2seg_csc = None  # optional transpose/format cache
        self._affil_spot2seg_csr_T = None  # transpose cache if needed

        # Morphology one-hot and aggregates
        self._imgtype_keep_labels = None
        # self._seg_imgtype_labels = None    # shape (Nseg,)
        self._seg_imgtype_onehot_csr = None  # (Nseg x G_kept)
        self._mortype_in_spot = None  # (G_kept x S_nonzero)
        self._imgtype_ratio = None  # (G_kept, ) global normalized vector

        # Counts and quotas
        self._cell_counts_per_spot = None  # (S, )
        self._int_ct_ratios_per_spot = None  # (S_nonzero x K) integerized per-spot quotas
        self._global_ct_quota = None  # (K, ) global integer quota across all segments
        self._nonzero_spot_mask = None  # (S, ) bool

    # ---------- Utilities ----------

    @staticmethod
    def _safe_row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x[x < 0] = 0.0
        s = x.sum(axis=1, keepdims=True)
        s[s < eps] = 1.0
        return x / s

    @staticmethod
    def _integerize_proportions(p: np.ndarray, total: int) -> np.ndarray:
        """Discretize a proportion vector p into integers that sum exactly to `total` (auto-normalizes and handles zeros)."""
        p = np.maximum(np.asarray(p, dtype=float), 0.0)
        if p.sum() <= 0:
            warnings.warn(
                "Proportion vector sums to zero or is entirely non-positive. " "Falling back to uniform proportions.",
                RuntimeWarning,
                stacklevel=2,
            )
            p = np.full_like(p, 1.0 / len(p))
        else:
            p = p / p.sum()
        raw = p * total
        flo = np.floor(raw).astype(int)
        deficit = int(total - flo.sum())
        if deficit > 0:
            resid = raw - flo
            idx = np.argpartition(-resid, deficit - 1)[:deficit]
            flo[idx] += 1
        return flo

    # ---------- Preprocessing: filtering & affiliation matrices ----------

    def filter_and_build_affiliations(self):
        """
        Steps:
        1) Use SR-spot coverage to filter segments (batch inclusion).
        2) Build SR-spot→segment and spot→segment sparse membership matrices (CSR/CSC).
        3) Row-normalize SR-spot→segment for propagating SR proportions.
        """
        # Initial SR-spot coverage for filtering
        affil_sr2seg = radius_membership_sparse(
            base_points=self._sr_spatial, query_points=self._seg_spatial, r=self.radius, metric="chebyshev"
        )
        # Keep segments covered by at least one SR-spot
        keep_mask_seg = affil_sr2seg.getnnz(axis=1) != 0
        self.seg_adata = self.seg_adata[keep_mask_seg].copy()
        self._seg_spatial = np.asarray(self.seg_adata.obsm["spatial"])

        # Recompute memberships after filtering; adopt the (S_* x Nseg) shape convention
        affil_sr2seg = radius_membership_sparse(
            self._sr_spatial, self._seg_spatial, r=self.radius, metric="chebyshev"
        ).transpose()  # (S_sr x Nseg)
        affil_spot2seg = radius_membership_sparse(
            self._spot_spatial, self._seg_spatial, r=self.radius, metric="euclidean"
        ).transpose()  # (S x Nseg)

        # Cache CSR
        self._affil_sr2seg_csr = affil_sr2seg.tocsr()  # (S_sr x Nseg)
        self._affil_spot2seg_csr = affil_spot2seg.tocsr()  # (S x Nseg)

        # A segment covered by two spots would appear in two spot-level equality
        # constraints at once, which makes the assignment model over-determined
        # (upstream only warns, then lets the MILP fail).  Keep the nearest spot
        # for those segments so every downstream quantity -- per-spot counts,
        # integer quotas, and the MILP/flow model -- stays consistent.
        # Visium-like geometries never trigger this (spot discs are disjoint).
        if self.dedup_overlapping_spots:
            self._affil_spot2seg_csr = self._dedup_spot_membership(self._affil_spot2seg_csr)
        # self._affil_spot2seg_csr_T = self._affil_spot2seg_csr.transpose().tocsr()  # (Nseg x S)

        # Row-normalize SR-spot→segment
        A = self._affil_sr2seg_csr.copy().astype(np.float32)  # or float64
        row_sums = A.sum(axis=1).A1

        inv = np.divide(1.0, row_sums, out=np.zeros_like(row_sums, dtype=A.data.dtype), where=row_sums != 0)

        A.data *= inv.repeat(np.diff(A.indptr)).astype(A.data.dtype, copy=False)
        self._affil_sr2seg_norm_csr = A  # (S_sr x Nseg), row-normalized

        # Build morphology one-hot and aggregates when available
        if self.mode == "mor":
            logger.info("Building morphology one-hot and spot aggregates...")
            self._build_imgtype_onehot_and_spot_aggregates()

    def _dedup_spot_membership(self, A):
        """Keep at most one spot per segment (the nearest one) in ``A`` (S x Nseg)."""
        cover_per_seg = np.asarray(A.sum(axis=0)).ravel()
        multi = np.nonzero(cover_per_seg > 1)[0]
        if multi.size == 0:
            return A

        logger.warning(
            "%d segment(s) fall inside more than one spot; keeping the nearest spot for each.",
            multi.size,
        )
        A = A.tocoo()
        rows, cols = A.row, A.col
        multi_set = set(multi.tolist())
        # squared distance from each (spot, segment) pair, only where needed
        keep = np.ones(rows.size, dtype=bool)
        interesting = np.array([c in multi_set for c in cols], dtype=bool)
        if interesting.any():
            r_i, c_i = rows[interesting], cols[interesting]
            d2 = np.sum((self._spot_spatial[r_i] - self._seg_spatial[c_i]) ** 2, axis=1)
            order = np.lexsort((d2, c_i))
            seen = set()
            drop_local = np.zeros(r_i.size, dtype=bool)
            for pos in order:
                c = c_i[pos]
                if c in seen:
                    drop_local[pos] = True
                else:
                    seen.add(c)
            idx_interesting = np.nonzero(interesting)[0]
            keep[idx_interesting[drop_local]] = False

        A = sp.csr_matrix((A.data[keep], (rows[keep], cols[keep])), shape=A.shape, dtype=A.dtype)
        return A

    # ---------- Morphology one-hot & spot aggregates ----------

    def _build_imgtype_onehot_and_spot_aggregates(self):
        """
        Build segment-level morphology one-hot (keeping valid classes) and aggregate counts per spot.
        """
        # Mapping of integer labels to names; drop unlabeled and dead cells
        label_dict = {
            0: "nolabel",
            1: "Neoplastic cells",
            2: "Inflammatory",
            3: "Connective/Soft tissue cells",
            4: "Dead Cells",
            5: "Epithelial",
        }
        remove = {"nolabel", "Dead Cells"}
        keep = [v for v in label_dict.values() if v not in remove]
        # Name → compact column index
        keep_name2col = {name: i for i, name in enumerate(keep)}
        # Original integer labels
        raw = np.asarray(self.seg_adata.obs["img_type"]).astype(int)
        # Map to names
        mapped_names = np.vectorize(label_dict.get)(raw)
        valid_mask = np.array([name in keep_name2col for name in mapped_names], dtype=bool)
        # One-hot (Nseg x G_kept)
        rows = np.nonzero(valid_mask)[0]
        cols = np.fromiter((keep_name2col[n] for n in mapped_names[valid_mask]), count=rows.size, dtype=int)
        data = np.ones_like(rows, dtype=np.int32)
        nseg = self.seg_adata.n_obs
        gk = len(keep_name2col)
        onehot = sp.csr_matrix((data, (rows, cols)), shape=(nseg, gk))
        self._seg_imgtype_onehot_csr = onehot
        self._imgtype_keep_labels = keep
        # self._seg_imgtype_labels = raw  # optional cache

        # Aggregate to spots: (#S x #Nseg) @ (#Nseg x #G_kept) = (#S x #G_kept)
        spot_morph_counts = (self._affil_spot2seg_csr @ onehot).astype(np.float64)  # (S x G_kept)
        # Consider only spots containing at least one segment
        self._cell_counts_per_spot = np.asarray(self._affil_spot2seg_csr.sum(axis=1)).ravel().astype(int)
        self._nonzero_spot_mask = self._cell_counts_per_spot != 0
        if not np.any(self._nonzero_spot_mask):
            raise ValueError("No spots contain any segments after filtering.")

        spot_morph_counts = spot_morph_counts[self._nonzero_spot_mask, :]  # (S_nz x G_kept)
        # Transpose to (G_kept x S_nz) to align with OT code
        self._mortype_in_spot = spot_morph_counts.T.toarray()  # dense for OT
        # Global morphology proportions (G_kept,)
        total = np.sum(self._mortype_in_spot)
        if total <= 0:
            # Degenerate case: no valid morphology; fall back to uniform
            self._imgtype_ratio = np.full(self._mortype_in_spot.shape[0], 1.0 / self._mortype_in_spot.shape[0])
        else:
            self._imgtype_ratio = self._mortype_in_spot.sum(axis=1) / total

    # ---------- Counts & integer quotas ----------

    def compute_counts_and_integerize(self):
        """
        Compute number of segments per spot; discretize spot-level cell-type proportions
        into integer per-spot quotas; and compute a global quota across all segments.
        """
        if self._affil_spot2seg_csr is None:
            raise RuntimeError("Call filter_and_build_affiliations() first.")

        # Number of segments per spot
        if self._cell_counts_per_spot is None:
            self._cell_counts_per_spot = np.asarray(self._affil_spot2seg_csr.sum(axis=1)).ravel().astype(int)

        self._nonzero_spot_mask = self._cell_counts_per_spot != 0
        if not np.any(self._nonzero_spot_mask):
            raise ValueError("No non-empty spots found.")

        # Proportions and counts for non-empty spots
        spot_ratios_nz = self.spot_ct_ratios[self._nonzero_spot_mask, :]  # (S_nz x K)
        counts_nz = self._cell_counts_per_spot[self._nonzero_spot_mask]  # (S_nz, )

        # Integerize each spot's cell-type quota V (row sums are exact)
        int_ratios_list = [
            self._integerize_proportions(spot_ratios_nz[s, :], int(counts_nz[s]))
            for s in range(spot_ratios_nz.shape[0])
        ]
        self._int_ct_ratios_per_spot = np.vstack(int_ratios_list)  # (S_nz x K)

        # Global quotas: integerize based on aggregated per-spot counts
        total_segments = self.seg_adata.n_obs
        global_ratio = self._int_ct_ratios_per_spot.sum(axis=0).astype(float)
        if global_ratio.sum() <= 0:
            global_ratio = np.full_like(global_ratio, 1.0 / len(global_ratio))
        else:
            global_ratio = global_ratio / global_ratio.sum()
        self._global_ct_quota = self._integerize_proportions(global_ratio, total_segments)  # (K,)

    # ---------- OT: align cell types ↔ morphology categories ----------

    def build_type_transfer(self, factor: float = 2.0):
        """
        Align cell types and morphology categories via OT using their distributions across spots.

        Cost:
          • cosine distance between (K x S_nz) and (G x S_nz) signatures.
        Priors:
          • if priori_type_affinities is provided, down-weight preferred pairs by dividing cost by `factor`.
        Output:
          • self.type_transfer_prop: (K x G) column-normalized transport plan, interpreted as
            P(cell type | morphology category).
        """
        if self._mortype_in_spot is None and self.mode == "mor":
            raise RuntimeError("Morphology aggregates not built. Call filter_and_build_affiliations() first.")

        # Signatures across spot space
        ct_signature = self._int_ct_ratios_per_spot.T.astype(float)  # (K x S_nz)
        morph_signature = self._mortype_in_spot.astype(float)  # (G x S_nz)

        # Cosine distance cost (K x G)
        cost = ot.dist(ct_signature, morph_signature, metric="cosine")
        logger.debug("cost:\n" + str(pd.DataFrame(cost, index=self.cell_types, columns=self._imgtype_keep_labels)))

        # Apply prior affinities by reducing cost for preferred pairs
        if self.priori_type_affinities:
            adjusted = cost.copy()
            for g_idx, g_name in enumerate(self._imgtype_keep_labels):
                prefer_cts = self.priori_type_affinities.get(g_name, [])
                for ct_name in prefer_cts:
                    if ct_name in self.cell_types:
                        k = self.cell_types.index(ct_name)
                        adjusted[k, g_idx] /= float(factor)
            cost = adjusted

        # Source and target distributions
        ct_ratio_global = self._global_ct_quota.astype(float)
        if ct_ratio_global.sum() <= 0:
            ct_ratio_global = np.full_like(ct_ratio_global, 1.0 / len(ct_ratio_global))
        else:
            ct_ratio_global = ct_ratio_global / ct_ratio_global.sum()

        morph_ratio_global = self._imgtype_ratio

        # Solve OT
        if self.ot_mode == "sinkhorn":
            logger.debug("Using Sinkhorn for OT...")
            logger.debug("ct_ratio_global:\n" + str(pd.Series(ct_ratio_global, index=self.cell_types)))
            logger.debug("morph_ratio_global:\n" + str(pd.Series(morph_ratio_global, index=self._imgtype_keep_labels)))
            gamma = ot.sinkhorn(ct_ratio_global, morph_ratio_global, cost, reg=self.sinkhorn_reg, numItermax=2000)
        else:
            logger.debug("Using exact EMD for OT...")
            logger.debug("ct_ratio_global:\n" + str(pd.Series(ct_ratio_global, index=self.cell_types)))
            logger.debug("morph_ratio_global:\n" + str(pd.Series(morph_ratio_global, index=self._imgtype_keep_labels)))
            gamma = ot.emd(ct_ratio_global, morph_ratio_global, cost, numItermax=1000)

        # Column-normalize to obtain P(cell type | morphology)
        col_sums = gamma.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        self.type_transfer_prop = gamma / col_sums  # (K x G)
        logger.debug(
            "type_transfer_prop:\n"
            + str(pd.DataFrame(self.type_transfer_prop, index=self.cell_types, columns=self._imgtype_keep_labels))
        )

    # ---------- Final assignment: QP relaxation (optional exact MILP) ----------

    def infer_cell_types(self):
        """
        Produce the segment→cell-type assignment.

        Scoring:
          • SR-spot proportions propagated to segments: (_affil_sr2seg_norm_csr @ sr_ct_ratios).
          • If morphology is active: blend with (segment one-hot × OT transfer) using weight `alpha`.

        Optimization:
          • Always solves exact MILP for 0/1 assignment with row=1, col=quota, and spot-level constraints.
          • Uses Gurobi if available (fastest), otherwise SCIP (open-source fallback).
          • Both solvers produce mathematically equivalent results.

        Returns
        -------
        seg_adata_pred : AnnData
            A copy of seg_adata with:
              - obs['pred_cell_type'] : predicted type
              - one-hot columns for each cell type
        """
        if self._affil_sr2seg_norm_csr is None:
            raise RuntimeError("Call filter_and_build_affiliations() and compute_counts_and_integerize() first.")

        # (1) Propagate SR-spot proportions to segments (Nseg x K)
        # Use the transpose of (S_sr x Nseg) to multiply (Nseg x S_sr) @ (S_sr x K)
        sr2seg_norm_T = self._affil_sr2seg_norm_csr.transpose().tocsr()  # (Nseg x S_sr)
        sr_scores = np.asarray(sr2seg_norm_T @ self.sr_ct_ratios)  # dense (Nseg x K)

        # (2) Morphology branch (if active)
        if self.mode == "mor":
            # segment one-hot (Nseg x G) × (G x K); use (K x G)^T
            morph_scores = np.asarray(self._seg_imgtype_onehot_csr @ self.type_transfer_prop.T)
            scores = (1.0 - self.alpha) * sr_scores + self.alpha * morph_scores
        else:
            scores = sr_scores

        nseg, ntypes = scores.shape
        quotas = self._global_ct_quota.copy().astype(int)  # (K,)

        # ---------- Solve ----------
        logger.info(f"nseg={nseg}, ntypes={ntypes}, arcs≈{nseg * ntypes:,}")
        # _GUROBI_AVAILABLE = False
        quo = {k: v for k, v in zip(self.cell_types, quotas)}
        logger.info(f"global quotas: " + ", ".join(f"{k}: {v}" for k, v in quo.items()))
        chosen = self._choose_solver()
        if chosen == "flow":
            logger.info("Using OR-Tools min-cost flow for the exact assignment.")
            X = self._solve_flow(
                scores,
                quotas,
                global_quota=self._global_quota,
                spot_quota=self._spot_quota,
            )
        elif chosen == "gurobi":
            # Exact MILP with Gurobi
            logger.info("Using Gurobi MILP for exact assignment with spot-level quotas.")
            X = self._solve_mip(scores, quotas, global_quota=self._global_quota, spot_quota=self._spot_quota)
        else:
            # Exact MILP with SCIP (open-source fallback)
            logger.warning("Using SCIP MILP for exact assignment with spot-level quotas.")
            logger.info("NOTE!!! SCIP is orders of magnitude slower than the flow solver on whole slides.")
            X = self._solve_mip_scip(scores, quotas)

        # ---------- Write back to AnnData ----------
        max_idx = np.argmax(X, axis=1)
        seg_cp = self.seg_adata.copy()
        seg_cp.obs["pred_cell_type"] = [self.cell_types[i] for i in max_idx]
        # one-hot columns
        for k, ct in enumerate(self.cell_types):
            seg_cp.obs[ct] = (max_idx == k).astype(int)

        return seg_cp

    # ---------- Solver selection ----------

    def _choose_solver(self) -> str:
        """Resolve ``self.solver`` against what is actually installed."""
        available = {
            "flow": _ORTOOLS_AVAILABLE,
            "gurobi": _GUROBI_AVAILABLE,
            "scip": _SCIP_AVAILABLE,
        }
        if self.solver != "auto":
            if not available[self.solver]:
                raise RuntimeError(
                    f"solver='{self.solver}' was requested but is not available "
                    f"(installed: {[k for k, v in available.items() if v]})."
                )
            return self.solver

        # 'auto' reproduces upstream PanoSpace exactly: the published method
        # solves this MILP with Gurobi (Nat Comput Sci 2026, Methods), and the
        # reference implementation falls back to SCIP when Gurobi is absent.
        # The equivalent min-cost-flow reformulation is available but must be
        # asked for explicitly with solver='flow'.
        for name in ("gurobi", "scip"):
            if available[name]:
                return name
        raise RuntimeError(
            "No MILP solver available. Please install either Gurobi or SCIP:\n"
            "  - Gurobi (recommended): pip install gurobipy\n"
            "  - SCIP (open-source): pip install pyscipopt\n"
            "(solver='flow' also solves this model exactly, via OR-Tools.)"
        )

    # ---------- Exact assignment as a min-cost flow ----------

    def _solve_flow(
        self,
        scores: np.ndarray,
        quotas: np.ndarray,
        global_quota: bool = True,
        spot_quota: bool = True,
        cost_scale: float = 1e7,
    ) -> np.ndarray:
        """Exact segment -> cell-type assignment, solved as a min-cost flow.

        This solves *the same* optimisation problem as :meth:`_solve_mip`:

            max  sum_ik  scores[i, k] * X[i, k]
            s.t. sum_k X[i, k] = 1                        for every segment i
                 sum_i X[i, k] = quotas[k]                (global quota)
                 sum_{i in spot s} X[i, k] = V[s, k]      (spot quota)
                 X binary

        Once every segment belongs to at most one spot (guaranteed by
        :meth:`_dedup_spot_membership`), the spot constraints partition the
        covered segments, so the constraint matrix is the incidence matrix of a
        bipartite transportation network: segments on one side, one node per
        (spot, cell type) bucket plus one node per cell type for the segments
        that fall outside every spot.  That network is totally unimodular, so
        the min-cost flow optimum *is* the MILP optimum -- no relaxation, no
        rounding.

        Costs are integers, obtained by scaling ``max(scores) - scores`` by
        ``cost_scale / (max(scores) - min(scores))``.  Shifting by a constant
        is free (every segment ships exactly one unit) and keeps all costs
        non-negative; the residual quantisation is ~1e-7 of the score range.
        """
        if not _ORTOOLS_AVAILABLE:
            raise RuntimeError("OR-Tools is not available; cannot use the flow solver.")

        scores = np.asarray(scores, dtype=float)
        quotas = np.asarray(quotas, dtype=np.int64)
        nseg, ntypes = scores.shape

        if global_quota and quotas.sum() != nseg:
            raise ValueError(f"Invalid quotas: sum(quotas)={quotas.sum()} must equal nseg={nseg}.")

        # --- which spot does each segment belong to (-1 = none)? -------------
        A_csc = self._affil_spot2seg_csr.tocsc()  # (S x Nseg)
        per_seg = np.diff(A_csc.indptr)
        if per_seg.size and per_seg.max() > 1:
            raise RuntimeError(
                "Some segments belong to several spots; the flow model needs a "
                "partition. Re-run with dedup_overlapping_spots=True."
            )
        spot_of_seg = np.full(nseg, -1, dtype=np.int64)
        covered_cols = np.nonzero(per_seg == 1)[0]
        spot_of_seg[covered_cols] = A_csc.indices[A_csc.indptr[covered_cols]]

        nz_mask = np.asarray(self._nonzero_spot_mask, dtype=bool)
        # row index inside _int_ct_ratios_per_spot for every spot (-1 if empty)
        nz_row_of_spot = np.full(nz_mask.size, -1, dtype=np.int64)
        nz_row_of_spot[nz_mask] = np.arange(int(nz_mask.sum()))
        V_nz = np.asarray(self._int_ct_ratios_per_spot, dtype=np.int64)  # (S_nz x K)

        # --- assemble the buckets each segment may be routed to --------------
        # bucket_of_seg[i] = index of the first bucket usable by segment i,
        # the K buckets of a group being contiguous.  -1 marks an unconstrained
        # segment, which simply takes its arg-max type.
        bucket_base = np.full(nseg, -1, dtype=np.int64)
        demands: list = []

        if spot_quota:
            covered = spot_of_seg >= 0
            used_spots = np.unique(spot_of_seg[covered])
            rows_used = nz_row_of_spot[used_spots]
            if np.any(rows_used < 0):
                bad = used_spots[rows_used < 0]
                raise RuntimeError(f"Spot(s) {bad[:5]} cover segments but are flagged empty.")

            V_used = V_nz[rows_used]  # (S_used x K)
            demands = V_used.ravel().astype(np.int64).tolist()
            # bucket group of a segment = rank of its spot among used_spots
            bucket_base[covered] = np.searchsorted(used_spots, spot_of_seg[covered]) * ntypes

            n_covered = int(covered.sum())
            seg_per_spot = np.bincount(spot_of_seg[covered], minlength=nz_mask.size)
            mismatch = np.nonzero(V_used.sum(axis=1) != seg_per_spot[used_spots])[0]
            if mismatch.size:
                i = mismatch[0]
                raise ValueError(
                    f"Spot {used_spots[i]}: quota sum {int(V_used[i].sum())} != "
                    f"{int(seg_per_spot[used_spots[i]])} segments inside it."
                )

            if global_quota:
                V_sum_k = V_used.sum(axis=0).astype(np.int64)
                remaining = quotas - V_sum_k
                if np.any(remaining < 0):
                    bad = np.nonzero(remaining < 0)[0]
                    raise ValueError(
                        "Infeasible: per-spot quotas exceed global quotas for "
                        + ", ".join(f"{self.cell_types[k]}: {V_sum_k[k]} > {quotas[k]}" for k in bad)
                    )
                n_uncovered = nseg - n_covered
                if int(remaining.sum()) != n_uncovered:
                    raise ValueError(
                        f"Infeasible: {n_uncovered} segments lie outside every spot but the "
                        f"remaining global quota sums to {int(remaining.sum())}."
                    )
                free_base = len(demands)
                demands.extend(int(v) for v in remaining)
                bucket_base[~covered] = free_base
            # else: uncovered segments stay unconstrained (bucket_base == -1)
        elif global_quota:
            demands = [int(v) for v in quotas]
            bucket_base[:] = 0
        else:
            # No constraint at all beyond "one type per segment".
            X = np.zeros((nseg, ntypes), dtype=int)
            X[np.arange(nseg), np.argmax(scores, axis=1)] = 1
            return X

        demands_arr = np.asarray(demands, dtype=np.int64)
        n_buckets = demands_arr.size
        constrained = bucket_base >= 0
        n_constrained = int(constrained.sum())
        if int(demands_arr.sum()) != n_constrained:
            raise ValueError(
                f"Infeasible: bucket demands sum to {int(demands_arr.sum())} but "
                f"{n_constrained} segments must be assigned."
            )

        # --- integer costs ----------------------------------------------------
        smax = float(scores.max()) if scores.size else 0.0
        smin = float(scores.min()) if scores.size else 0.0
        span = max(smax - smin, 1e-12)
        scale = float(cost_scale) / span
        cost = np.rint((smax - scores) * scale).astype(np.int64)  # >= 0

        # --- build the network ------------------------------------------------
        seg_idx = np.nonzero(constrained)[0]
        starts = np.repeat(seg_idx, ntypes)
        ends = nseg + np.repeat(bucket_base[seg_idx], ntypes) + np.tile(np.arange(ntypes, dtype=np.int64), seg_idx.size)
        caps = np.ones(starts.size, dtype=np.int64)
        costs = cost[seg_idx].ravel()

        supplies = np.zeros(nseg + n_buckets, dtype=np.int64)
        supplies[seg_idx] = 1
        supplies[nseg:] = -demands_arr

        logger.info(
            "min-cost flow: %d segments (%d constrained), %d buckets, %d arcs",
            nseg,
            n_constrained,
            n_buckets,
            starts.size,
        )
        t0 = time.time()
        smcf = _ortools_mcf.SimpleMinCostFlow()
        arcs = smcf.add_arcs_with_capacity_and_unit_cost(starts, ends, caps, costs)
        smcf.set_nodes_supplies(np.arange(nseg + n_buckets, dtype=np.int64), supplies)
        status = smcf.solve()
        logger.info("min-cost flow solved in %.2f s", time.time() - t0)
        if status != smcf.OPTIMAL:
            raise RuntimeError(f"Min-cost flow did not reach optimality (status={status}).")

        flows = np.asarray(smcf.flows(arcs)).reshape(seg_idx.size, ntypes)

        X = np.zeros((nseg, ntypes), dtype=int)
        X[seg_idx] = flows.astype(int)
        free = np.nonzero(~constrained)[0]
        if free.size:
            X[free, np.argmax(scores[free], axis=1)] = 1

        if not (X.sum(axis=1) == 1).all():
            raise RuntimeError("Row constraint violated by the flow solution.")
        if global_quota and not (X.sum(axis=0) == quotas).all():
            raise RuntimeError("Global quota violated by the flow solution.")
        return X

    # ---------- QP relaxation + rounding ----------

    def _solve_mip_scip(self, scores: np.ndarray, quotas: np.ndarray) -> np.ndarray:
        """
        SCIP-based exact MILP solver.
        Mathematical model is STRICTLY equivalent to the Gurobi version.
        """

        if not _SCIP_AVAILABLE:
            raise RuntimeError("SCIP is not available. Please install PySCIPOpt.")

        from pyscipopt import Model, quicksum

        scores = np.asarray(scores, dtype=float)
        quotas = np.asarray(quotas, dtype=int)

        nseg, ntypes = scores.shape
        if quotas.sum() != nseg:
            raise ValueError(f"Invalid quotas: sum(quotas)={quotas.sum()} must equal nseg={nseg}.")

        # --- same internal objects as Gurobi version ---
        A = self._affil_spot2seg_csr.tocsr()  # (S x Nseg)
        V_nz = np.asarray(self._int_ct_ratios_per_spot, dtype=int)  # (S_nz x K)
        nz_mask = np.asarray(self._nonzero_spot_mask, dtype=bool)  # (S,)

        # ---- feasibility checks (IDENTICAL semantics) ----
        cover_per_seg = np.asarray(A.sum(axis=0)).ravel()
        if np.any(cover_per_seg > 1):
            logger.warning(f"{np.sum(cover_per_seg > 1)} segments are covered by multiple spots.")

        spot_seg_counts = np.asarray(A.sum(axis=1)).ravel().astype(int)
        s_nz = 0
        total_covered = 0
        for s in range(A.shape[0]):
            if not nz_mask[s]:
                continue
            seg_ids = A.indices[A.indptr[s] : A.indptr[s + 1]]
            c = len(seg_ids)
            if c != int(V_nz[s_nz].sum()):
                raise ValueError(f"Spot {s}: segments={c} but sum(V)={int(V_nz[s_nz].sum())}.")
            total_covered += c
            s_nz += 1

        V_sum_k = V_nz.sum(axis=0)
        if np.any(V_sum_k > quotas):
            bad = np.where(V_sum_k > quotas)[0]
            raise ValueError(
                "Infeasible: per-spot quotas exceed global quotas for some types: "
                + ", ".join(f"{self.cell_types[k]}: {V_sum_k[k]} > {quotas[k]}" for k in bad)
            )

        # ---- build SCIP model ----
        model = Model("CellTypeAssign_with_spot_quotas")
        # Diagnostics only -- neither knob changes the model or its optimum.
        # PANOSPACE_SCIP_VERBLEVEL: 0 = silent (upstream), 4 = SCIP's progress
        #   table (node count, primal/dual bound, gap), 5 = full.
        # PANOSPACE_SCIP_TIMELIMIT: seconds; SCIP then stops without proving
        #   optimality, and the caller raises rather than returning a
        #   suboptimal assignment.
        verblevel = int(os.environ.get("PANOSPACE_SCIP_VERBLEVEL", 0))
        model.setParam("display/verblevel", verblevel)
        if verblevel:
            model.setParam("display/freq", 100)
        time_limit = float(os.environ.get("PANOSPACE_SCIP_TIMELIMIT", 0) or 0)
        if time_limit > 0:
            model.setParam("limits/time", time_limit)
            logger.info("SCIP time limit set to %.0f s", time_limit)

        # Variables
        logger.info("SCIP: declaring %d x %d = %d binary variables...", nseg, ntypes, nseg * ntypes)
        t_build = time.time()
        X = {}
        for i in range(nseg):
            for k in range(ntypes):
                X[i, k] = model.addVar(vtype="B", name=f"X_{i}_{k}")

        # Objective
        model.setObjective(
            quicksum(scores[i, k] * X[i, k] for i in range(nseg) for k in range(ntypes)), sense="maximize"
        )

        # (1) Each segment chooses exactly one type
        for i in range(nseg):
            model.addCons(quicksum(X[i, k] for k in range(ntypes)) == 1, name=f"row_{i}")

        # (2) Global quotas
        for k in range(ntypes):
            model.addCons(quicksum(X[i, k] for i in range(nseg)) == int(quotas[k]), name=f"global_quota_{k}")

        # (3) Spot-level quotas (HARD constraints)
        s_nz = 0
        for s in range(A.shape[0]):
            if not nz_mask[s]:
                continue
            seg_ids = A.indices[A.indptr[s] : A.indptr[s + 1]]
            for k in range(ntypes):
                model.addCons(quicksum(X[i, k] for i in seg_ids) == int(V_nz[s_nz, k]), name=f"spot_{s}_type_{k}")
            s_nz += 1

        # Solve
        logger.info("SCIP: model built in %.1f s (%d constraints)", time.time() - t_build, model.getNConss())
        logger.info("SCIP optimization started...")
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        logger.info(f"SCIP optimization completed in {solve_time:.2f} seconds ({solve_time/60:.2f} minutes)")

        status = model.getStatus()
        logger.info(
            "SCIP status=%s  nodes=%s  primal=%s  dual=%s  gap=%s",
            status,
            model.getNNodes(),
            model.getPrimalbound(),
            model.getDualbound(),
            model.getGap(),
        )
        if status != "optimal":
            raise RuntimeError(
                f"SCIP MILP not optimal. status={status}. "
                f"Re-run with PANOSPACE_SCIP_VERBLEVEL=4 to watch its progress, "
                f"or with --solver flow for the exact min-cost-flow reformulation."
            )

        # Extract solution
        X_sol = np.zeros((nseg, ntypes), dtype=int)
        for i in range(nseg):
            for k in range(ntypes):
                X_sol[i, k] = int(round(model.getVal(X[i, k])))

        # Sanity checks (same as Gurobi)
        if not (X_sol.sum(axis=1) == 1).all():
            raise RuntimeError("Row constraint violated.")
        if not (X_sol.sum(axis=0) == quotas).all():
            raise RuntimeError("Global quota violated.")

        return X_sol

    @staticmethod
    def _repair_quotas(hard_X: np.ndarray, quotas: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """
        Adjust a one-hot assignment matrix so that column sums equal `quotas`.

        Strategy:
          • For any overfilled column, release the lowest marginal-loss items.
          • Reassign released items to underfilled columns using their best available scores.
        """
        nseg, ntypes = hard_X.shape
        col_sums = hard_X.sum(axis=0).astype(int)
        quotas = quotas.astype(int)

        surplus = col_sums - quotas
        deficit = quotas - col_sums

        # Release phase: remove items from overfilled columns with smallest loss
        release_pool = []  # entries are (loss, i, old_k)
        for k in np.where(surplus > 0)[0]:
            idx = np.where(hard_X[:, k] == 1)[0]
            if idx.size == 0:
                continue
            # For each i, find best alternative score (excluding k)
            alt_scores = scores[idx, :].copy()
            alt_scores[:, k] = -np.inf
            best_alt = np.max(alt_scores, axis=1)
            loss = scores[idx, k] - best_alt
            # Take surplus[k] items with smallest loss
            take = surplus[k]
            if take > 0 and idx.size > 0:
                pick = np.argpartition(loss, take - 1)[:take]
                for ii in pick:
                    i = idx[ii]
                    release_pool.append((loss[ii], i, k))
                hard_X[idx[pick], k] = 0  # release
        # Update deficits after release
        col_sums = hard_X.sum(axis=0).astype(int)
        deficit = quotas - col_sums

        # Reassignment phase: assign released items to underfilled columns
        release_pool.sort(key=lambda t: t[0])
        for _, i, _oldk in release_pool:
            need_ks = np.where(deficit > 0)[0]
            if need_ks.size == 0:
                # No remaining deficit: choose best column for this row
                new_k = np.argmax(scores[i, :])
            else:
                # Among deficit columns, choose the best-scoring one
                k_best = need_ks[np.argmax(scores[i, need_ks])]
                new_k = int(k_best)
            hard_X[i, new_k] = 1
            deficit[new_k] -= 1

        # If some deficit remains (rare), assign free rows as a fallback
        if np.any(deficit > 0):
            free_rows = np.where(hard_X.sum(axis=1) == 0)[0]
            for k in np.where(deficit > 0)[0]:
                need = deficit[k]
                if need <= 0:
                    continue
                if free_rows.size > 0:
                    take = min(len(free_rows), need)
                    take_rows = free_rows[:take]
                    hard_X[take_rows, k] = 1
                    free_rows = free_rows[take:]
                    deficit[k] -= take

        return hard_X

    # ---------- Gurobi MILP (exact 0/1 assignment with row=1, col=quota) ----------
    def _solve_mip(self, scores: np.ndarray, quotas: np.ndarray, global_quota=True, spot_quota=True) -> np.ndarray:
        if not _GUROBI_AVAILABLE:
            raise RuntimeError("Gurobi not available, cannot run MILP.")

        scores = np.asarray(scores, dtype=float)
        quotas = np.asarray(quotas, dtype=int)

        nseg, ntypes = scores.shape
        if quotas.sum() != nseg:
            raise ValueError(f"Invalid quotas: sum(quotas)={quotas.sum()} must equal nseg={nseg}.")

        A = self._affil_spot2seg_csr.tocsr()  # (S x Nseg)
        V_nz = np.asarray(self._int_ct_ratios_per_spot, dtype=int)  # (S_nz x K)
        nz_mask = np.asarray(self._nonzero_spot_mask, dtype=bool)  # (S,)

        # ---- Check overlap (optional but recommended) ----
        cover_per_seg = np.asarray(A.sum(axis=0)).ravel()
        if np.any(cover_per_seg > 1):
            # Could raise error instead, or use logging.warning
            logger.warning(
                f"{np.sum(cover_per_seg>1)} segments are covered by multiple spots. "
                f"Spot-level equality constraints may be infeasible/too rigid."
            )

        # ---- Per-spot consistency: |seg(s)| == sum_k V[s,k] ----
        spot_seg_counts = np.asarray(A.sum(axis=1)).ravel().astype(int)  # (S,)
        s_nz = 0
        total_covered = 0
        for s in range(A.shape[0]):
            if not nz_mask[s]:
                continue
            seg_ids = A.indices[A.indptr[s] : A.indptr[s + 1]]
            c = len(seg_ids)
            if c != spot_seg_counts[s]:
                raise ValueError("Internal error: CSR count mismatch.")
            if c != int(V_nz[s_nz].sum()):
                raise ValueError(f"Spot {s} mismatch: segments_in_spot={c} but sum(V)={int(V_nz[s_nz].sum())}.")
            total_covered += c
            s_nz += 1

        # ---- Global-vs-spot feasibility: sum_s V[s,k] <= quotas[k] ----
        V_sum_k = V_nz.sum(axis=0)  # (K,)
        if np.any(V_sum_k > quotas):
            bad = np.where(V_sum_k > quotas)[0]
            raise ValueError(
                "Infeasible: per-spot quotas exceed global quotas for some cell types: "
                + ", ".join([f"{self.cell_types[k]}: Vsum={V_sum_k[k]} > quota={quotas[k]}" for k in bad])
            )

        # ---- uncovered sanity ----
        uncovered = nseg - total_covered
        if uncovered < 0:
            raise ValueError(f"Invalid: total covered segments {total_covered} exceeds nseg {nseg} (overlap likely).")
        remain = int((quotas - V_sum_k).sum())
        if remain != uncovered:
            logger.warning(
                f"uncovered segments={uncovered} but remaining global quota sum={remain}. "
                f"This can happen with overlap or filtering differences; check membership construction."
            )

        # ---- Build MILP ----
        model = gp.Model("CellTypeAssign_with_spot_quotas")
        model.setParam("OutputFlag", 0)

        X = model.addVars(nseg, ntypes, vtype=GRB.BINARY, name="X")

        model.setObjective(
            gp.quicksum(scores[i, k] * X[i, k] for i in range(nseg) for k in range(ntypes)), GRB.MAXIMIZE
        )

        # Each segment chooses exactly one type
        for i in range(nseg):
            model.addConstr(gp.quicksum(X[i, k] for k in range(ntypes)) == 1)

        if global_quota:
            # Global quotas
            for k in range(ntypes):
                model.addConstr(gp.quicksum(X[i, k] for i in range(nseg)) == int(quotas[k]))

        if spot_quota:
            # Spot-level quotas for covered segments
            s_nz = 0
            for s in range(A.shape[0]):
                if not nz_mask[s]:
                    continue
                seg_ids = A.indices[A.indptr[s] : A.indptr[s + 1]]
                for k in range(ntypes):
                    model.addConstr(gp.quicksum(X[i, k] for i in seg_ids) == int(V_nz[s_nz, k]))
                s_nz += 1

        logger.info("Gurobi optimization started...")
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        logger.info(f"Gurobi optimization completed in {solve_time:.2f} seconds ({solve_time/60:.2f} minutes)")

        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"MILP not optimal. status={model.status}")

        return np.array([[int(X[i, k].X) for k in range(ntypes)] for i in range(nseg)], dtype=int)
